"""maatml distill: validator gating, provenance, and offline replay.

A fake in-process teacher stands in for the OpenAI-compatible client, so the
gating and record/replay behaviour is tested without a network.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from maatml.config import load_model_def
from maatml.data import distill as distill_mod
from maatml.data.distill import (
    DistillConfig,
    DistillConfigError,
    TeacherCache,
    _prompt_hash,
    run_distill,
)
from maatml.registry import register_validator
from maatml.utils.io import iter_jsonl, write_jsonl
from maatml.validation.base import ValidationError, ValidationResult

PROMPTS = ["good prompt one", "good prompt two", "bad prompt"]


@register_validator("distill_test_validator")
def _validator(raw_output, *, schema_path=None, contracts_path=None, user_prompt=None, **_kw):
    """Accepts any JSON object whose 'ok' is True."""
    result = ValidationResult(raw_output=raw_output, n_layers=1, required_layers={1})
    try:
        parsed = json.loads(raw_output)
    except (json.JSONDecodeError, TypeError):
        result.errors.append(ValidationError(layer=1, code="bad_json", message="x"))
        return result
    result.parsed = parsed
    if isinstance(parsed, dict) and parsed.get("ok") is True:
        result.passed_layers.add(1)
    else:
        result.errors.append(ValidationError(layer=1, code="not_ok", message="ok!=True"))
    return result


class _FakeTeacher:
    """Returns a canned label per prompt; the 'bad' prompt yields a failing one."""

    def __init__(self, *args, **kwargs) -> None:
        _FakeTeacher.calls = getattr(_FakeTeacher, "calls", 0)

    def chat_completions(self, messages, **kwargs):
        type(self).calls = getattr(type(self), "calls", 0) + 1
        prompt = messages[-1]["content"]
        if "bad" in prompt:
            return json.dumps({"ok": False, "why": "wrong"})
        return json.dumps({"ok": True, "label": prompt.upper()})


def _model(tmp_path: Path, *, distill_section: dict | None = None):
    mdir = tmp_path / "model"
    (mdir / "datasets").mkdir(parents=True, exist_ok=True)
    write_jsonl(mdir / "datasets" / "prompts.jsonl", [{"request": p} for p in PROMPTS])
    section = distill_section or {
        "prompt_source": "datasets/prompts.jsonl",
        "teacher_model": "fake",
        "teacher_revision": "v1",
        "cache": "datasets/cache.jsonl",
    }
    body = {
        "name": "distill-test",
        "model_id": "distill-test",
        "architecture": "causal_sft",
        "version": "0.1.0",
        "dataset": {
            "seed_samples": "datasets/samples/seed_samples.jsonl",
            "target_field": "expected_output",
        },
        "evaluation": {"validator": "distill_test_validator"},
        "distill": section,
    }
    import yaml

    (mdir / "model.yml").write_text(yaml.safe_dump(body), encoding="utf-8")
    return load_model_def(mdir)


@pytest.fixture
def teacher(monkeypatch):
    _FakeTeacher.calls = 0
    monkeypatch.setattr(distill_mod, "TeacherClient", _FakeTeacher)
    return _FakeTeacher


# --- gating ----------------------------------------------------------------


def test_rejected_teacher_row_is_absent_from_the_corpus(tmp_path, teacher) -> None:
    """Exit criterion: a validator-rejected teacher label never enters seeds."""
    md = _model(tmp_path)
    summary = run_distill(md)

    assert summary["accepted"] == 2
    assert summary["rejected"] == 1
    rows = list(iter_jsonl(md.resolve("datasets/samples/seed_samples.jsonl")))
    assert len(rows) == 2
    # The bad prompt's label (ok:false) is nowhere in the corpus.
    assert all(row["expected_output"]["ok"] is True for row in rows)
    assert all("bad" not in row["request"] for row in rows)

    reject_path = md.resolve("datasets/samples/seed_samples.distill_rejected.jsonl")
    rejects = list(iter_jsonl(reject_path))
    assert rejects[0]["_reject_reason"] == "invalid_target"


def test_accepted_rows_carry_provenance(tmp_path, teacher) -> None:
    md = _model(tmp_path)
    run_distill(md)
    row = next(iter_jsonl(md.resolve("datasets/samples/seed_samples.jsonl")))
    prov = row["provenance"]
    assert prov["teacher_model"] == "fake"
    assert prov["teacher_revision"] == "v1"
    assert prov["prompt_sha256"] == row["family"].split(":")[1]
    assert row["source"] == "distill"


def test_distill_requires_a_validator(tmp_path, teacher) -> None:
    md = _model(tmp_path)
    md.evaluation.pop("validator")
    with pytest.raises(DistillConfigError, match="requires evaluation.validator"):
        run_distill(md)


# --- record / replay -------------------------------------------------------


def test_live_run_records_a_cache_that_replays_offline(tmp_path, teacher) -> None:
    """Exit criterion: replay with no network reproduces the accepted corpus."""
    md = _model(tmp_path)
    live = run_distill(md)
    assert teacher.calls == len(PROMPTS)
    cache_path = md.resolve("datasets/cache.jsonl")
    assert cache_path.is_file()

    # Wipe the produced corpus, then replay from the cache with the teacher
    # replaced by one that raises if it is ever called.
    md.resolve("datasets/samples/seed_samples.jsonl").unlink()

    class _NoNetwork:
        def __init__(self, *a, **k):
            raise AssertionError("replay must not construct a teacher")

    import maatml.data.distill as mod

    mod.TeacherClient = _NoNetwork
    try:
        replayed = run_distill(md, replay=True)
    finally:
        mod.TeacherClient = _FakeTeacher

    assert replayed["accepted"] == live["accepted"] == 2
    assert replayed["cache_hits"] == len(PROMPTS)
    replay_rows = list(iter_jsonl(md.resolve("datasets/samples/seed_samples.jsonl")))
    assert len(replay_rows) == 2


def test_offline_run_skips_uncached_prompts(tmp_path) -> None:
    md = _model(tmp_path)
    # No cache exists and no teacher is available: every prompt is a cache miss.
    class _NoNetwork:
        def __init__(self, *a, **k):
            raise AssertionError("offline must not construct a teacher")

    import maatml.data.distill as mod

    mod.TeacherClient = _NoNetwork
    try:
        summary = run_distill(md, offline=True)
    finally:
        mod.TeacherClient = _FakeTeacher

    assert summary["accepted"] == 0
    assert summary["cache_misses"] == len(PROMPTS)
    assert not md.resolve("datasets/samples/seed_samples.jsonl").is_file()


def test_a_different_teacher_does_not_reuse_cached_labels(tmp_path, teacher) -> None:
    md = _model(tmp_path)
    run_distill(md)  # records under fake@v1

    # A different revision is a different cache key, so replay finds nothing.
    md.distill["teacher_revision"] = "v2"
    md.resolve("datasets/samples/seed_samples.jsonl").unlink()
    summary = run_distill(md, replay=True)
    assert summary["cache_hits"] == 0
    assert summary["cache_misses"] == len(PROMPTS)


def test_second_live_run_dedups(tmp_path, teacher) -> None:
    md = _model(tmp_path)
    run_distill(md)
    again = run_distill(md)
    assert again["duplicates"] == 2
    assert again["accepted"] == 0
    assert len(list(iter_jsonl(md.resolve("datasets/samples/seed_samples.jsonl")))) == 2


# --- prompt loading + config ----------------------------------------------


def test_load_prompts_from_jsonl_and_text(tmp_path) -> None:
    from maatml.data.distill import load_prompts

    (tmp_path / "p.jsonl").write_text(
        json.dumps({"request": "a"}) + "\n" + json.dumps({"prompt": "b"}) + "\n",
        encoding="utf-8",
    )
    assert load_prompts(tmp_path / "p.jsonl", "request") == ["a", "b"]
    (tmp_path / "p.txt").write_text("line one\nline two\n\n", encoding="utf-8")
    assert load_prompts(tmp_path / "p.txt", "request") == ["line one", "line two"]


def test_cache_key_binds_prompt_and_teacher() -> None:
    a = TeacherCache.key(_prompt_hash("x"), "m", "r1")
    b = TeacherCache.key(_prompt_hash("x"), "m", "r2")
    assert a != b


def test_config_rejects_unknown_keys() -> None:
    with pytest.raises(Exception):
        DistillConfig(prompt_source="p.jsonl", teecher_model="typo")


# --- the shipped triage example --------------------------------------------

REPO = Path(__file__).resolve().parents[1]
TRIAGE = REPO / "examples" / "support-ticket-triage"


def test_triage_distill_replays_offline_from_the_shipped_cache(tmp_path, monkeypatch) -> None:
    """The example replays with no network and rejects the wrong-routing label.

    Mirrors the CI exit criterion without touching the committed corpus: the
    prompt pool and cache ship in the repo; output goes to a temp seed file.
    """
    import sys

    sys.path.insert(0, str(TRIAGE))
    from maatml.registry import discover_plugins, load_model_plugins

    # force=True: the autouse registry-restore fixture may have dropped the
    # triage validator that an earlier test loaded, while load_model_plugins
    # still records the module as imported.
    discover_plugins()
    load_model_plugins(TRIAGE, ["./triage_plugin"], force=True)

    class _NoNetwork:
        def __init__(self, *a, **k):
            raise AssertionError("the shipped replay must not hit the network")

    monkeypatch.setattr(distill_mod, "TeacherClient", _NoNetwork)

    md = load_model_def(TRIAGE)
    out = tmp_path / "seeds.jsonl"
    summary = run_distill(md, replay=True, append=False, out_path=str(out))

    # Four prompts, three valid labels, one rejected by the routing contract.
    assert summary["prompts"] == 4
    assert summary["accepted"] == 3
    assert summary["rejected"] == 1
    assert summary["cache_hits"] == 4

    rows = list(iter_jsonl(out))
    teams = {r["expected_output"]["team"] for r in rows}
    assert teams == {"payments", "identity", "docs"}
    # The billing->platform row violates the routing contract and is absent.
    assert "platform" not in teams
    for row in rows:
        assert row["provenance"]["teacher_model"] == "recorded-teacher"
