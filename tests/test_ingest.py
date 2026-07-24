"""Ingest external JSONL into seed corpus."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from maatml.config import ModelDefinition
from maatml.data.ingest import ingest_samples
from maatml.registry import VALIDATORS
from maatml.utils.io import iter_jsonl, write_jsonl


def _model(model_dir, *, evaluation=None):
    md = ModelDefinition(
        name="toy",
        model_id="toy",
        architecture="causal_sft",
        dataset={
            "seed_samples": "datasets/samples/seed_samples.jsonl",
            "request_field": "request",
            "target_field": "target",
        },
        evaluation=evaluation or {},
    )
    object.__setattr__(md, "model_dir", model_dir)
    return md


def test_ingest_jsonl_to_seeds(tmp_path: Path) -> None:
    model_dir = tmp_path / "model"
    samples = model_dir / "datasets" / "samples"
    samples.mkdir(parents=True)
    seeds = samples / "seed_samples.jsonl"
    write_jsonl(
        seeds,
        [
            {
                "sample_id": "existing-1",
                "request": "old",
                "target": {"x": 1},
                "source": "seed",
            }
        ],
    )

    inp = tmp_path / "incoming.jsonl"
    write_jsonl(
        inp,
        [
            {"id": "a", "text": "hello", "label": {"y": 1}},
            {"id": "b", "text": "world", "label": {"y": 2}},
            {"id": "a", "text": "dup", "label": {"y": 9}},  # dup after map
        ],
    )

    md = ModelDefinition(
        name="toy",
        model_id="toy",
        version="0.1.0",
        architecture="causal_sft",
        dataset={
            "seed_samples": "datasets/samples/seed_samples.jsonl",
            "request_field": "request",
            "target_field": "target",
        },
    )
    object.__setattr__(md, "model_dir", model_dir)

    result = ingest_samples(
        md,
        inp,
        field_map={"sample_id": "id", "request": "text", "target": "label"},
        append=True,
    )
    assert result["accepted"] == 2
    assert result["rejected"] == 1
    rows = list(iter_jsonl(seeds))
    assert len(rows) == 3  # existing + 2
    assert rows[-1]["source"].startswith("ingest:")
    reject = json.loads(Path(result["reject_path"]).read_text(encoding="utf-8"))
    assert reject["rejected"] == 1


def test_ingest_map_unmatched_source_errors(tmp_path: Path) -> None:
    """G4: a --map source column matching zero input rows is a config error."""
    model_dir = tmp_path / "model"
    (model_dir / "datasets" / "samples").mkdir(parents=True)
    inp = tmp_path / "incoming.jsonl"
    write_jsonl(inp, [{"a": 1, "b": 2}, {"a": 3, "b": 4}])
    md = _model(model_dir)
    with pytest.raises(ValueError, match="zero input rows"):
        ingest_samples(md, inp, field_map={"request": "text"})


def test_ingest_skips_unvalidated_when_validator_configured(tmp_path: Path) -> None:
    """G4: with a validator configured, a row missing gold is counted and
    excluded (skipped_unvalidated), never silently accepted."""

    class _Res:
        ok = True

    VALIDATORS.register("ing_ok", lambda raw, **k: _Res(), source="test")

    model_dir = tmp_path / "model"
    (model_dir / "datasets" / "samples").mkdir(parents=True)
    inp = tmp_path / "incoming.jsonl"
    write_jsonl(
        inp,
        [
            {"sample_id": "a", "request": "x", "target": {"y": 1}},  # has gold
            {"sample_id": "b", "request": "z"},  # missing gold
        ],
    )
    md = _model(model_dir, evaluation={"validator": "ing_ok"})
    result = ingest_samples(md, inp, append=False)
    assert result["accepted"] == 1
    assert result["skipped_unvalidated"] == 1
    assert result["rejected"] == 0
    rows = list(iter_jsonl(md.resolve("datasets/samples/seed_samples.jsonl")))
    assert [r["sample_id"] for r in rows] == ["a"]


def test_ingest_refuses_unapproved_serve_capture(tmp_path: Path) -> None:
    """Exit criterion: a captured row lacking approval is refused, not ingested."""
    model_dir = tmp_path / "model"
    (model_dir / "datasets" / "samples").mkdir(parents=True)
    md = _model(model_dir)

    inp = tmp_path / "capture.jsonl"
    write_jsonl(
        inp,
        [
            # Straight off serve --capture: unreviewed, must be refused.
            {"request": "unreviewed", "target": {"x": 1}, "source": "serve_capture",
             "approved": False, "needs_review": True},
            # A reviewer corrected and approved this one.
            {"request": "reviewed", "target": {"x": 2}, "source": "serve_capture",
             "approved": True, "needs_review": False},
        ],
    )
    result = ingest_samples(md, inp, append=False)

    assert result["unapproved_capture"] == 1
    assert result["accepted"] == 1
    seeds = list(iter_jsonl(md.resolve("datasets/samples/seed_samples.jsonl")))
    assert len(seeds) == 1
    assert seeds[0]["request"] == "reviewed"
    # The approved row sheds its review markers and is provenance-stamped.
    assert "approved" not in seeds[0]
    assert "needs_review" not in seeds[0]
    assert seeds[0]["source"].startswith("ingest:")


def test_ingest_capture_stripped_of_approval_is_still_refused(tmp_path: Path) -> None:
    """A capture row cannot slip through by dropping the approval flag."""
    model_dir = tmp_path / "model"
    (model_dir / "datasets" / "samples").mkdir(parents=True)
    md = _model(model_dir)
    inp = tmp_path / "c.jsonl"
    write_jsonl(inp, [{"request": "x", "target": {"a": 1}, "source": "serve_capture"}])
    result = ingest_samples(md, inp, append=False)
    assert result["unapproved_capture"] == 1
    assert result["accepted"] == 0
