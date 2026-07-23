"""Datagen gating and data-safety: fail-closed validator, no seed truncation."""
from __future__ import annotations

import pytest

from maatml.config import ModelDefinition
from maatml.data import datagen as datagen_mod
from maatml.data.datagen import DatagenConfigError, _default_validate_fn, run_datagen
from maatml.registry import GENERATORS
from maatml.utils.io import iter_jsonl, write_jsonl_atomic


def _md(tmp_path, *, evaluation=None, generator="dummy_gen"):
    md = ModelDefinition(
        name="t",
        model_id="t",
        dataset={"generator": generator, "seed_samples": "seeds.jsonl"},
        evaluation=evaluation or {},
    )
    object.__setattr__(md, "model_dir", tmp_path)
    return md


def _register_dummy_generator(name="dummy_gen"):
    def _factory(model_def, *, seed=0):
        state = {"n": 0}

        def _gen():
            state["n"] += 1
            return {
                "request": f"r{state['n']}",
                "target": {"x": state["n"]},
                "sample_id": f"s{state['n']}",
            }

        return _gen

    GENERATORS.register(name, _factory, source="test")


# --- _default_validate_fn (G1 fail-closed, G6 unresolvable) ------------------


def test_missing_validator_raises_fail_closed():
    md = ModelDefinition(name="t", model_id="t")
    with pytest.raises(DatagenConfigError):
        _default_validate_fn(md)


def test_missing_validator_allow_ungated_returns_accept_all():
    md = ModelDefinition(name="t", model_id="t")
    fn = _default_validate_fn(md, allow_ungated=True)
    assert fn({"anything": 1}) is True


def test_unresolvable_validator_raises_even_with_allow_ungated():
    md = ModelDefinition(name="t", model_id="t", evaluation={"validator": "nope_missing"})
    with pytest.raises(DatagenConfigError) as exc:
        _default_validate_fn(md)
    assert "nope_missing" in str(exc.value)
    # allow_ungated must NOT bypass a set-but-unresolvable validator
    with pytest.raises(DatagenConfigError):
        _default_validate_fn(md, allow_ungated=True)


def test_datagen_config_error_is_value_error():
    assert issubclass(DatagenConfigError, ValueError)


# --- run_datagen fail-closed ------------------------------------------------


def test_run_datagen_fails_closed_before_generating(tmp_path):
    _register_dummy_generator()
    md = _md(tmp_path, evaluation={})  # no validator
    with pytest.raises(DatagenConfigError):
        run_datagen(md, target=1, max_attempts=3)


# --- D2: no seed truncation on zero-accept ----------------------------------


def test_no_truncation_when_nothing_accepted(tmp_path, monkeypatch):
    _register_dummy_generator()
    seed = tmp_path / "seeds.jsonl"
    write_jsonl_atomic(seed, [{"keep": 1}, {"keep": 2}])
    monkeypatch.setattr(datagen_mod, "_default_validate_fn", lambda md, **k: (lambda r: False))

    md = _md(tmp_path, evaluation={})
    result = run_datagen(md, target=1, append=False, max_attempts=3)

    assert result["seed_written"] is False
    assert result["protected_existing"] is True
    assert list(iter_jsonl(seed)) == [{"keep": 1}, {"keep": 2}]  # untouched


def test_accepted_rows_written_and_gated_card(tmp_path, monkeypatch):
    _register_dummy_generator()
    monkeypatch.setattr(datagen_mod, "_default_validate_fn", lambda md, **k: (lambda r: True))

    md = _md(tmp_path, evaluation={"validator": "some_validator"})
    result = run_datagen(md, target=2, append=False, max_attempts=10)

    assert result["seed_written"] is True
    assert result["gated"] is True
    assert result["accepted"] == 2
    assert len(list(iter_jsonl(tmp_path / "seeds.jsonl"))) == 2
    card = (tmp_path / "seeds.datagen_card.md").read_text()
    assert "status: GATED" in card


def test_ungated_run_marks_card_ungated(tmp_path):
    _register_dummy_generator()
    md = _md(tmp_path, evaluation={})
    result = run_datagen(md, target=2, append=False, allow_ungated=True, max_attempts=10)

    assert result["gated"] is False
    assert result["validator"] is None
    card = (tmp_path / "seeds.datagen_card.md").read_text()
    assert "UNGATED" in card


# --- write_jsonl_atomic ------------------------------------------------------


def test_atomic_write_roundtrip_no_temp_left(tmp_path):
    dest = tmp_path / "a.jsonl"
    write_jsonl_atomic(dest, [{"x": 1}, {"y": 2}])
    assert list(iter_jsonl(dest)) == [{"x": 1}, {"y": 2}]
    leftovers = [p.name for p in tmp_path.iterdir() if p.suffix == ".tmp"]
    assert leftovers == []


def test_atomic_write_preserves_original_on_failure(tmp_path):
    dest = tmp_path / "a.jsonl"
    write_jsonl_atomic(dest, [{"orig": 1}])

    def _rows():
        yield {"ok": 1}
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError):
        write_jsonl_atomic(dest, _rows())
    assert list(iter_jsonl(dest)) == [{"orig": 1}]  # unchanged
    assert [p.name for p in tmp_path.iterdir() if p.suffix == ".tmp"] == []
