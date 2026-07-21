"""Unit tests for the append-only run registry."""
from __future__ import annotations

from pathlib import Path

import pytest

from maatml.config import ModelDefinition
from maatml.runs import (
    finish_run,
    latest_completed_run,
    list_runs,
    resolve_checkpoint,
    start_run,
)


def _md(tmp_path: Path) -> ModelDefinition:
    md = ModelDefinition(
        name="run-test",
        model_id="run-test",
        architecture="causal_sft",
        version="0.1.0",
    )
    object.__setattr__(md, "model_dir", tmp_path)
    return md


def test_start_finish_list_latest(tmp_path: Path) -> None:
    md = _md(tmp_path)
    r1 = start_run(md, smoke=True, device="cpu", profile="cpu")
    assert r1.status == "running"
    assert Path(r1.out_dir).is_dir()
    assert (tmp_path / "output" / "runs.jsonl").is_file()

    finish_run(md, r1.run_id, "completed", metrics={"loss": 0.5})
    runs = list_runs(md)
    assert len(runs) == 1
    assert runs[0].status == "completed"
    assert runs[0].metrics == {"loss": 0.5}

    r2 = start_run(md, smoke=False, device="mps", profile="mps")
    finish_run(md, r2.run_id, "aborted", error="boom")
    latest = latest_completed_run(md)
    assert latest is not None
    assert latest.run_id == r1.run_id


def test_resolve_checkpoint_by_run_id_and_path(tmp_path: Path) -> None:
    md = _md(tmp_path)
    rec = start_run(md)
    finish_run(md, rec.run_id, "completed")

    by_id = resolve_checkpoint(md, rec.run_id)
    assert by_id == Path(rec.out_dir)

    by_path = resolve_checkpoint(md, rec.out_dir)
    assert by_path == Path(rec.out_dir).resolve()

    default = resolve_checkpoint(md, None)
    assert default == Path(rec.out_dir)


def test_resolve_checkpoint_legacy_mtime(tmp_path: Path) -> None:
    md = _md(tmp_path)
    ckpt = md.checkpoints_dir / "legacy@version"
    ckpt.mkdir(parents=True)
    (ckpt / "model.safetensors").write_text("x", encoding="utf-8")
    resolved = resolve_checkpoint(md, None)
    assert resolved == ckpt


def test_finish_unknown_run_raises(tmp_path: Path) -> None:
    md = _md(tmp_path)
    with pytest.raises(KeyError):
        finish_run(md, "missing-id", "completed")
