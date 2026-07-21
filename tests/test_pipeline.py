"""Pipeline tests for generic prepare against ModelDefinition.

Uses the real example model folders (plugins register sanitizers).
"""
from __future__ import annotations

from pathlib import Path

from maatml.config import load_model_def
from maatml.data.pipeline import prepare
from maatml.registry import discover_plugins
from maatml.utils.io import iter_jsonl

REPO_ROOT = Path(__file__).resolve().parents[1]
JCL_DIR = REPO_ROOT / "examples" / "jcl-validator"
SPOOL_DIR = REPO_ROOT / "examples" / "spool-interpreter"


def test_prepare_jcl_writes_splits(tmp_path: Path) -> None:
    discover_plugins(force=True)
    # Copy minimal example into tmp so prepare writes under tmp.
    import shutil

    mdir = tmp_path / "jcl-validator"
    shutil.copytree(
        JCL_DIR,
        mdir,
        ignore=shutil.ignore_patterns("output", "__pycache__", "*.pyc", ".DS_Store"),
    )
    md = load_model_def(mdir)
    seeds = mdir / "datasets" / "samples" / "seed_samples.jsonl"
    seed_count = sum(1 for line in seeds.read_text().splitlines() if line.strip())
    summary = prepare(md)
    total = sum(summary["split_counts"].values())
    assert total >= seed_count
    for split in ("train", "val", "test"):
        path = md.prepared_dir / f"{split}.jsonl"
        assert path.exists()
        rows = list(iter_jsonl(path))
        assert len(rows) == summary["split_counts"][split]
        for row in rows:
            assert row["request"]
            assert row["expected_validation_result"]
            assert row["category"]
    assert (md.prepared_dir / "dataset_card.md").exists()


def test_prepare_spool_writes_splits(tmp_path: Path) -> None:
    discover_plugins(force=True)
    import shutil

    mdir = tmp_path / "spool-interpreter"
    shutil.copytree(
        SPOOL_DIR,
        mdir,
        ignore=shutil.ignore_patterns("output", "__pycache__", "*.pyc", ".DS_Store"),
    )
    md = load_model_def(mdir)
    seeds = mdir / "datasets" / "samples" / "seed_samples.jsonl"
    seed_count = sum(1 for line in seeds.read_text().splitlines() if line.strip())
    summary = prepare(md)
    total = sum(summary["split_counts"].values())
    assert total >= seed_count
    for split in ("train", "val", "test"):
        path = md.prepared_dir / f"{split}.jsonl"
        assert path.exists()
        rows = list(iter_jsonl(path))
        assert len(rows) == summary["split_counts"][split]
        for row in rows:
            assert row["request"]
            assert row["expected_interpretation"]
            assert row["category"]
