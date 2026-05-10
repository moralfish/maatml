"""Pipeline tests for prepare_jcl / prepare_spool against ModelDefinition.

Each test builds a minimal `models/<name>/` folder under tmp_path and writes a
synthetic `model.yml` pointing at the real on-disk dataset assets, then runs
the prepare_* function.  This mirrors how the CLI invokes the pipeline.
"""
from __future__ import annotations

import shutil
from pathlib import Path

from flow_ml.config import load_model_def
from flow_ml.data.pipeline import prepare_jcl, prepare_spool
from flow_ml.utils.io import iter_jsonl

REPO_ROOT = Path(__file__).resolve().parents[1]
JCL_SEEDS = REPO_ROOT / "models" / "jcl-validator" / "datasets" / "samples" / "seed_samples.jsonl"
SPOOL_SEEDS = REPO_ROOT / "models" / "spool-interpreter" / "datasets" / "samples" / "seed_samples.jsonl"


def _make_model_folder(
    tmp_path: Path,
    model_yml: str,
    *,
    seeds_path: Path | None = None,
) -> Path:
    """Build a tmp `models/foo/` folder containing model.yml and a copy of the
    seed samples file, then return the folder path."""
    mdir = tmp_path / "model"
    mdir.mkdir()
    (mdir / "model.yml").write_text(model_yml, encoding="utf-8")
    (mdir / "datasets" / "samples").mkdir(parents=True)
    if seeds_path is not None:
        shutil.copy2(seeds_path, mdir / "datasets" / "samples" / "seed_samples.jsonl")
    return mdir


def test_prepare_jcl_writes_splits(tmp_path: Path) -> None:
    mdir = _make_model_folder(
        tmp_path,
        """name: foo
model_id: foo:v1
task: jcl_validation
data:
  seed: 7
  seed_samples: datasets/samples/seed_samples.jsonl
  split_ratios: [0.6, 0.2, 0.2]
  raw_field: request
""",
        seeds_path=JCL_SEEDS,
    )
    md = load_model_def(mdir)
    seed_count = sum(
        1 for line in JCL_SEEDS.read_text().splitlines() if line.strip()
    )
    summary = prepare_jcl(md)
    total = sum(summary["split_counts"].values())
    assert total == seed_count
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
    mdir = _make_model_folder(
        tmp_path,
        """name: foo
model_id: foo:v1
task: spool_interpretation
data:
  seed: 11
  seed_samples: datasets/samples/seed_samples.jsonl
  split_ratios: [0.5, 0.25, 0.25]
  raw_field: raw_spool
""",
        seeds_path=SPOOL_SEEDS,
    )
    md = load_model_def(mdir)
    # Count seeds dynamically: this corpus has grown over time as new
    # categories were added (Smart/RESTART pack, etc.). Pinning to a
    # constant breaks every time the seed file is extended.
    seed_count = sum(
        1
        for line in (mdir / "datasets/samples/seed_samples.jsonl").read_text().splitlines()
        if line.strip()
    )
    summary = prepare_spool(md)
    total = sum(summary["split_counts"].values())
    assert total == seed_count
    for split in ("train", "val", "test"):
        path = md.prepared_dir / f"{split}.jsonl"
        assert path.exists()
        rows = list(iter_jsonl(path))
        assert len(rows) == summary["split_counts"][split]
        for row in rows:
            assert row["request"]
            assert row["expected_interpretation"]
            assert row["category"]
            assert "raw_spool" not in row
            assert "sanitized_spool" not in row
