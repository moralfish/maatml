"""Family-aware group split: members of one family must not straddle splits."""
from __future__ import annotations

from pathlib import Path

from maatml.config import load_model_def
from maatml.data.pipeline import _group_key, prepare
from maatml.utils.io import iter_jsonl, write_jsonl


def test_family_group_does_not_straddle(tmp_path: Path) -> None:
    mdir = tmp_path / "model"
    (mdir / "datasets" / "samples").mkdir(parents=True)
    seeds = mdir / "datasets" / "samples" / "seed_samples.jsonl"
    # Two families, three samples each — enough that a per-sample hash
    # would often split a family, but group-hash must keep them together.
    rows = []
    for family in ("alpha", "beta", "gamma", "delta"):
        for i in range(4):
            rows.append(
                {
                    "sample_id": f"{family}-{i}",
                    "source": "hand",
                    "family": family,
                    "category": family,
                    "request": f"req {family} {i}",
                    "expected_output": {"answer": f"{family}-{i}"},
                }
            )
    write_jsonl(seeds, rows)
    (mdir / "model.yml").write_text(
        """name: group-split-test
model_id: group-split-test
task: support_ticket_triage
architecture: causal_sft
version: 0.1.0
dataset:
  seed_samples: datasets/samples/seed_samples.jsonl
  split_ratios: [0.5, 0.25, 0.25]
  target_field: expected_output
""",
        encoding="utf-8",
    )

    md = load_model_def(mdir)
    prepare(md)

    family_to_splits: dict[str, set[str]] = {}
    for split in ("train", "val", "test"):
        for row in iter_jsonl(md.prepared_dir / f"{split}.jsonl"):
            fam = row["family"]
            family_to_splits.setdefault(fam, set()).add(split)

    for fam, splits in family_to_splits.items():
        assert len(splits) == 1, f"family {fam!r} straddled splits: {splits}"


def test_group_by_config_prefers_field() -> None:
    row = {"family": "fam", "source": "src", "sample_id": "sid", "bucket": "b1"}
    assert _group_key(row, group_by="bucket") == "bucket:b1"
    # Missing preferred field → fall back to family chain
    assert _group_key({"family": "fam"}, group_by="bucket") == "family:fam"
    # Empty / null group_by keeps default chain
    assert _group_key(row, group_by=None) == "family:fam"
    assert _group_key(row, group_by="") == "family:fam"


def test_group_by_config_in_prepare(tmp_path: Path) -> None:
    mdir = tmp_path / "model"
    (mdir / "datasets" / "samples").mkdir(parents=True)
    seeds = mdir / "datasets" / "samples" / "seed_samples.jsonl"
    rows = []
    for bucket in ("x", "y"):
        for i in range(3):
            rows.append(
                {
                    "sample_id": f"{bucket}-{i}",
                    "bucket": bucket,
                    "family": "shared",  # would merge all if family used
                    "request": f"r {bucket} {i}",
                    "expected_output": {"a": i},
                }
            )
    write_jsonl(seeds, rows)
    (mdir / "model.yml").write_text(
        """name: group-by-cfg
model_id: group-by-cfg
architecture: causal_sft
version: 0.1.0
dataset:
  seed_samples: datasets/samples/seed_samples.jsonl
  split_ratios: [0.5, 0.25, 0.25]
  group_by: bucket
""",
        encoding="utf-8",
    )
    md = load_model_def(mdir)
    prepare(md)
    bucket_to_splits: dict[str, set[str]] = {}
    for split in ("train", "val", "test"):
        path = md.prepared_dir / f"{split}.jsonl"
        if not path.exists():
            continue
        for row in iter_jsonl(path):
            bucket_to_splits.setdefault(row["bucket"], set()).add(split)
    for b, splits in bucket_to_splits.items():
        assert len(splits) == 1, f"bucket {b!r} straddled: {splits}"
