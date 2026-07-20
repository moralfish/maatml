"""Family-aware group split: members of one family must not straddle splits."""
from __future__ import annotations

from pathlib import Path

from flow_ml.config import load_model_def
from flow_ml.data.pipeline import prepare
from flow_ml.utils.io import iter_jsonl, write_jsonl


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
