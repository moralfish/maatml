"""Ingest external JSONL into seed corpus."""
from __future__ import annotations

import json
from pathlib import Path

from maatml.config import ModelDefinition
from maatml.data.ingest import ingest_samples
from maatml.utils.io import iter_jsonl, write_jsonl


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
