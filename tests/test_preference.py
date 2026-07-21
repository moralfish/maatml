"""Preference JSONL normalize + mint_preference_pairs (no TRL / torch)."""
from __future__ import annotations

from pathlib import Path

from maatml.config import load_model_def
from maatml.data.preference import (
    mint_preference_pairs,
    normalize_preference,
    prepare_preference_jsonl,
)
from maatml.registry import FORMATS, discover_plugins
from maatml.utils.io import write_jsonl


def test_normalize_preference() -> None:
    row = {
        "prompt": "Q?",
        "chosen": "good",
        "rejected": "bad",
        "sample_id": "p1",
        "family": "f",
    }
    out = normalize_preference(row)
    assert out == {
        "prompt": "Q?",
        "chosen": "good",
        "rejected": "bad",
        "sample_id": "p1",
        "family": "f",
    }


def test_normalize_preference_aliases() -> None:
    out = normalize_preference(
        {"request": "hi", "chosen": "a", "rejected": "b", "id": "x"}
    )
    assert out["prompt"] == "hi"
    assert out["sample_id"] == "x"


def test_mint_preference_pairs_with_list_candidates() -> None:
    def validator(prompt: str, completion: str) -> bool:
        return completion.startswith("OK:")

    pairs = mint_preference_pairs(
        ["p1", "p2", "all-pass"],
        [
            ["OK: yes", "NO: nope"],
            ["bad", "OK: fine"],
            ["OK: a", "OK: b"],  # skipped — no reject
        ],
        validator,
    )
    assert len(pairs) == 2
    assert pairs[0]["chosen"] == "OK: yes"
    assert pairs[0]["rejected"] == "NO: nope"
    assert pairs[1]["chosen"] == "OK: fine"
    assert pairs[1]["rejected"] == "bad"


def test_mint_preference_pairs_with_callable() -> None:
    def candidates(prompt: str):
        return [f"OK:{prompt}", f"BAD:{prompt}"]

    pairs = mint_preference_pairs(
        ["hello"],
        candidates,
        lambda _p, c: c.startswith("OK:"),
    )
    assert len(pairs) == 1
    assert pairs[0]["chosen"] == "OK:hello"
    assert pairs[0]["rejected"] == "BAD:hello"


def test_preference_jsonl_prepare(tmp_path: Path) -> None:
    discover_plugins()
    assert FORMATS.get("preference_jsonl") is not None

    mdir = tmp_path / "pref-model"
    (mdir / "datasets" / "samples").mkdir(parents=True)
    seeds = mdir / "datasets" / "samples" / "seed_samples.jsonl"
    write_jsonl(
        seeds,
        [
            {
                "prompt": f"q{i}",
                "chosen": f"c{i}",
                "rejected": f"r{i}",
                "sample_id": f"id-{i}",
                "family": f"fam-{i}",
            }
            for i in range(8)
        ],
    )
    (mdir / "model.yml").write_text(
        """name: pref-test
model_id: pref-test
architecture: dpo
version: 0.1.0
dataset:
  format: preference_jsonl
  seed_samples: datasets/samples/seed_samples.jsonl
  split_ratios: [0.5, 0.25, 0.25]
  seed: 7
""",
        encoding="utf-8",
    )
    md = load_model_def(mdir)
    result = prepare_preference_jsonl(md)
    assert (md.prepared_dir / "train.jsonl").is_file()
    total = sum(result["split_counts"].values())
    assert total == 8
    assert result["split_counts"]["train"] >= 1
