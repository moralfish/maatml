"""Alpaca / ShareGPT format normalization and prepare smoke."""
from __future__ import annotations

from pathlib import Path

import pytest

from maatml.config import load_model_def
from maatml.data.formats import normalize_alpaca, normalize_sharegpt, prepare_alpaca
from maatml.registry import FORMATS, discover_plugins
from maatml.utils.io import iter_jsonl, write_jsonl


def test_normalize_alpaca_with_optional_system() -> None:
    row = {
        "instruction": "Classify",
        "input": "hello",
        "output": '{"ok":true}',
        "system": "You are helpful.",
        "sample_id": "a1",
        "family": "fam",
    }
    out = normalize_alpaca(row)
    assert out["sample_id"] == "a1"
    assert out["family"] == "fam"
    roles = [m["role"] for m in out["messages"]]
    assert roles == ["system", "user", "assistant"]
    assert "hello" in out["messages"][1]["content"]
    assert out["messages"][2]["content"] == '{"ok":true}'


def test_normalize_sharegpt() -> None:
    row = {
        "id": "s1",
        "conversations": [
            {"from": "human", "value": "Hi"},
            {"from": "gpt", "value": "Hello"},
        ],
        "source": "web",
    }
    out = normalize_sharegpt(row)
    assert out["sample_id"] == "s1"
    assert out["source"] == "web"
    assert out["messages"] == [
        {"role": "user", "content": "Hi"},
        {"role": "assistant", "content": "Hello"},
    ]


def test_alpaca_prepare_writes_splits(tmp_path: Path) -> None:
    discover_plugins()
    assert FORMATS.get("alpaca") is not None

    mdir = tmp_path / "alpaca-model"
    (mdir / "datasets" / "samples").mkdir(parents=True)
    seeds = mdir / "datasets" / "samples" / "seed_samples.jsonl"
    write_jsonl(
        seeds,
        [
            {
                "instruction": f"q{i}",
                "input": "",
                "output": f"a{i}",
                "sample_id": f"id-{i}",
                "family": "f" if i < 2 else "g",
            }
            for i in range(4)
        ],
    )
    (mdir / "model.yml").write_text(
        """name: alpaca-test
model_id: alpaca-test
architecture: causal_sft
version: 0.1.0
dataset:
  format: alpaca
  seed_samples: datasets/samples/seed_samples.jsonl
  split_ratios: [0.5, 0.25, 0.25]
  group_by: family
""",
        encoding="utf-8",
    )
    md = load_model_def(mdir)
    summary = prepare_alpaca(md)
    assert summary["split_counts"]["train"] + summary["split_counts"]["val"] + summary[
        "split_counts"
    ]["test"] == 4
    # Canonical messages present in prepared rows
    any_row = next(iter_jsonl(md.prepared_dir / "train.jsonl"), None)
    if any_row is None:
        any_row = next(iter_jsonl(md.prepared_dir / "val.jsonl"))
    assert "messages" in any_row
    assert any_row["messages"][0]["role"] in ("user", "system")


def test_alpaca_sanitize_declared_raises(tmp_path: Path) -> None:
    """G3: declaring sanitize on a format that cannot sanitize must error,
    not silently skip while the dataset card claims it ran."""
    discover_plugins()
    mdir = tmp_path / "alpaca-san"
    (mdir / "datasets" / "samples").mkdir(parents=True)
    write_jsonl(
        mdir / "datasets" / "samples" / "seed_samples.jsonl",
        [{"instruction": "q", "input": "", "output": "a", "sample_id": "id-0", "family": "f"}],
    )
    (mdir / "model.yml").write_text(
        """name: alpaca-san
model_id: alpaca-san
architecture: causal_sft
version: 0.1.0
dataset:
  format: alpaca
  seed_samples: datasets/samples/seed_samples.jsonl
  sanitize: [jcl]
""",
        encoding="utf-8",
    )
    md = load_model_def(mdir)
    with pytest.raises(ValueError, match="does not sanitize"):
        prepare_alpaca(md)


def test_degenerate_rows_are_dropped_and_counted(tmp_path: Path) -> None:
    from maatml.data.formats import is_degenerate

    mdir = tmp_path / "model"
    (mdir / "datasets").mkdir(parents=True)
    seeds = mdir / "datasets" / "seeds.jsonl"
    write_jsonl(
        seeds,
        [
            {"instruction": "do a thing", "output": "done", "family": "a"},
            {"instruction": "", "input": "", "output": "", "family": "b"},
            {"instruction": "another", "output": "", "family": "c"},
            {"instruction": "third", "output": "ok", "family": "d"},
        ],
    )
    (mdir / "model.yml").write_text(
        """name: alpaca-degenerate
model_id: alpaca-degenerate
architecture: causal_sft
version: 0.1.0
dataset:
  format: alpaca
  seed_samples: datasets/seeds.jsonl
""",
        encoding="utf-8",
    )
    md = load_model_def(mdir)
    with pytest.warns(RuntimeWarning, match="dropped 2 row"):
        summary = prepare_alpaca(md)
    assert summary["degenerate_dropped"] == 2
    assert sum(summary["split_counts"].values()) == 2

    assert is_degenerate({"messages": []}) is True
    assert is_degenerate({"messages": [{"role": "user", "content": "hi"}]}) is True
    assert (
        is_degenerate(
            {
                "messages": [
                    {"role": "user", "content": "hi"},
                    {"role": "assistant", "content": "yo"},
                ]
            }
        )
        is False
    )


def test_all_degenerate_corpus_fails(tmp_path: Path) -> None:
    from maatml.data.formats import prepare_sharegpt

    mdir = tmp_path / "model"
    (mdir / "datasets").mkdir(parents=True)
    write_jsonl(mdir / "datasets" / "seeds.jsonl", [{"conversations": []}])
    (mdir / "model.yml").write_text(
        """name: sharegpt-empty
model_id: sharegpt-empty
architecture: causal_sft
version: 0.1.0
dataset:
  format: sharegpt
  seed_samples: datasets/seeds.jsonl
""",
        encoding="utf-8",
    )
    md = load_model_def(mdir)
    with pytest.raises(ValueError, match="No usable rows"):
        prepare_sharegpt(md)
