"""apply_overrides + expand_param_grid (no training)."""
from __future__ import annotations

from pathlib import Path

import pytest

from maatml.config import load_model_def
from maatml.overrides import (
    apply_overrides,
    coerce_override_value,
    expand_param_grid,
    overrides_from_mapping,
    parse_override,
    pick_metric,
)
from maatml.scaffold import scaffold_model


def test_coerce_override_value() -> None:
    assert coerce_override_value("true") is True
    assert coerce_override_value("false") is False
    assert coerce_override_value("null") is None
    assert coerce_override_value("42") == 42
    assert coerce_override_value("1e-4") == pytest.approx(1e-4)
    assert coerce_override_value('{"a":1}') == {"a": 1}
    assert coerce_override_value("hello") == "hello"


def test_parse_override() -> None:
    key, val = parse_override("training.learning_rate=1e-4")
    assert key == "training.learning_rate"
    assert val == pytest.approx(1e-4)


def test_apply_overrides_nested(tmp_path: Path) -> None:
    target = tmp_path / "ovr"
    scaffold_model(target, architecture="causal_sft", name="ovr")
    md = load_model_def(target)
    apply_overrides(
        md,
        [
            "training.learning_rate=3e-4",
            "training.lora.r=8",
            "smoke.max_steps=2",
            "dataset.seed=99",
        ],
    )
    assert md.training["learning_rate"] == pytest.approx(3e-4)
    assert md.training["lora"]["r"] == 8
    assert md.smoke["max_steps"] == 2
    assert md.dataset["seed"] == 99


def test_expand_param_grid_cartesian() -> None:
    grid = expand_param_grid(
        [
            "training.learning_rate=1e-4,3e-4",
            "training.lora.r=8,16",
        ]
    )
    assert len(grid) == 4
    lrs = sorted(g["training.learning_rate"] for g in grid)
    rs = sorted(g["training.lora.r"] for g in grid)
    assert lrs[0] == pytest.approx(1e-4)
    assert lrs[-1] == pytest.approx(3e-4)
    assert set(rs) == {8, 16}


def test_expand_param_grid_max_trials() -> None:
    grid = expand_param_grid(
        ["a=1,2", "b=3,4"],
        max_trials=2,
    )
    assert len(grid) == 2


def test_overrides_from_mapping_roundtrip() -> None:
    specs = overrides_from_mapping({"training.learning_rate": 1e-4, "training.lora.r": 8})
    assert any(s.startswith("training.learning_rate=") for s in specs)
    assert "training.lora.r=8" in specs


def test_pick_metric() -> None:
    key, val = pick_metric({"eval_loss": 0.5, "accuracy": 0.9}, "eval_loss")
    assert key == "eval_loss"
    assert val == 0.5
    key2, val2 = pick_metric({"accuracy": 0.9})
    assert key2 == "accuracy"
    assert val2 == 0.9
