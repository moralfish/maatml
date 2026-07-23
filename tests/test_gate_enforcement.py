"""`--gate` / enforce_gates must never pass vacuously (v0.5.1 truth-and-safety)."""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from maatml.config import load_model_def
from maatml.evaluation.harness import GateConfigError, resolve_gate_spec

REPO_ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = sorted(p.parent.name for p in REPO_ROOT.glob("examples/*/model.yml"))


def test_resolve_gate_spec_returns_floats() -> None:
    md = SimpleNamespace(evaluation={"gates": {"json_parse_rate": 0.95, "acc": 1}})
    assert resolve_gate_spec(md) == {"json_parse_rate": 0.95, "acc": 1.0}


@pytest.mark.parametrize(
    "evaluation",
    [None, {}, {"gates": None}, {"gates": {}}, {"metrics": "x"}],
    ids=["no-eval", "empty-eval", "gates-none", "gates-empty", "no-gates-key"],
)
def test_resolve_gate_spec_raises_without_gates(evaluation) -> None:
    md = SimpleNamespace(evaluation=evaluation)
    with pytest.raises(GateConfigError):
        resolve_gate_spec(md)


def test_resolve_gate_spec_raises_on_none_model_def() -> None:
    with pytest.raises(GateConfigError):
        resolve_gate_spec(None)


@pytest.mark.parametrize("example", EXAMPLES)
def test_every_example_declares_gates(example: str) -> None:
    """The runner (v0.6) and strict --gate depend on every example gating."""
    md = load_model_def(REPO_ROOT / "examples" / example)
    spec = resolve_gate_spec(md)  # raises if the example has no gates
    assert spec and all(isinstance(v, float) for v in spec.values())
