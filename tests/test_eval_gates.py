"""Eval gate pass/fail logic."""
from __future__ import annotations

import pytest

from maatml.evaluation.harness import Report, check_gates


def test_check_gates_pass() -> None:
    out = check_gates(
        {"json_parse_rate": 0.995, "accuracy": 0.9},
        {"json_parse_rate": 0.99, "accuracy": 0.8},
    )
    assert out["passed"] is True
    assert out["results"]["json_parse_rate"]["passed"] is True


def test_check_gates_fail_missing_and_low() -> None:
    out = check_gates(
        {"json_parse_rate": 0.5},
        {"json_parse_rate": 0.99, "accuracy": 0.8},
    )
    assert out["passed"] is False
    assert out["results"]["json_parse_rate"]["passed"] is False
    assert out["results"]["accuracy"]["actual"] is None
    assert out["results"]["accuracy"]["passed"] is False


def test_report_includes_gates_fields() -> None:
    r = Report(
        model_id="m",
        name="m",
        version="0.1.0",
        metrics={"json_parse_rate": 1.0},
        gates={"passed": True, "results": {}},
        passed=True,
    )
    assert r.passed is True
    assert r.gates is not None


# --- smoke-tier gates ------------------------------------------------------


def test_smoke_gates_override_production_gates() -> None:
    from types import SimpleNamespace

    from maatml.evaluation.harness import effective_gates, uses_smoke_gates

    md = SimpleNamespace(
        evaluation={"gates": {"accuracy": 0.9}},
        smoke={"gates": {"output_nonempty_rate": 0.5}},
    )
    assert effective_gates(md) == {"accuracy": 0.9}
    assert effective_gates(md, smoke=True) == {"output_nonempty_rate": 0.5}
    assert uses_smoke_gates(md) is True

    # Without a smoke tier a smoke run is held to the production thresholds.
    plain = SimpleNamespace(evaluation={"gates": {"accuracy": 0.9}}, smoke={})
    assert effective_gates(plain, smoke=True) == {"accuracy": 0.9}
    assert uses_smoke_gates(plain) is False


def test_non_numeric_gate_value_is_a_config_error() -> None:
    from types import SimpleNamespace

    from maatml.evaluation.harness import GateConfigError, effective_gates

    md = SimpleNamespace(evaluation={"gates": {"accuracy": "high"}}, smoke={})
    with pytest.raises(GateConfigError, match="must be a number"):
        effective_gates(md)


def test_smoke_gates_do_not_reach_the_trainer_config(tmp_path) -> None:
    """`smoke.gates` is a lifecycle knob, not a training one."""
    from maatml.config import load_model_def

    mdir = tmp_path / "m"
    mdir.mkdir()
    (mdir / "model.yml").write_text(
        "name: m\nmodel_id: m\nversion: 0.1.0\n"
        "training:\n  epochs: 4\n"
        "smoke:\n  epochs: 1\n  gates:\n    output_nonempty_rate: 0.5\n",
        encoding="utf-8",
    )
    merged = load_model_def(mdir).merged_smoke()
    assert merged["epochs"] == 1
    assert "gates" not in merged


def test_coverage_metric_is_always_reported() -> None:
    from maatml.evaluation.harness import COVERAGE_METRIC, coverage_metrics
    from maatml.evaluation.harness import RowEval
    from maatml.validation.base import ValidationResult

    def _row(text: str) -> RowEval:
        return RowEval(row={}, gen_text=text, result=ValidationResult(raw_output=text))

    assert coverage_metrics([_row("{}"), _row("")])[COVERAGE_METRIC] == 0.5
    assert coverage_metrics([_row("  ")])[COVERAGE_METRIC] == 0.0
    assert coverage_metrics([])[COVERAGE_METRIC] == 0.0
