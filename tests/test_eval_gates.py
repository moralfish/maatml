"""Eval gate pass/fail logic."""
from __future__ import annotations

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
