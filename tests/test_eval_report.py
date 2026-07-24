from __future__ import annotations

from pathlib import Path

from maatml.evaluation.runner import (
    LatencyStats,
    Report,
    _baseline_delta,
    _binary_prf,
    _per_class_prf,
    _percentile,
    write_markdown_summary,
)


def test_percentile_basic() -> None:
    assert _percentile([], 0.5) == 0.0
    assert _percentile([1.0], 0.5) == 1.0
    assert _percentile([1.0, 2.0, 3.0, 4.0, 5.0], 0.5) == 3.0
    assert _percentile([1.0, 2.0, 3.0, 4.0, 5.0], 0.0) == 1.0
    assert _percentile([1.0, 2.0, 3.0, 4.0, 5.0], 1.0) == 5.0


def test_binary_prf_zero_safe() -> None:
    assert _binary_prf(0, 0, 0) == {"precision": 0.0, "recall": 0.0, "f1": 0.0, "support": 0.0}
    prf = _binary_prf(tp=2, fp=1, fn=1)
    assert prf["precision"] == pytest_approx(2 / 3)
    assert prf["recall"] == pytest_approx(2 / 3)
    assert prf["f1"] == pytest_approx(2 / 3)
    assert prf["support"] == 3.0


def test_per_class_prf_handles_unseen_labels() -> None:
    out = _per_class_prf(["a", "a", "b"], ["a", "b", "b"], ["a", "b", "c"])
    assert out["a"]["precision"] == 1.0
    assert out["a"]["recall"] == 0.5
    assert out["c"]["support"] == 0.0


def test_report_round_trip(tmp_path: Path) -> None:
    r = Report(
        model_id="m",
        name="m",
        version="0.1.0",
        task="jcl_validation",
        dataset="d",
        n=4,
        metrics={"seq_accuracy": 0.75},
        per_class={"missing_dd": {"precision": 1.0, "recall": 0.5, "f1": 0.66, "support": 2.0}},
        latency_ms=LatencyStats(p50=10.0, p95=15.0, mean=12.0, n=4),
        extras={"note": "x"},
    )
    path = tmp_path / "report.json"
    r.write(path)
    loaded = Report.read(path)
    assert loaded == r


def test_baseline_delta_subtraction(tmp_path: Path) -> None:
    base = Report(
        model_id="m1",
        task="jcl_validation",
        dataset="d",
        n=10,
        metrics={"seq_accuracy": 0.5, "category_accuracy": 0.4},
    )
    base_path = tmp_path / "base.json"
    base.write(base_path)
    delta = _baseline_delta(
        {"seq_accuracy": 0.7, "category_accuracy": 0.5, "new_metric": 0.9},
        base_path,
    )
    assert delta == {
        "seq_accuracy": pytest_approx(0.2),
        "category_accuracy": pytest_approx(0.1),
    }


def test_markdown_summary_writes_file(tmp_path: Path) -> None:
    r = Report(
        model_id="m",
        task="spool_interpretation",
        dataset="d",
        n=5,
        metrics={"json_validity": 0.8, "category_accuracy": 0.6},
        latency_ms=LatencyStats(p50=100.0, p95=200.0, mean=120.0, n=5),
    )
    path = tmp_path / "report.md"
    write_markdown_summary(r, path)
    body = path.read_text(encoding="utf-8")
    assert "spool_interpretation" in body
    assert "json_validity: 0.8000" in body
    assert "p50: 100.00" in body


def pytest_approx(v):
    import pytest

    return pytest.approx(v)


def _row_eval(category: str, ok: bool):
    from maatml.evaluation.harness import RowEval
    from maatml.validation.base import ValidationError, ValidationResult

    result = ValidationResult(raw_output="{}", n_layers=1)
    if ok:
        result.passed_layers.add(1)
    else:
        result.errors.append(ValidationError(layer=1, code="bad", message="nope"))
    return RowEval(row={"category": category}, gen_text="{}", result=result)


def test_category_buckets_report_pass_rate_not_fabricated_prf() -> None:
    from maatml.evaluation.harness import _category_buckets

    buckets = _category_buckets(
        [
            _row_eval("missing_dd", True),
            _row_eval("missing_dd", False),
            _row_eval("valid", True),
        ]
    )
    assert buckets["missing_dd"] == {"pass_rate": 0.5, "passed": 1.0, "n": 2.0}
    assert buckets["valid"] == {"pass_rate": 1.0, "passed": 1.0, "n": 1.0}
    # The old shape invented recall=1.0 / f1=0.0 for every bucket.
    for stats in buckets.values():
        assert "recall" not in stats
        assert "f1" not in stats


def test_markdown_summary_renders_bucket_and_prf_shapes(tmp_path: Path) -> None:
    r = Report(
        model_id="m",
        task="t",
        dataset="d",
        n=3,
        per_class={
            "missing_dd": {"pass_rate": 0.5, "passed": 1.0, "n": 2.0},
            "valid": {"precision": 1.0, "recall": 0.5, "f1": 0.66, "support": 2.0},
        },
    )
    body = write_markdown_summary(r, tmp_path / "report.md").read_text(encoding="utf-8")
    assert "- missing_dd: n=2 pass_rate=0.500 passed=1" in body
    assert "- valid: f1=0.660 precision=1.000 recall=0.500 support=2" in body


def test_metrics_list_runs_every_entry() -> None:
    from maatml.evaluation.harness import _merge_metrics, _resolve_metrics
    from maatml.registry import METRICS

    METRICS.register("t_a", lambda rows: {"a": 1.0}, source="test")
    METRICS.register("t_b", lambda rows: {"b": 2.0}, source="test")
    try:
        callables = _resolve_metrics(["t_a", "t_b"])
        assert _merge_metrics(callables, [], ["t_a", "t_b"]) == {"a": 1.0, "b": 2.0}
    finally:
        METRICS.unregister("t_a")
        METRICS.unregister("t_b")


def test_metrics_key_collision_is_a_config_error() -> None:
    import pytest

    from maatml.evaluation.harness import GateConfigError, _merge_metrics, _resolve_metrics
    from maatml.registry import METRICS

    METRICS.register("t_a", lambda rows: {"same": 1.0}, source="test")
    METRICS.register("t_b", lambda rows: {"same": 2.0}, source="test")
    try:
        callables = _resolve_metrics(["t_a", "t_b"])
        with pytest.raises(GateConfigError, match="both report"):
            _merge_metrics(callables, [], ["t_a", "t_b"])
    finally:
        METRICS.unregister("t_a")
        METRICS.unregister("t_b")
