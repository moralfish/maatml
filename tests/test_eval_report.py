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
