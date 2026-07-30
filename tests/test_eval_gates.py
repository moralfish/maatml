"""Eval gate pass/fail logic."""

from __future__ import annotations

from pathlib import Path

import pytest

from maatml.config import load_model_def
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
    from maatml.evaluation.harness import COVERAGE_METRIC, RowEval, coverage_metrics
    from maatml.validation.base import ValidationResult

    def _row(text: str) -> RowEval:
        return RowEval(row={}, gen_text=text, result=ValidationResult(raw_output=text))

    assert coverage_metrics([_row("{}"), _row("")])[COVERAGE_METRIC] == 0.5
    assert coverage_metrics([_row("  ")])[COVERAGE_METRIC] == 0.0
    assert coverage_metrics([])[COVERAGE_METRIC] == 0.0


def test_declared_but_missing_contracts_is_an_error_not_a_silent_none(
    tmp_path: Path,
) -> None:
    """A typo in dataset.contracts must fail with the path it could not find,
    not resolve to None and surface later as a TypeError from the validator."""
    from maatml.evaluation.harness import DeclaredAssetMissing, resolve_eval_asset

    mdir = tmp_path / "model"
    mdir.mkdir(parents=True)
    (mdir / "model.yml").write_text(
        """name: assets
model_id: assets
architecture: causal_sft
version: 0.1.0
dataset:
  seed_samples: seeds.jsonl
  contracts: datasets/nod_contracts.json
""",
        encoding="utf-8",
    )
    md = load_model_def(mdir)
    with pytest.raises(DeclaredAssetMissing, match="nod_contracts.json"):
        resolve_eval_asset(
            "contracts",
            model_def=md,
            checkpoint_dir=tmp_path / "ckpt",
            filenames=("node_contracts.json",),
        )


def test_undeclared_missing_asset_stays_optional(tmp_path: Path) -> None:
    """The optional path is unchanged: an asset that is simply absent raises
    plain FileNotFoundError, which callers treat as 'no asset'."""
    from maatml.evaluation.harness import DeclaredAssetMissing, resolve_eval_asset

    mdir = tmp_path / "model"
    mdir.mkdir(parents=True)
    (mdir / "model.yml").write_text(
        """name: assets2
model_id: assets2
architecture: causal_sft
version: 0.1.0
dataset:
  seed_samples: seeds.jsonl
""",
        encoding="utf-8",
    )
    md = load_model_def(mdir)
    with pytest.raises(FileNotFoundError) as excinfo:
        resolve_eval_asset(
            "contracts",
            model_def=md,
            checkpoint_dir=tmp_path / "ckpt",
            filenames=("node_contracts.json",),
        )
    assert not isinstance(excinfo.value, DeclaredAssetMissing)


def test_default_eval_keys_keeps_every_metrics_entry(tmp_path: Path) -> None:
    """evaluation.metrics may be a list and every entry runs. A duplicated copy
    in scripts/evaluate_all.py truncated it to metrics[0], so the sweep silently
    reported only the first plugin's metrics."""
    from maatml.evaluation.harness import default_eval_keys

    mdir = tmp_path / "model"
    mdir.mkdir(parents=True)
    (mdir / "model.yml").write_text(
        """name: multi
model_id: multi
architecture: causal_sft
version: 0.1.0
dataset:
  seed_samples: seeds.jsonl
evaluation:
  metrics: [alpha, beta]
""",
        encoding="utf-8",
    )
    md = load_model_def(mdir)
    _predictor, _validator, metrics = default_eval_keys(md)
    assert metrics == ["alpha", "beta"], "metrics list was truncated"


def test_only_one_default_eval_keys_implementation() -> None:
    """Regression guard: the helper lived in both cli.py (dead) and
    scripts/evaluate_all.py (live and stale), and the two had drifted."""
    import pathlib

    assert "_default_eval_keys" not in pathlib.Path("src/maatml/cli.py").read_text()
    script = pathlib.Path("scripts/evaluate_all.py").read_text()
    assert "def _default_eval_keys" not in script
    assert "default_eval_keys(md)" in script
