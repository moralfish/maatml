"""Evaluation entry points and report helpers.

Task evaluation goes through :func:`maatml.evaluation.harness.run_evaluation`
(via the CLI). Shared report types live in ``harness`` and are re-exported
here for backward compatibility.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from rich.console import Console

from .harness import (
    LatencyStats,
    Report,
    baseline_delta,
    binary_prf,
    latency_stats,
    per_class_prf,
    percentile,
    run_evaluation,
)

# Backward-compatible private aliases used by tests.
_percentile = percentile
_latency_stats = latency_stats
_binary_prf = binary_prf
_per_class_prf = per_class_prf
_baseline_delta = baseline_delta

console = Console()

__all__ = [
    "LatencyStats",
    "Report",
    "evaluate_model",
    "run_evaluation",
    "write_markdown_summary",
    "_percentile",
    "_latency_stats",
    "_binary_prf",
    "_per_class_prf",
    "_baseline_delta",
]


def evaluate_model(
    model_def,
    *,
    checkpoint: Optional[str] = None,
    split: str = "test",
    device: str = "auto",
    baseline: Optional[Path] = None,
    max_input_tokens: Optional[int] = None,
    limit: Optional[int] = None,
    gate: bool = False,
    smoke: bool = False,
) -> tuple["Report", Path]:
    """Evaluate a checkpoint of ``model_def`` and write report.{json,md}.

    The single implementation behind ``maatml evaluate`` and the lifecycle
    runner's evaluate step, so both enforce gates, resolve the token budget,
    and record results on the run identically. Configuration problems raise
    before the checkpoint is resolved or loaded.
    """
    from ..runs import get_run, resolve_checkpoint, update_run_gates
    from . import predictors as _predictors  # noqa: F401  register built-ins
    from .harness import (
        _resolve_metrics,
        resolve_gate_spec,
        resolve_validator,
        run_evaluation,
        uses_smoke_gates,
    )

    evaluation = model_def.evaluation or {}
    predictor = evaluation.get("predictor")
    validator = evaluation.get("validator")
    metrics = evaluation.get("metrics")
    if isinstance(metrics, list) and not metrics:
        metrics = None

    if predictor is None:
        from ..registry import PREDICTORS
        from ..scaffold import normalize_architecture

        arch = normalize_architecture(model_def.architecture)
        if PREDICTORS.get(model_def.architecture):
            predictor = model_def.architecture
        elif PREDICTORS.get(arch):
            predictor = arch
    if predictor is None:
        raise KeyError(
            f"No predictor for architecture={model_def.architecture!r}; "
            "set evaluation.predictor in model.yml"
        )

    gate_spec = None
    smoke_gated = False
    if gate:
        gate_spec = resolve_gate_spec(model_def, smoke=smoke)
        smoke_gated = smoke and uses_smoke_gates(model_def)
    if validator is not None:
        resolve_validator(validator)
    _resolve_metrics(metrics)

    ckpt = resolve_checkpoint(model_def, checkpoint)
    model_def.eval_dir.mkdir(parents=True, exist_ok=True)
    out_path = model_def.eval_dir / f"{ckpt.name}.json"
    budget = (
        max_input_tokens
        if max_input_tokens is not None
        else model_def.packaging.max_input_tokens
    )

    report = run_evaluation(
        checkpoint_dir=ckpt,
        dataset_dir=model_def.prepared_dir,
        out_path=out_path,
        model_def=model_def,
        predictor=predictor,
        validator=validator,
        metrics_fn=metrics,
        device=device,
        split=split,
        max_input_tokens=budget,
        baseline_path=baseline,
        limit=limit,
        task=model_def.task,
        enforce_gates=gate,
        gate_spec=gate_spec,
        smoke_gated=smoke_gated,
    )
    write_markdown_summary(report, out_path.with_suffix(".md"))

    run_rec = get_run(model_def, ckpt.name)
    if run_rec is not None and report.gates is not None:
        update_run_gates(
            model_def,
            run_rec.run_id,
            report.gates,
            metrics=report.metrics,
            smoke_gated=smoke_gated,
        )
    return report, out_path


_COUNT_KEYS = frozenset({"n", "support", "passed"})


def _format_class_stats(vals: dict[str, float]) -> str:
    """Render whatever per-class keys the report carries.

    Category buckets report ``pass_rate`` / ``passed`` / ``n``; metrics plugins
    that compute real per-class P/R/F1 report those instead. Neither shape is
    padded with invented keys.
    """
    parts = []
    for key in sorted(vals):
        value = vals[key]
        if key in _COUNT_KEYS:
            parts.append(f"{key}={int(value)}")
        else:
            parts.append(f"{key}={value:.3f}")
    return " ".join(parts)


def write_markdown_summary(report: Report, path: str | Path) -> Path:
    title = report.task or report.name or "eval"
    lines = [
        f"# {title} eval report",
        "",
        f"- model: `{report.model_id}`",
    ]
    if report.name:
        lines.append(f"- name: `{report.name}`")
    if report.version:
        lines.append(f"- version: `{report.version}`")
    lines.extend(
        [
            f"- dataset: `{report.dataset}`",
            f"- n: {report.n}",
            "",
            "## Metrics",
            "",
        ]
    )
    for k, v in sorted(report.metrics.items()):
        lines.append(f"- {k}: {v:.4f}")
    if report.gates is not None:
        lines.extend(["", "## Gates", ""])
        if report.passed is not None:
            lines.append(f"- passed: {report.passed}")
        results = report.gates.get("results") or {}
        for name, info in sorted(results.items()):
            lines.append(
                f"- {name}: actual={info.get('actual')} "
                f"minimum={info.get('minimum')} passed={info.get('passed')}"
            )
    if report.latency_ms:
        lines.extend(
            [
                "",
                "## Latency (ms)",
                f"- p50: {report.latency_ms.p50:.2f}",
                f"- p95: {report.latency_ms.p95:.2f}",
                f"- mean: {report.latency_ms.mean:.2f}",
                f"- n: {report.latency_ms.n}",
            ]
        )
    if report.per_class:
        lines.extend(["", "## Per-class", ""])
        for label, vals in sorted(report.per_class.items()):
            lines.append(f"- {label}: {_format_class_stats(vals)}")
    if report.baseline_delta:
        lines.extend(["", "## Baseline delta", ""])
        for k, v in sorted(report.baseline_delta.items()):
            sign = "+" if v >= 0 else ""
            lines.append(f"- {k}: {sign}{v:.4f}")
    if report.extras:
        lines.extend(["", "## Extras", ""])
        for k, v in sorted(report.extras.items()):
            lines.append(f"- {k}: {v}")
    out = Path(path)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out
