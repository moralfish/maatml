"""Evaluation entry points and report helpers.

Task-specific ``evaluate_*`` functions are thin wrappers around
:func:`flow_ml.evaluation.harness.run_evaluation`. Shared report types live
in ``harness`` and are re-exported here for backward compatibility.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from rich.console import Console

from ..config import ModelDefinition
from ..registry import discover_plugins
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
    "evaluate_jcl",
    "evaluate_spool",
    "run_evaluation",
    "write_markdown_summary",
    "_percentile",
    "_latency_stats",
    "_binary_prf",
    "_per_class_prf",
    "_baseline_delta",
]


def evaluate_jcl(
    model_dir: str | Path,
    dataset_dir: str | Path,
    out_path: str | Path,
    *,
    model_def: Optional[ModelDefinition] = None,
    schema_path: Optional[str | Path] = None,
    contracts_path: Optional[str | Path] = None,
    baseline_path: Optional[str | Path] = None,
    device: str = "auto",
    split: str = "test",
    max_input_tokens: int = 1024,
    failures_to_keep: int = 20,
    limit: Optional[int] = None,
) -> Report:
    """Evaluate the BERT multi-head JCL classifier on the held-out split.

    Schema/contracts must come from ``model_def``, explicit paths, or files
    under ``model_dir`` (checkpoint). Repo-root fallbacks are not used.
    """
    discover_plugins()
    # Import predictors so registrations land even if discover skipped them.
    from . import predictors as _predictors  # noqa: F401

    return run_evaluation(
        checkpoint_dir=Path(model_dir),
        dataset_dir=Path(dataset_dir),
        out_path=Path(out_path),
        model_def=model_def,
        predictor="multi_head_classifier",
        validator="jcl",
        metrics_fn="jcl",
        device=device,
        split=split,
        max_input_tokens=max_input_tokens,
        baseline_path=Path(baseline_path) if baseline_path else None,
        failures_to_keep=failures_to_keep,
        limit=limit,
        schema_path=Path(schema_path) if schema_path else None,
        contracts_path=Path(contracts_path) if contracts_path else None,
        task="jcl_validation",
    )


def evaluate_spool(
    model_dir: str | Path,
    dataset_dir: str | Path,
    out_path: str | Path,
    *,
    model_def: Optional[ModelDefinition] = None,
    schema_path: Optional[str | Path] = None,
    contracts_path: Optional[str | Path] = None,
    baseline_path: Optional[str | Path] = None,
    device: str = "auto",
    split: str = "test",
    max_input_tokens: int = 1024,
    failures_to_keep: int = 20,
    limit: Optional[int] = None,
) -> Report:
    """Evaluate the seq2seq Spool Interpreter on the held-out split.

    Schema/contracts must come from ``model_def``, explicit paths, or files
    under ``model_dir`` (checkpoint). Repo-root fallbacks are not used.
    """
    discover_plugins()
    from . import predictors as _predictors  # noqa: F401

    return run_evaluation(
        checkpoint_dir=Path(model_dir),
        dataset_dir=Path(dataset_dir),
        out_path=Path(out_path),
        model_def=model_def,
        predictor="seq2seq",
        validator="spool",
        metrics_fn="spool",
        device=device,
        split=split,
        max_input_tokens=max_input_tokens,
        baseline_path=Path(baseline_path) if baseline_path else None,
        failures_to_keep=failures_to_keep,
        limit=limit,
        schema_path=Path(schema_path) if schema_path else None,
        contracts_path=Path(contracts_path) if contracts_path else None,
        task="spool_interpretation",
    )


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
            lines.append(
                f"- {label}: P={vals['precision']:.3f} R={vals['recall']:.3f} "
                f"F1={vals['f1']:.3f} support={int(vals['support'])}"
            )
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
