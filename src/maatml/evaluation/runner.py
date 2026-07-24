"""Evaluation entry points and report helpers.

Task evaluation goes through :func:`maatml.evaluation.harness.run_evaluation`
(via the CLI). Shared report types live in ``harness`` and are re-exported
here for backward compatibility.
"""
from __future__ import annotations

from pathlib import Path

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
    "run_evaluation",
    "write_markdown_summary",
    "_percentile",
    "_latency_stats",
    "_binary_prf",
    "_per_class_prf",
    "_baseline_delta",
]


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
