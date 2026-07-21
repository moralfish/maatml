"""Evaluation package — harness, predictors, and report helpers."""

from __future__ import annotations

from typing import Any

__all__ = [
    "LatencyStats",
    "Report",
    "run_evaluation",
    "write_markdown_summary",
]


def __getattr__(name: str) -> Any:
    if name in {"LatencyStats", "Report", "run_evaluation"}:
        from .harness import LatencyStats, Report, run_evaluation

        return {"LatencyStats": LatencyStats, "Report": Report, "run_evaluation": run_evaluation}[
            name
        ]
    if name == "write_markdown_summary":
        from .runner import write_markdown_summary

        return write_markdown_summary
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
