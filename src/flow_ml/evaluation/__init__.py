"""Evaluation package — harness, predictors, and task wrappers."""

from __future__ import annotations

from typing import Any

__all__ = [
    "LatencyStats",
    "Report",
    "run_evaluation",
    "evaluate_jcl",
    "evaluate_spool",
    "write_markdown_summary",
]


def __getattr__(name: str) -> Any:
    if name in {"LatencyStats", "Report", "run_evaluation"}:
        from .harness import LatencyStats, Report, run_evaluation

        return {"LatencyStats": LatencyStats, "Report": Report, "run_evaluation": run_evaluation}[
            name
        ]
    if name in {"evaluate_jcl", "evaluate_spool", "write_markdown_summary"}:
        from .runner import evaluate_jcl, evaluate_spool, write_markdown_summary

        return {
            "evaluate_jcl": evaluate_jcl,
            "evaluate_spool": evaluate_spool,
            "write_markdown_summary": write_markdown_summary,
        }[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
