"""Support-ticket-triage example plugin — validator + metrics.

The triage model uses the built-in ``causal_sft`` predictor; this package only
adds the task's out-of-model contract (validator) and its scoring (metrics).
"""
from __future__ import annotations

from maatml.registry import register_metrics, register_validator

from .metrics import compute_triage_metrics
from .validator import validate_triage

register_validator("triage")(validate_triage)
register_metrics("triage")(compute_triage_metrics)

__all__ = ["compute_triage_metrics", "validate_triage"]
