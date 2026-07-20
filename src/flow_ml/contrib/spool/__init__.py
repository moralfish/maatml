"""Spool contrib registrations.

Validators stay in ``flow_ml.validation.spool_validator`` and are registered
here. Metrics live under ``contrib.spool.metrics``.
"""

from flow_ml.registry import register_metrics, register_validator
from flow_ml.validation.spool_validator import validate_spool_result

from .metrics import compute_spool_metrics

register_validator("spool")(validate_spool_result)
register_metrics("spool")(compute_spool_metrics)

__all__ = ["compute_spool_metrics", "validate_spool_result"]
