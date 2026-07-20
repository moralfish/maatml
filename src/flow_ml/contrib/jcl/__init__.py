"""JCL contrib registrations.

Validators stay in ``flow_ml.validation.jcl_validator`` and are registered
here. Metrics live under ``contrib.jcl.metrics``. The custom JCL tokenizer
remains in ``flow_ml.tokenization``; this package documents that coupling.
"""

from flow_ml.registry import register_metrics, register_validator
from flow_ml.validation.jcl_validator import validate_jcl_result

from .metrics import compute_jcl_metrics

register_validator("jcl")(validate_jcl_result)
register_metrics("jcl")(compute_jcl_metrics)

__all__ = ["compute_jcl_metrics", "validate_jcl_result"]
