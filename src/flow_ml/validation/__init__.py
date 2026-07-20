"""Out-of-model validation pipelines.

One validator per generative task. Each follows the same shape: a
N-layer pipeline returning a result with `passed_layers` and a list of
categorised errors. The per-task evaluator and any seed-row gating share
this uniform interface.
"""

from flow_ml.validation.base import (
    ValidationError,
    ValidationResult,
    strip_fences,
)
from flow_ml.validation.jcl_validator import (
    JclValidationError as JclGateError,
    JclValidationResultGate,
    validate_jcl_result,
)
from flow_ml.validation.spool_validator import (
    SpoolValidationError as SpoolGateError,
    SpoolValidationResultGate,
    validate_spool_result,
)

__all__ = [
    "ValidationError",
    "ValidationResult",
    "strip_fences",
    "JclGateError",
    "JclValidationResultGate",
    "validate_jcl_result",
    "SpoolGateError",
    "SpoolValidationResultGate",
    "validate_spool_result",
]
