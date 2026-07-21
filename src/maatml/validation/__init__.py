"""Out-of-model validation shared types.

Task-specific validators live in example plugins and register via
``@register_validator``. Core only owns the shared result / fence helpers.
"""

from maatml.validation.base import (
    ValidationError,
    ValidationResult,
    strip_fences,
)

__all__ = [
    "ValidationError",
    "ValidationResult",
    "strip_fences",
]
