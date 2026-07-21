"""JCL task schemas (sample + validation result shapes)."""
from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field

from maatml.data.schemas import Split


class Severity(str, Enum):
    error = "error"
    warning = "warning"
    info = "info"


class ErrorCategory(str, Enum):
    missing_dd = "missing_dd"
    invalid_job_card = "invalid_job_card"
    unresolved_symbolic_parameter = "unresolved_symbolic_parameter"
    continuation_error = "continuation_error"
    invalid_exec_statement = "invalid_exec_statement"
    invalid_dataset_reference_structure = "invalid_dataset_reference_structure"
    other = "other"
    none = "none"


class JclError(BaseModel):
    line: int = Field(ge=1)
    column: Optional[int] = Field(default=None, ge=1)
    severity: Severity = Severity.error
    code: ErrorCategory
    message: str
    suggestion: Optional[str] = None


class JclValidationResult(BaseModel):
    valid: bool
    errors: list[JclError] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)


class JclSample(BaseModel):
    """Training sample for the JCL Validator."""

    model_config = ConfigDict(extra="forbid")

    sample_id: str
    source: str
    category: str
    request: str
    expected_validation_result: JclValidationResult
    split: Split
    family: Optional[str] = None
