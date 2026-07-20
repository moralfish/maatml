from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field


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


class FailureCategory(str, Enum):
    dataset_resolution_failure = "dataset_resolution_failure"
    allocation_failure = "allocation_failure"
    permission_or_security_failure = "permission_or_security_failure"
    jcl_syntax_failure = "jcl_syntax_failure"
    utility_parameter_failure = "utility_parameter_failure"
    execution_abend = "execution_abend"
    scheduler_or_environment_issue = "scheduler_or_environment_issue"
    other = "other"


class Split(str, Enum):
    train = "train"
    val = "val"
    test = "test"


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
    """Generative SFT training sample for the JCL Validator.

    `request` is the sanitized JCL text the user supplies;
    `expected_validation_result` is the gold `JclValidationResult` JSON
    the model should emit. `category` aligns with the ErrorCategory enum
    plus a `valid` bucket for clean inputs.

    The legacy multi-head classifier carried per-error fields directly on
    the sample (is_valid, error_category, error_line, ...). With the
    generative SFT pivot those move into `expected_validation_result.errors[]`.
    """

    model_config = ConfigDict(extra="forbid")

    sample_id: str
    source: str
    category: str
    request: str
    expected_validation_result: JclValidationResult
    split: Split
    family: Optional[str] = None


class SpoolInterpretation(BaseModel):
    """Runtime response shape for the Spool Interpreter.

    Includes `explanation` (2-4 sentence narrative, distinct from `summary`)
    and `relatedDocs` (doc-key array per failure category). Both are
    optional at the schema level — the validator (layers 7-8) enforces
    `explanation` non-empty when `status != "completed"`.
    """

    model_config = ConfigDict(extra="forbid")

    summary: str = Field(min_length=1)
    status: str
    returnCode: Optional[str] = None
    rootCause: str = Field(min_length=1)
    suggestedFix: str = Field(min_length=1)
    explanation: Optional[str] = None
    relatedDocs: list[str] = Field(default_factory=list)
    failureCategory: Optional[str] = None
    confidence: float = Field(ge=0.0, le=1.0)


class SpoolSample(BaseModel):
    """Generative SFT training sample for the Spool Interpreter.

    `request` is the sanitized spool dump; `expected_interpretation` is the
    gold `SpoolInterpretation` JSON. `category` aligns with the
    FailureCategory enum plus a `completed` bucket for clean-completion
    samples.

    The legacy classifier-shape carried per-field rows (sanitized_spool,
    failure_category, root_cause, suggested_fix). With the generative SFT
    pivot those move into `expected_interpretation`.
    """

    model_config = ConfigDict(extra="forbid")

    sample_id: str
    source: str
    category: str
    request: str
    expected_interpretation: SpoolInterpretation
    split: Split
    family: Optional[str] = None
