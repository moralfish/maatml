"""Spool task schemas (sample + interpretation shapes)."""
from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field

from maatml.data.schemas import Split


class FailureCategory(str, Enum):
    dataset_resolution_failure = "dataset_resolution_failure"
    allocation_failure = "allocation_failure"
    permission_or_security_failure = "permission_or_security_failure"
    jcl_syntax_failure = "jcl_syntax_failure"
    utility_parameter_failure = "utility_parameter_failure"
    execution_abend = "execution_abend"
    scheduler_or_environment_issue = "scheduler_or_environment_issue"
    other = "other"


class SpoolInterpretation(BaseModel):
    """Runtime response shape for the Spool Interpreter."""

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
    """Training sample for the Spool Interpreter."""

    model_config = ConfigDict(extra="forbid")

    sample_id: str
    source: str
    category: str
    request: str
    expected_interpretation: SpoolInterpretation
    split: Split
    family: Optional[str] = None
