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
    # Smart/RESTART and Smart/RRSAF specific buckets. Sourced from
    # `flow-starter/docs/smart-restart/messages.md`. Synced into the
    # spool-interpreter `prompt_spec.json` `failure_categories` array via
    # `flow-ml/scripts/sync-smart-restart-knowledge.sh`.
    smart_restart_resource_unavailable = "smart_restart_resource_unavailable"
    smart_restart_configuration = "smart_restart_configuration"
    smart_restart_application_logic = "smart_restart_application_logic"
    smart_restart_input_syntax = "smart_restart_input_syntax"


class Split(str, Enum):
    train = "train"
    val = "val"
    test = "test"


class JclSample(BaseModel):
    model_config = ConfigDict(extra="forbid")

    sample_id: str
    source: str
    sanitized_jcl: str
    is_valid: bool
    error_category: Optional[ErrorCategory] = None
    error_line: Optional[int] = Field(default=None, ge=1)
    error_column: Optional[int] = Field(default=None, ge=1)
    suggestion: Optional[str] = None
    split: Split


class SpoolSample(BaseModel):
    model_config = ConfigDict(extra="forbid")

    sample_id: str
    source: str
    sanitized_spool: str
    status: str
    return_code: Optional[str] = None
    failure_category: FailureCategory
    root_cause: str
    suggested_fix: str
    split: Split


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


class SpoolInterpretation(BaseModel):
    summary: str
    status: str
    returnCode: Optional[str] = None
    rootCause: str
    suggestedFix: str
    confidence: float = Field(ge=0.0, le=1.0)


class DslSample(BaseModel):
    """Training sample for the DSL Generator. `description` is plain English;
    `dsl` is a canonical Flow DSL document the model should reproduce."""

    model_config = ConfigDict(extra="forbid")

    sample_id: str
    source: str
    description: str
    dsl: str
    split: Split


class DslGeneration(BaseModel):
    """Runtime response shape declared in `prompt_spec.response_schema`."""

    dsl: str


class AgentToolCall(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    arguments: dict


class AgentPlanStep(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    title: str
    action: str
    depends_on: list[str] = Field(default_factory=list)


class AgentPlan(BaseModel):
    """Runtime response shape for the local Flow Studio workflow-planning agent."""

    model_config = ConfigDict(extra="forbid")

    intent_summary: str
    plan_steps: list[AgentPlanStep]
    tool_calls: list[AgentToolCall] = Field(default_factory=list)
    dsl_patch: Optional[str] = None
    dsl: Optional[str] = None
    confidence: float = Field(ge=0.0, le=1.0)
    refusal_reason: Optional[str] = None


class AgentSample(BaseModel):
    """Training sample for the Agent Planner.

    `request` is the user's natural-language ask. `context` is optional Flow
    Studio state, serialized as a short string. `agent_plan` is the strict JSON
    object the model should emit.
    """

    model_config = ConfigDict(extra="forbid")

    sample_id: str
    source: str
    request: str
    context: str = ""
    expected_intent: str
    agent_plan: AgentPlan
    split: Split
