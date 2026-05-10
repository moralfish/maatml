from __future__ import annotations

from enum import Enum
from typing import Any, Literal, Optional

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
    # `flow-studio/docs/smart-restart/messages.md`. Synced into the
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


class SpoolInterpretation(BaseModel):
    """Runtime response shape for the Spool Interpreter."""

    model_config = ConfigDict(extra="forbid")

    summary: str = Field(min_length=1)
    status: str
    returnCode: Optional[str] = None
    rootCause: str = Field(min_length=1)
    suggestedFix: str = Field(min_length=1)
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


# ---------------------------------------------------------------------------
# FlowGraphGenerator (mirrors flow-studio's apps/shared-types/src/graph.ts)
# ---------------------------------------------------------------------------

NodeKind = Literal["action", "ai", "cloud_ai", "utility"]
EdgeOutcome = Literal["pass", "fail", "always"]


class Position(BaseModel):
    """react-flow node position. flow-studio re-lays out on import; the model
    can use a simple grid (e.g. step k → x=k*200, y=100)."""

    model_config = ConfigDict(extra="forbid")

    x: float
    y: float


class FlowNodeDto(BaseModel):
    """Mirror of `FlowNodeDto` in flow-studio/apps/shared-types/src/graph.ts.

    `data` is polymorphic by `type`: action carries adapter+actionId+payload,
    ai carries modelId, cloud_ai carries provider+modelId+prompt, utility
    carries actionId. Validation of per-kind required fields lives in
    `flow_ml.validation.flow_graph_validator` (layer 5).
    """

    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1)
    type: NodeKind
    position: Position
    data: dict[str, Any]


class FlowEdgeDto(BaseModel):
    """Mirror of `FlowEdgeDto` in flow-studio. `outcome` defaults to `always`
    when omitted. Edge ids follow `e-<source>-<outcome>-<target>` by
    convention but are free-form strings for the validator."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1)
    source: str = Field(min_length=1)
    target: str = Field(min_length=1)
    label: Optional[str] = None
    condition: Optional[str] = None
    outcome: Optional[EdgeOutcome] = None


class FlowGraphProposal(BaseModel):
    """Runtime response shape: the JSON object FlowGraphGenerator emits.
    Mirrors `FlowGraphDto` plus a `warnings` array for the safety / ambiguity
    surface that flow-studio's TS type didn't carry but the spec requires.
    """

    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1)
    name: str
    version: str
    nodes: list[FlowNodeDto] = Field(default_factory=list)
    edges: list[FlowEdgeDto] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class FlowGraphSample(BaseModel):
    """Training sample for FlowGraphGenerator.

    `request` is the user's natural-language ask. `expected_graph` is the
    gold FlowGraphProposal (validated against the same schema as runtime
    output). `category` tracks the §7 taxonomy (simple, conditional,
    parallel, jcl-validation, ..., unsafe, repair) for stratified eval.
    """

    model_config = ConfigDict(extra="forbid")

    sample_id: str
    source: str
    category: str
    request: str
    expected_graph: FlowGraphProposal
    split: Split
