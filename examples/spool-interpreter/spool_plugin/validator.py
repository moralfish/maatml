"""Out-of-model validator for Spool Interpreter output.

Seven layers, mirroring the JCL validator:

  1. JSON parse: text → dict
  2. JSON schema: matches `SpoolInterpretation` JSON Schema
  3. status enum: status ∈ {completed, failed, abended, skipped, running}
  4. failureCategory enum: failureCategory ∈ FailureCategory enum or null
  5. Field shape: non-empty summary/rootCause; returnCode is string-or-null
  6. Consistency: `status: completed` ⟹ failureCategory ∈ {null, "other"};
                                       confidence ∈ [0, 1]
  7. Remediation: non-empty `suggestedFix` when status != "completed"

Layers 6 and 7 are the cross-field contracts: each relates two fields to one
another and neither can be expressed by the schema alone.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import jsonschema

from maatml.validation.base import (
    ValidationError,
    ValidationResult,
    _load_json,
    strip_fences as strip_model_fences,
)

# Backward-compatible aliases.
SpoolValidationError = ValidationError
SpoolValidationResultGate = ValidationResult

_SPOOL_LAYERS = frozenset({1, 2, 3, 4, 5, 6, 7})


def validate_spool_result(
    raw_output: str,
    *,
    schema_path: str | Path,
    contracts_path: str | Path,
    user_prompt: Optional[str] = None,
    strip_fences: bool = True,
) -> ValidationResult:
    del user_prompt  # API symmetry
    schema = _load_json(schema_path)
    contracts = _load_json(contracts_path)
    result = ValidationResult(
        raw_output=raw_output,
        required_layers=set(_SPOOL_LAYERS),
        n_layers=7,
    )

    text = strip_model_fences(raw_output) if strip_fences else raw_output.strip()

    # Layer 1
    try:
        result.parsed = json.loads(text)
        result.passed_layers.add(1)
    except json.JSONDecodeError as exc:
        result.errors.append(
            ValidationError(layer=1, code="invalid_json", message=str(exc))
        )
        return result

    # Layer 2
    try:
        jsonschema.validate(instance=result.parsed, schema=schema)
        result.passed_layers.add(2)
    except jsonschema.exceptions.ValidationError as exc:
        result.errors.append(
            ValidationError(
                layer=2,
                code="schema_error",
                message=exc.message,
                location=".".join(str(p) for p in exc.absolute_path),
            )
        )

    statuses = set(contracts.get("statuses", []))
    failure_categories = set(contracts.get("failure_categories", []))
    non_empty = set(contracts.get("non_empty_strings", []))

    status = result.parsed.get("status")
    failure_category = result.parsed.get("failureCategory")
    summary = result.parsed.get("summary")
    root_cause = result.parsed.get("rootCause")
    suggested_fix = result.parsed.get("suggestedFix")
    return_code = result.parsed.get("returnCode")
    confidence = result.parsed.get("confidence")

    # Layer 3: status enum
    if status in statuses:
        result.passed_layers.add(3)
    else:
        result.errors.append(
            ValidationError(
                layer=3,
                code="invalid_status",
                message=f"status {status!r} is not one of {sorted(statuses)}",
                location="status",
            )
        )

    # Layer 4: failureCategory enum (or null)
    if failure_category is None or failure_category in failure_categories:
        result.passed_layers.add(4)
    else:
        result.errors.append(
            ValidationError(
                layer=4,
                code="invalid_failure_category",
                message=(
                    f"failureCategory {failure_category!r} is not one of "
                    f"{sorted(failure_categories)} (or null)"
                ),
                location="failureCategory",
            )
        )

    # Layer 5: field shape
    layer5_ok = True
    field_map = {"summary": summary, "rootCause": root_cause, "suggestedFix": suggested_fix}
    for fname in non_empty:
        v = field_map.get(fname)
        if not isinstance(v, str) or not v.strip():
            result.errors.append(
                ValidationError(
                    layer=5,
                    code="empty_required_field",
                    message=f"{fname} must be a non-empty string",
                    location=fname,
                )
            )
            layer5_ok = False
    if return_code is not None and not isinstance(return_code, str):
        result.errors.append(
            ValidationError(
                layer=5,
                code="invalid_return_code_type",
                message=f"returnCode must be string or null; got {type(return_code).__name__}",
                location="returnCode",
            )
        )
        layer5_ok = False
    if layer5_ok:
        result.passed_layers.add(5)

    # Layer 6: consistency
    layer6_ok = True
    if status == "completed" and failure_category not in (None, "other"):
        result.errors.append(
            ValidationError(
                layer=6,
                code="completed_with_failure_category",
                message=(
                    f"status=completed must have failureCategory null or 'other'; "
                    f"got {failure_category!r}"
                ),
            )
        )
        layer6_ok = False
    if not isinstance(confidence, (int, float)) or not (0.0 <= float(confidence) <= 1.0):
        result.errors.append(
            ValidationError(
                layer=6,
                code="invalid_confidence",
                message=f"confidence must be float in [0,1]; got {confidence!r}",
            )
        )
        layer6_ok = False
    if layer6_ok:
        result.passed_layers.add(6)

    # Layer 7: suggestedFix required when status != "completed"
    if status == "completed":
        if suggested_fix is None or isinstance(suggested_fix, str):
            result.passed_layers.add(7)
        else:
            result.errors.append(
                ValidationError(
                    layer=7,
                    code="invalid_suggested_fix_type",
                    message=(
                        f"suggestedFix must be string or null; "
                        f"got {type(suggested_fix).__name__}"
                    ),
                    location="suggestedFix",
                )
            )
    elif isinstance(suggested_fix, str) and suggested_fix.strip():
        result.passed_layers.add(7)
    else:
        result.errors.append(
            ValidationError(
                layer=7,
                code="missing_suggested_fix",
                message=f"suggestedFix must be non-empty when status={status!r}",
                location="suggestedFix",
            )
        )

    return result
