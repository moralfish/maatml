"""Out-of-model validator for Spool Interpreter output.

Six layers, mirroring the JCL + FlowGraph validators:

  1. JSON parse                      — text → dict
  2. JSON schema                     — matches `SpoolInterpretation` JSON Schema
  3. status enum                     — status ∈ {completed, failed, abended, skipped, running}
  4. failureCategory enum            — failureCategory ∈ FailureCategory enum or null
  5. Field shape                     — non-empty summary/rootCause/suggestedFix;
                                       returnCode is string-or-null
  6. Consistency                     — `status: completed` ⟹ failureCategory ∈ {null, "other"};
                                       confidence ∈ [0, 1]
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import jsonschema


_FENCE_RX = re.compile(r"^```(?:json)?\s*\n(.*)\n```\s*$", re.DOTALL)
_THINK_BLOCK_RX = re.compile(r"<think>.*?</think>\s*", re.DOTALL | re.IGNORECASE)


@dataclass
class SpoolValidationError:
    layer: int
    code: str
    message: str
    location: Optional[str] = None


@dataclass
class SpoolValidationResultGate:
    raw_output: str
    parsed: Optional[dict[str, Any]] = None
    errors: list[SpoolValidationError] = field(default_factory=list)
    passed_layers: set[int] = field(default_factory=set)

    @property
    def ok(self) -> bool:
        return self.passed_layers == {1, 2, 3, 4, 5, 6}


def _strip_fences(text: str) -> str:
    """Strip Qwen3 `<think>...</think>` reasoning blocks and ```json fences
    before layer-1 JSON parse. Mirrors the FlowGraph validator's behaviour
    so all three tasks normalise the same wrappers uniformly.
    """
    text = text.strip()
    text = _THINK_BLOCK_RX.sub("", text).strip()
    fence = _FENCE_RX.match(text)
    if fence:
        return fence.group(1).strip()
    return text


def _load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def validate_spool_result(
    raw_output: str,
    *,
    schema_path: str | Path,
    contracts_path: str | Path,
    user_prompt: Optional[str] = None,
    strip_fences: bool = True,
) -> SpoolValidationResultGate:
    schema = _load_json(schema_path)
    contracts = _load_json(contracts_path)
    result = SpoolValidationResultGate(raw_output=raw_output)

    text = _strip_fences(raw_output) if strip_fences else raw_output.strip()

    # Layer 1
    try:
        result.parsed = json.loads(text)
        result.passed_layers.add(1)
    except json.JSONDecodeError as exc:
        result.errors.append(
            SpoolValidationError(layer=1, code="invalid_json", message=str(exc))
        )
        return result

    # Layer 2
    try:
        jsonschema.validate(instance=result.parsed, schema=schema)
        result.passed_layers.add(2)
    except jsonschema.exceptions.ValidationError as exc:
        result.errors.append(
            SpoolValidationError(
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

    # Layer 3 — status enum
    if status in statuses:
        result.passed_layers.add(3)
    else:
        result.errors.append(
            SpoolValidationError(
                layer=3,
                code="invalid_status",
                message=f"status {status!r} is not one of {sorted(statuses)}",
                location="status",
            )
        )

    # Layer 4 — failureCategory enum (or null)
    if failure_category is None or failure_category in failure_categories:
        result.passed_layers.add(4)
    else:
        result.errors.append(
            SpoolValidationError(
                layer=4,
                code="invalid_failure_category",
                message=(
                    f"failureCategory {failure_category!r} is not one of "
                    f"{sorted(failure_categories)} (or null)"
                ),
                location="failureCategory",
            )
        )

    # Layer 5 — field shape
    layer5_ok = True
    field_map = {"summary": summary, "rootCause": root_cause, "suggestedFix": suggested_fix}
    for fname in non_empty:
        v = field_map.get(fname)
        if not isinstance(v, str) or not v.strip():
            result.errors.append(
                SpoolValidationError(
                    layer=5,
                    code="empty_required_field",
                    message=f"{fname} must be a non-empty string",
                    location=fname,
                )
            )
            layer5_ok = False
    if return_code is not None and not isinstance(return_code, str):
        result.errors.append(
            SpoolValidationError(
                layer=5,
                code="invalid_return_code_type",
                message=f"returnCode must be string or null; got {type(return_code).__name__}",
                location="returnCode",
            )
        )
        layer5_ok = False
    if layer5_ok:
        result.passed_layers.add(5)

    # Layer 6 — consistency
    layer6_ok = True
    if status == "completed" and failure_category not in (None, "other"):
        result.errors.append(
            SpoolValidationError(
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
            SpoolValidationError(
                layer=6,
                code="invalid_confidence",
                message=f"confidence must be float in [0,1]; got {confidence!r}",
            )
        )
        layer6_ok = False
    if layer6_ok:
        result.passed_layers.add(6)

    return result
