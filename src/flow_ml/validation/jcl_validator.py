"""Out-of-model validator for JCL Validator output.

Six layers, mirroring the FlowGraph validator's design:

  1. JSON parse                       — text → dict
  2. JSON schema                      — matches `JclValidationResult` JSON Schema
  3. Severity enum                    — every error.severity ∈ {error, warning, info}
  4. Code enum                        — every error.code ∈ ErrorCategory enum
  5. Field shape                      — line ≥ 1, message non-empty, max 5 errors
  6. Consistency                      — `valid: false` ⟺ `errors` non-empty;
                                       `confidence` ∈ [0, 1]

Uses the same `FlowGraphValidationResult`/`FlowGraphValidationError` shape
as the FlowGraph validator so callers (per-task evaluator, future seed-row
gating) can treat all three task validators uniformly.
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
class JclValidationError:
    layer: int
    code: str
    message: str
    location: Optional[str] = None


@dataclass
class JclValidationResultGate:
    """Outcome of running the 6 validator layers on one model output.
    Distinct name from the gold-output Pydantic `JclValidationResult`
    in `flow_ml.data.schemas` — this is the *gate* result.
    """

    raw_output: str
    parsed: Optional[dict[str, Any]] = None
    errors: list[JclValidationError] = field(default_factory=list)
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


def validate_jcl_result(
    raw_output: str,
    *,
    schema_path: str | Path,
    contracts_path: str | Path,
    user_prompt: Optional[str] = None,
    strip_fences: bool = True,
) -> JclValidationResultGate:
    """Run all 6 layers against a model output. `user_prompt` is unused
    today (kept for API symmetry with the FlowGraph validator) — JCL
    has no prompt-side safety classifier."""
    schema = _load_json(schema_path)
    contracts = _load_json(contracts_path)
    result = JclValidationResultGate(raw_output=raw_output)

    text = _strip_fences(raw_output) if strip_fences else raw_output.strip()

    # Layer 1
    try:
        result.parsed = json.loads(text)
        result.passed_layers.add(1)
    except json.JSONDecodeError as exc:
        result.errors.append(
            JclValidationError(layer=1, code="invalid_json", message=str(exc))
        )
        return result

    # Layer 2
    try:
        jsonschema.validate(instance=result.parsed, schema=schema)
        result.passed_layers.add(2)
    except jsonschema.exceptions.ValidationError as exc:
        result.errors.append(
            JclValidationError(
                layer=2,
                code="schema_error",
                message=exc.message,
                location=".".join(str(p) for p in exc.absolute_path),
            )
        )

    errors = result.parsed.get("errors") or []
    valid_flag = result.parsed.get("valid")
    confidence = result.parsed.get("confidence")
    severities = set(contracts.get("severities", []))
    codes = set(contracts.get("error_codes", []))
    max_errors = int(contracts.get("max_errors_per_result", 5))

    # Layer 3 — severity enum
    layer3_ok = True
    for i, e in enumerate(errors):
        sev = e.get("severity")
        if sev not in severities:
            result.errors.append(
                JclValidationError(
                    layer=3,
                    code="invalid_severity",
                    message=f"severity {sev!r} is not one of {sorted(severities)}",
                    location=f"errors[{i}].severity",
                )
            )
            layer3_ok = False
    if layer3_ok:
        result.passed_layers.add(3)

    # Layer 4 — code enum
    layer4_ok = True
    for i, e in enumerate(errors):
        code = e.get("code")
        if code not in codes:
            result.errors.append(
                JclValidationError(
                    layer=4,
                    code="invalid_error_code",
                    message=f"code {code!r} is not one of {sorted(codes)}",
                    location=f"errors[{i}].code",
                )
            )
            layer4_ok = False
    if layer4_ok:
        result.passed_layers.add(4)

    # Layer 5 — field shape
    layer5_ok = True
    if len(errors) > max_errors:
        result.errors.append(
            JclValidationError(
                layer=5,
                code="too_many_errors",
                message=f"errors[] has {len(errors)} entries; max is {max_errors}",
                location="errors",
            )
        )
        layer5_ok = False
    for i, e in enumerate(errors):
        line = e.get("line")
        if not isinstance(line, int) or line < 1:
            result.errors.append(
                JclValidationError(
                    layer=5,
                    code="invalid_line",
                    message=f"line must be int >= 1; got {line!r}",
                    location=f"errors[{i}].line",
                )
            )
            layer5_ok = False
        if not e.get("message"):
            result.errors.append(
                JclValidationError(
                    layer=5,
                    code="empty_message",
                    message="error.message must be a non-empty string",
                    location=f"errors[{i}].message",
                )
            )
            layer5_ok = False
    if layer5_ok:
        result.passed_layers.add(5)

    # Layer 6 — consistency
    layer6_ok = True
    if not isinstance(valid_flag, bool):
        result.errors.append(
            JclValidationError(
                layer=6,
                code="missing_valid",
                message=f"`valid` must be a boolean; got {type(valid_flag).__name__}",
            )
        )
        layer6_ok = False
    elif bool(errors) == valid_flag:
        # valid=true with errors is inconsistent; valid=false with no errors is too
        result.errors.append(
            JclValidationError(
                layer=6,
                code="valid_errors_inconsistent",
                message=(
                    f"`valid={valid_flag}` is inconsistent with errors length "
                    f"{len(errors)} (valid=true iff errors empty)"
                ),
            )
        )
        layer6_ok = False
    if not isinstance(confidence, (int, float)) or not (0.0 <= float(confidence) <= 1.0):
        result.errors.append(
            JclValidationError(
                layer=6,
                code="invalid_confidence",
                message=f"confidence must be float in [0,1]; got {confidence!r}",
            )
        )
        layer6_ok = False
    if layer6_ok:
        result.passed_layers.add(6)

    return result
