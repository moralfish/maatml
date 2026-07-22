"""Validator for vision-vlm description JSON outputs."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from maatml.validation.base import ValidationError, ValidationResult
from maatml.validation.base import strip_fences as _strip_fences

_MAX_WORDS = 40


def validate_vision_vlm(
    raw_output: str,
    *,
    schema_path: str | Path | None = None,
    contracts_path: str | Path | None = None,
    user_prompt: Optional[str] = None,
    strip_fences: bool = True,
) -> ValidationResult:
    """Layers: JSON → schema → brevity → sanity."""
    del user_prompt, contracts_path
    text = _strip_fences(raw_output) if strip_fences else raw_output
    result = ValidationResult(raw_output=raw_output, n_layers=4, required_layers={1, 2, 3, 4})

    # Layer 1: JSON object
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as exc:
        result.errors.append(
            ValidationError(layer=1, code="invalid_json", message=str(exc))
        )
        return result
    if not isinstance(parsed, dict):
        result.errors.append(
            ValidationError(layer=1, code="not_object", message="root must be object")
        )
        return result
    result.parsed = parsed
    result.passed_layers.add(1)

    # Layer 2: JSON Schema
    if schema_path is not None and Path(schema_path).is_file():
        try:
            import jsonschema

            schema = json.loads(Path(schema_path).read_text(encoding="utf-8"))
            jsonschema.validate(parsed, schema)
            result.passed_layers.add(2)
        except Exception as exc:  # noqa: BLE001
            result.errors.append(
                ValidationError(layer=2, code="schema", message=str(exc))
            )
    else:
        # Minimal shape check without schema file.
        if "description" not in parsed or not isinstance(parsed.get("description"), str):
            result.errors.append(
                ValidationError(
                    layer=2, code="missing_description", message="description string required"
                )
            )
        else:
            result.passed_layers.add(2)

    desc = parsed.get("description") if isinstance(parsed, dict) else None
    if not isinstance(desc, str):
        return result

    # Layer 3: brevity
    words = [w for w in desc.strip().split() if w]
    ok3 = True
    if not words:
        result.errors.append(
            ValidationError(layer=3, code="empty", message="description is empty")
        )
        ok3 = False
    if len(words) > _MAX_WORDS:
        result.errors.append(
            ValidationError(
                layer=3,
                code="too_long",
                message=f"description has {len(words)} words (max {_MAX_WORDS})",
            )
        )
        ok3 = False
    if "\n" in desc.strip():
        result.errors.append(
            ValidationError(layer=3, code="multiline", message="description must be one line")
        )
        ok3 = False
    if ok3:
        result.passed_layers.add(3)

    # Layer 4: sanity (printable, ends with punctuation optional)
    ok4 = True
    if any(ord(c) < 9 for c in desc):
        result.errors.append(
            ValidationError(layer=4, code="control_chars", message="control characters present")
        )
        ok4 = False
    if ok4:
        result.passed_layers.add(4)
    return result
