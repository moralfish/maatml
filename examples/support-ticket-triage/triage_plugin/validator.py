"""Out-of-model validator for support-ticket triage outputs.

Layers:
  1. JSON parse
  2. JSON Schema (structure, required fields, enums)
  3. Routing contract — category must route to the mandated team
  4. Summary quality — non-empty, single line, <= MAX_SUMMARY_WORDS words

Layer 3 is the one a plain schema cannot express: it ties two fields together
by a task rule. That is where the validator earns its keep.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import jsonschema

from maatml.validation.base import ValidationError, ValidationResult
from maatml.validation.base import strip_fences as strip_model_fences

from .constants import MAX_SUMMARY_WORDS, ROUTING

_LAYERS = frozenset({1, 2, 3, 4})


def validate_triage(
    raw_output: str,
    *,
    schema_path: str | Path,
    contracts_path: str | Path | None = None,
    user_prompt: Optional[str] = None,
    strip_fences: bool = True,
) -> ValidationResult:
    del contracts_path, user_prompt  # routing contract is self-contained
    result = ValidationResult(
        raw_output=raw_output,
        required_layers=set(_LAYERS),
        n_layers=4,
    )
    text = strip_model_fences(raw_output) if strip_fences else raw_output.strip()

    # Layer 1 — JSON
    try:
        parsed = json.loads(text)
        if not isinstance(parsed, dict):
            raise json.JSONDecodeError("root must be an object", text, 0)
        result.parsed = parsed
        result.passed_layers.add(1)
    except json.JSONDecodeError as exc:
        result.errors.append(
            ValidationError(layer=1, code="invalid_json", message=str(exc))
        )
        return result

    # Layer 2 — schema (structure + enums + required fields)
    try:
        schema = json.loads(Path(schema_path).read_text(encoding="utf-8"))
        jsonschema.validate(instance=result.parsed, schema=schema)
        result.passed_layers.add(2)
    except jsonschema.ValidationError as exc:
        result.errors.append(
            ValidationError(
                layer=2,
                code="schema_error",
                message=exc.message,
                location="/".join(str(p) for p in exc.absolute_path) or None,
            )
        )
    except Exception as exc:  # noqa: BLE001 — malformed schema file, etc.
        result.errors.append(
            ValidationError(layer=2, code="schema_error", message=str(exc))
        )

    category = result.parsed.get("category")
    team = result.parsed.get("team")
    summary = result.parsed.get("summary")

    # Layer 3 — routing contract (category → team)
    expected_team = ROUTING.get(category) if isinstance(category, str) else None
    if expected_team is None:
        # Unknown/……invalid category is a schema (layer 2) failure; don't
        # double-count it here. Nothing to check against the contract.
        result.errors.append(
            ValidationError(
                layer=3,
                code="unroutable_category",
                message=f"category {category!r} has no routing rule",
                location="category",
            )
        )
    elif team == expected_team:
        result.passed_layers.add(3)
    else:
        result.errors.append(
            ValidationError(
                layer=3,
                code="misrouted",
                message=(
                    f"category {category!r} must route to {expected_team!r}, "
                    f"got team {team!r}"
                ),
                location="team",
            )
        )

    # Layer 4 — summary quality
    if isinstance(summary, str) and summary.strip():
        words = summary.split()
        if len(words) <= MAX_SUMMARY_WORDS and "\n" not in summary:
            result.passed_layers.add(4)
        else:
            result.errors.append(
                ValidationError(
                    layer=4,
                    code="summary_shape",
                    message=(
                        f"summary must be one line of <= {MAX_SUMMARY_WORDS} "
                        f"words (got {len(words)} words)"
                    ),
                    location="summary",
                )
            )
    else:
        result.errors.append(
            ValidationError(
                layer=4,
                code="empty_summary",
                message="summary must be a non-empty string",
                location="summary",
            )
        )

    return result
