"""Out-of-model validator for vision-describer outputs.

Layers:
  1. JSON parse
  2. JSON Schema
  3. Field shape — non-empty description string
  4. Conciseness — ≤ MAX_DESCRIPTION_WORDS words, single sentence-ish
  5. Scene grounding — description mentions the request's scene label (when parsable)
  6. Object grounding — description reflects detection counts when objects exist
"""
from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Optional

import jsonschema

from maatml.validation.base import ValidationError, ValidationResult
from maatml.validation.base import strip_fences as strip_model_fences

from .constants import MAX_DESCRIPTION_WORDS, SCENE_LABELS, SHAPE_LABELS
from .linearize import parse_linearized

_LAYERS = frozenset({1, 2, 3, 4, 5, 6})
_WORD_RX = re.compile(r"[A-Za-z0-9']+")


def _words(text: str) -> list[str]:
    return _WORD_RX.findall(text.lower())


def _scene_from_request(user_prompt: Optional[str]) -> Optional[str]:
    cleaned = parse_linearized(user_prompt or "")
    if cleaned is None:
        return None
    scene = (cleaned.get("scene") or {}).get("label")
    return str(scene) if scene in SCENE_LABELS else None


def _counts_from_request(user_prompt: Optional[str]) -> Optional[Counter[str]]:
    cleaned = parse_linearized(user_prompt or "")
    if cleaned is None:
        return None
    counts: Counter[str] = Counter()
    for det in cleaned.get("detections") or []:
        if isinstance(det, dict) and det.get("label") in SHAPE_LABELS:
            counts[str(det["label"])] += 1
    return counts


def validate_vision_describer(
    raw_output: str,
    *,
    schema_path: str | Path,
    contracts_path: str | Path | None = None,
    user_prompt: Optional[str] = None,
    strip_fences: bool = True,
) -> ValidationResult:
    del contracts_path  # contracts optional; enums live in constants
    result = ValidationResult(
        raw_output=raw_output,
        required_layers=set(_LAYERS),
        n_layers=6,
    )
    text = strip_model_fences(raw_output) if strip_fences else raw_output.strip()

    # Layer 1 — JSON
    try:
        parsed = json.loads(text)
        if not isinstance(parsed, dict):
            raise json.JSONDecodeError("root must be object", text, 0)
        result.parsed = parsed
        result.passed_layers.add(1)
    except json.JSONDecodeError as exc:
        result.errors.append(
            ValidationError(layer=1, code="invalid_json", message=str(exc))
        )
        return result

    # Layer 2 — schema
    try:
        schema = json.loads(Path(schema_path).read_text(encoding="utf-8"))
        jsonschema.validate(instance=result.parsed, schema=schema)
        result.passed_layers.add(2)
    except Exception as exc:  # noqa: BLE001
        result.errors.append(
            ValidationError(layer=2, code="schema_error", message=str(exc))
        )

    description = result.parsed.get("description") if result.parsed else None

    # Layer 3 — field shape
    if isinstance(description, str) and description.strip():
        result.passed_layers.add(3)
    else:
        result.errors.append(
            ValidationError(
                layer=3,
                code="empty_description",
                message="description must be a non-empty string",
                location="description",
            )
        )

    # Layer 4 — conciseness
    if isinstance(description, str):
        words = description.split()
        if len(words) <= MAX_DESCRIPTION_WORDS and description.count(".") <= 2:
            result.passed_layers.add(4)
        else:
            result.errors.append(
                ValidationError(
                    layer=4,
                    code="too_long",
                    message=(
                        f"description has {len(words)} words "
                        f"(max {MAX_DESCRIPTION_WORDS})"
                    ),
                    location="description",
                )
            )

    # Layer 5 — scene grounding (skip-pass when request unparsable)
    scene = _scene_from_request(user_prompt)
    if scene is None:
        result.passed_layers.add(5)
    elif isinstance(description, str) and scene.lower() in description.lower():
        result.passed_layers.add(5)
    else:
        result.errors.append(
            ValidationError(
                layer=5,
                code="scene_ungrounded",
                message=f"description does not mention scene label {scene!r}",
                location="description",
            )
        )

    # Layer 6 — object grounding
    counts = _counts_from_request(user_prompt)
    if counts is None:
        result.passed_layers.add(6)
    elif not counts:
        # No detections → expect "no shapes" or no shape nouns.
        if isinstance(description, str):
            desc_l = description.lower()
            mentioned = [s for s in SHAPE_LABELS if s in desc_l]
            if "no shapes" in desc_l or not mentioned:
                result.passed_layers.add(6)
            else:
                result.errors.append(
                    ValidationError(
                        layer=6,
                        code="spurious_objects",
                        message=f"unexpected shape mentions: {mentioned}",
                        location="description",
                    )
                )
    else:
        if isinstance(description, str):
            desc_l = description.lower()
            missing = [lab for lab in counts if lab not in desc_l]
            if not missing:
                result.passed_layers.add(6)
            else:
                result.errors.append(
                    ValidationError(
                        layer=6,
                        code="objects_ungrounded",
                        message=f"missing object labels: {missing}",
                        location="description",
                    )
                )

    return result
