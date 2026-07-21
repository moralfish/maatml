"""Validator for multitask vision JSON outputs."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from maatml.validation.base import ValidationError, ValidationResult
from maatml.validation.base import strip_fences as _strip_fences

from .constants import KEYPOINT_NAMES, SCENE_LABELS, SHAPE_LABELS


def validate_vision_scene(
    raw_output: str,
    *,
    schema_path: str | Path | None = None,
    contracts_path: str | Path | None = None,
    user_prompt: Optional[str] = None,
    strip_fences: bool = True,
) -> ValidationResult:
    """Layered gate: parse → schema (optional) → field shape → label enums."""
    del user_prompt, contracts_path
    text = _strip_fences(raw_output) if strip_fences else raw_output
    result = ValidationResult(raw_output=raw_output, n_layers=4, required_layers={1, 2, 3, 4})

    # Layer 1: JSON
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

    # Layer 2: JSON Schema (when provided)
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
        result.passed_layers.add(2)

    # Layer 3: required keys / shapes
    scene = parsed.get("scene")
    dets = parsed.get("detections")
    pose = parsed.get("pose")
    ok3 = True
    if not isinstance(scene, dict) or "label" not in scene:
        result.errors.append(
            ValidationError(layer=3, code="scene_shape", message="scene.label required")
        )
        ok3 = False
    if not isinstance(dets, list):
        result.errors.append(
            ValidationError(layer=3, code="detections_shape", message="detections must be list")
        )
        ok3 = False
    else:
        for i, d in enumerate(dets):
            if not isinstance(d, dict) or "label" not in d or "box" not in d:
                result.errors.append(
                    ValidationError(
                        layer=3,
                        code="detection_item",
                        message=f"detections[{i}] needs label+box",
                        location=f"detections[{i}]",
                    )
                )
                ok3 = False
                break
            box = d.get("box")
            if not (isinstance(box, list) and len(box) == 4):
                result.errors.append(
                    ValidationError(
                        layer=3,
                        code="box_shape",
                        message="box must be [x1,y1,x2,y2]",
                        location=f"detections[{i}].box",
                    )
                )
                ok3 = False
                break
    if not isinstance(pose, dict) or not isinstance(pose.get("keypoints"), list):
        result.errors.append(
            ValidationError(layer=3, code="pose_shape", message="pose.keypoints list required")
        )
        ok3 = False
    if ok3:
        result.passed_layers.add(3)

    # Layer 4: enum membership
    ok4 = True
    if isinstance(scene, dict):
        label = scene.get("label")
        if label not in SCENE_LABELS:
            result.errors.append(
                ValidationError(
                    layer=4,
                    code="scene_label",
                    message=f"unknown scene label {label!r}",
                )
            )
            ok4 = False
    if isinstance(dets, list):
        for i, d in enumerate(dets):
            if isinstance(d, dict) and d.get("label") not in SHAPE_LABELS:
                result.errors.append(
                    ValidationError(
                        layer=4,
                        code="shape_label",
                        message=f"unknown shape {d.get('label')!r}",
                        location=f"detections[{i}].label",
                    )
                )
                ok4 = False
    if isinstance(pose, dict) and isinstance(pose.get("keypoints"), list):
        names = {k.get("name") for k in pose["keypoints"] if isinstance(k, dict)}
        missing = [n for n in KEYPOINT_NAMES if n not in names]
        if missing:
            result.errors.append(
                ValidationError(
                    layer=4,
                    code="keypoints_missing",
                    message=f"missing keypoints: {missing}",
                )
            )
            ok4 = False
    if ok4:
        result.passed_layers.add(4)
    return result
