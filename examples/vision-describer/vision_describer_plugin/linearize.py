"""Canonical vision-result linearizer shared by seed builders and compose client.

Full-precision vision JSON can exceed flan-t5-small's comfortable source budget.
This module filters low-confidence items, rounds floats, and emits a compact,
stable JSON string so training rows and serve-time inputs match.
"""
from __future__ import annotations

import json
from typing import Any, Optional

from .constants import (
    DEFAULT_NDIGITS,
    DEFAULT_SCORE_THRESH,
    KEYPOINT_NAMES,
    SCENE_LABELS,
    SHAPE_LABELS,
)


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _round_num(value: Any, ndigits: int) -> float:
    try:
        return round(_clamp01(float(value)), ndigits)
    except (TypeError, ValueError):
        return 0.0


def _score(obj: dict[str, Any], default: float = 1.0) -> float:
    conf = obj.get("confidence")
    if conf is None:
        return default
    try:
        return float(conf)
    except (TypeError, ValueError):
        return default


def clean_vision_result(
    payload: dict[str, Any] | str,
    *,
    score_thresh: float = DEFAULT_SCORE_THRESH,
    ndigits: int = DEFAULT_NDIGITS,
) -> dict[str, Any]:
    """Normalize a vision multitask payload into a compact structured dict."""
    if isinstance(payload, str):
        payload = json.loads(payload)
    if not isinstance(payload, dict):
        raise TypeError("vision payload must be a dict or JSON object string")

    scene_raw = payload.get("scene") if isinstance(payload.get("scene"), dict) else {}
    scene_label = str(scene_raw.get("label") or "plain")
    if scene_label not in SCENE_LABELS:
        scene_label = "plain"
    scene = {
        "label": scene_label,
        "confidence": round(_clamp01(_score(scene_raw)), ndigits),
    }

    detections_out: list[dict[str, Any]] = []
    raw_dets = payload.get("detections")
    if isinstance(raw_dets, list):
        for det in raw_dets:
            if not isinstance(det, dict):
                continue
            conf = _score(det)
            if conf < score_thresh:
                continue
            label = str(det.get("label") or "")
            if label not in SHAPE_LABELS:
                continue
            box = det.get("box")
            if not (isinstance(box, (list, tuple)) and len(box) == 4):
                continue
            x1, y1, x2, y2 = (_round_num(v, ndigits) for v in box)
            if x2 < x1:
                x1, x2 = x2, x1
            if y2 < y1:
                y1, y2 = y2, y1
            detections_out.append(
                {
                    "label": label,
                    "box": [x1, y1, x2, y2],
                    "confidence": round(_clamp01(conf), ndigits),
                }
            )
    # Stable order: label, then top-left y/x, then confidence desc.
    detections_out.sort(
        key=lambda d: (d["label"], d["box"][1], d["box"][0], -d["confidence"])
    )

    kp_by_name: dict[str, dict[str, Any]] = {}
    pose_raw = payload.get("pose") if isinstance(payload.get("pose"), dict) else {}
    raw_kps = pose_raw.get("keypoints") if isinstance(pose_raw, dict) else None
    if isinstance(raw_kps, list):
        for kp in raw_kps:
            if not isinstance(kp, dict):
                continue
            name = str(kp.get("name") or "")
            if name not in KEYPOINT_NAMES:
                continue
            conf = _score(kp)
            if conf < score_thresh:
                continue
            kp_by_name[name] = {
                "name": name,
                "x": _round_num(kp.get("x", 0.0), ndigits),
                "y": _round_num(kp.get("y", 0.0), ndigits),
                "confidence": round(_clamp01(conf), ndigits),
            }
    keypoints = [
        kp_by_name.get(
            name,
            {"name": name, "x": 0.0, "y": 0.0, "confidence": 0.0},
        )
        for name in KEYPOINT_NAMES
    ]

    return {
        "scene": scene,
        "detections": detections_out,
        "pose": {"keypoints": keypoints},
    }


def linearize_vision_result(
    payload: dict[str, Any] | str,
    *,
    score_thresh: float = DEFAULT_SCORE_THRESH,
    ndigits: int = DEFAULT_NDIGITS,
) -> str:
    """Return compact JSON text suitable as the describer ``request`` field."""
    cleaned = clean_vision_result(
        payload, score_thresh=score_thresh, ndigits=ndigits
    )
    return json.dumps(cleaned, ensure_ascii=False, separators=(",", ":"))


def parse_linearized(request: str) -> Optional[dict[str, Any]]:
    """Best-effort parse of a linearized request (or raw vision JSON)."""
    text = (request or "").strip()
    if not text:
        return None
    try:
        return clean_vision_result(text)
    except (json.JSONDecodeError, TypeError, ValueError):
        return None
