"""Synthetic vision-result → description generator for ``maatml datagen``."""
from __future__ import annotations

import random
from typing import Any, Callable, Optional

from maatml.config import ModelDefinition
from maatml.utils.io import stable_hash

from .constants import KEYPOINT_NAMES, SCENE_LABELS, SHAPE_LABELS
from .describe import description_payload
from .linearize import linearize_vision_result


def _rand_box(rng: random.Random) -> list[float]:
    x1 = rng.uniform(0.05, 0.55)
    y1 = rng.uniform(0.05, 0.55)
    w = rng.uniform(0.12, 0.35)
    h = rng.uniform(0.12, 0.35)
    return [
        round(x1, 4),
        round(y1, 4),
        round(min(0.98, x1 + w), 4),
        round(min(0.98, y1 + h), 4),
    ]


def _pose(rng: random.Random, style: str) -> dict[str, Any]:
    """Build a 12-keypoint stick figure with a controllable arm style."""
    cx = rng.uniform(0.35, 0.65)
    cy = rng.uniform(0.25, 0.4)
    scale = rng.uniform(0.18, 0.28)

    coords = {
        "head": (cx, cy - 0.12 * scale / 0.22),
        "neck": (cx, cy),
        "l_shoulder": (cx - 0.12, cy + 0.02),
        "r_shoulder": (cx + 0.12, cy + 0.02),
        "hip": (cx, cy + 0.28),
        "l_knee": (cx - 0.05, cy + 0.48),
        "r_knee": (cx + 0.05, cy + 0.48),
        "feet": (cx, cy + 0.68),
    }
    # Default elbows mid-way.
    coords["l_elbow"] = (cx - 0.16, cy + 0.16)
    coords["r_elbow"] = (cx + 0.16, cy + 0.16)

    if style == "both_up":
        coords["l_wrist"] = (cx - 0.14, cy - 0.10)
        coords["r_wrist"] = (cx + 0.14, cy - 0.10)
        coords["l_elbow"] = (cx - 0.15, cy - 0.02)
        coords["r_elbow"] = (cx + 0.15, cy - 0.02)
    elif style == "left_up":
        coords["l_wrist"] = (cx - 0.14, cy - 0.10)
        coords["r_wrist"] = (cx + 0.16, cy + 0.32)
        coords["l_elbow"] = (cx - 0.15, cy - 0.02)
    elif style == "right_up":
        coords["l_wrist"] = (cx - 0.16, cy + 0.32)
        coords["r_wrist"] = (cx + 0.14, cy - 0.10)
        coords["r_elbow"] = (cx + 0.15, cy - 0.02)
    elif style == "lowered":
        coords["l_wrist"] = (cx - 0.14, cy + 0.40)
        coords["r_wrist"] = (cx + 0.14, cy + 0.40)
        coords["l_elbow"] = (cx - 0.15, cy + 0.22)
        coords["r_elbow"] = (cx + 0.15, cy + 0.22)
    else:  # upright / neutral
        coords["l_wrist"] = (cx - 0.16, cy + 0.28)
        coords["r_wrist"] = (cx + 0.16, cy + 0.28)

    keypoints = []
    for name in KEYPOINT_NAMES:
        x, y = coords[name]
        # Tiny jitter so linearized inputs look prediction-like.
        x = max(0.0, min(1.0, x + rng.uniform(-0.01, 0.01)))
        y = max(0.0, min(1.0, y + rng.uniform(-0.01, 0.01)))
        keypoints.append(
            {
                "name": name,
                "x": round(x, 4),
                "y": round(y, 4),
                "confidence": round(rng.uniform(0.85, 1.0), 3),
            }
        )
    return {"keypoints": keypoints}


def build_synthetic_vision_payload(
    rng: random.Random,
    *,
    scene: Optional[str] = None,
    n_dets: Optional[int] = None,
    pose_style: Optional[str] = None,
) -> dict[str, Any]:
    scene = scene or rng.choice(SCENE_LABELS)
    if n_dets is None:
        n_dets = rng.randint(0, 3)
    pose_style = pose_style or rng.choice(
        ["both_up", "left_up", "right_up", "lowered", "upright"]
    )

    detections = []
    for _ in range(n_dets):
        detections.append(
            {
                "label": rng.choice(SHAPE_LABELS),
                "box": _rand_box(rng),
                "confidence": round(rng.uniform(0.55, 1.0), 3),
            }
        )
        # Occasional low-confidence distractor (filtered by linearizer).
        if rng.random() < 0.25:
            detections.append(
                {
                    "label": rng.choice(SHAPE_LABELS),
                    "box": _rand_box(rng),
                    "confidence": round(rng.uniform(0.05, 0.25), 3),
                }
            )

    return {
        "scene": {
            "label": scene,
            "confidence": round(rng.uniform(0.7, 1.0), 3),
        },
        "detections": detections,
        "pose": _pose(rng, pose_style),
    }


def make_sample_row(
    idx: int,
    *,
    seed: int = 0,
    scene: Optional[str] = None,
    n_dets: Optional[int] = None,
    pose_style: Optional[str] = None,
) -> dict[str, Any]:
    rng = random.Random(seed + idx * 9973)
    payload = build_synthetic_vision_payload(
        rng, scene=scene, n_dets=n_dets, pose_style=pose_style
    )
    request = linearize_vision_result(payload)
    expected = description_payload(payload)
    category = str(payload["scene"]["label"])
    sid = f"vd-{category}-{stable_hash(category, idx, seed)[:8]}"
    return {
        "sample_id": sid,
        "source": "synthetic:vision_describer",
        "family": f"vd:{category}:{idx // 8}",
        "category": category,
        "request": request,
        "expected_description": expected,
    }


def vision_describer_generator(
    model_def: ModelDefinition,
    *,
    seed: int = 0,
    **_kwargs: Any,
) -> Callable[[], Optional[dict[str, Any]]]:
    """Return a generate_fn for ``maatml datagen`` / gated corpus builders."""
    del model_def
    counter = {"n": 0}

    def _generate() -> Optional[dict[str, Any]]:
        counter["n"] += 1
        return make_sample_row(counter["n"], seed=seed)

    return _generate
