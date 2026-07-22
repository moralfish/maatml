"""Deterministic natural-language description of a synthetic scene."""
from __future__ import annotations

from collections import Counter
from typing import Any

from .constants import SCENE_LABELS, SHAPE_LABELS

_SCENE_PHRASE = {
    "plain": "a plain background",
    "gradient": "a gradient background",
    "striped": "a striped background",
    "noisy": "a noisy background",
    "checker": "a checkerboard background",
}

_COUNT_WORDS = {
    1: "one",
    2: "two",
    3: "three",
}


def _pose_flags(pose: dict[str, Any], size: float = 1.0) -> dict[str, str]:
    """Derive coarse pose flags from keypoints (normalized or pixel)."""
    kps = {k["name"]: k for k in (pose.get("keypoints") or []) if "name" in k}
    if not kps:
        return {"arms": "relaxed", "stance": "neutral"}

    def _xy(name: str) -> tuple[float, float]:
        k = kps.get(name) or {}
        return float(k.get("x", 0.0)), float(k.get("y", 0.0))

    # Detect if coords look like pixels (>1) vs normalized.
    sample_x, _ = _xy("hip")
    scale = size if sample_x <= 1.5 else 1.0

    l_wr_y = _xy("l_wrist")[1] / scale
    r_wr_y = _xy("r_wrist")[1] / scale
    sh_y = (_xy("l_shoulder")[1] + _xy("r_shoulder")[1]) / (2.0 * scale)
    avg_wr = (l_wr_y + r_wr_y) / 2.0
    if avg_wr < sh_y - 0.02:
        arms = "raised"
    elif avg_wr > sh_y + 0.08:
        arms = "lowered"
    else:
        arms = "relaxed"

    l_kn_x = _xy("l_knee")[0] / scale
    r_kn_x = _xy("r_knee")[0] / scale
    hip_x = _xy("hip")[0] / scale
    width = abs(r_kn_x - l_kn_x)
    if width > 0.18:
        stance = "wide"
    elif width < 0.08:
        stance = "narrow"
    else:
        stance = "neutral"
    del hip_x
    return {"arms": arms, "stance": stance}


def extract_gt(expected_or_scene: dict[str, Any]) -> dict[str, Any]:
    """Build compact gt dict from a vision expected payload or scene render."""
    scene = (expected_or_scene.get("scene") or {}).get("label")
    if scene is None and "label" in (expected_or_scene.get("scene") or {}):
        scene = expected_or_scene["scene"]["label"]
    dets = expected_or_scene.get("detections") or []
    counts: Counter[str] = Counter()
    for d in dets:
        label = d.get("label")
        if label in SHAPE_LABELS:
            counts[label] += 1
    pose = expected_or_scene.get("pose") or {}
    flags = _pose_flags(pose)
    return {
        "scene": scene if scene in SCENE_LABELS else "plain",
        "shape_counts": {k: int(v) for k, v in sorted(counts.items())},
        "arms": flags["arms"],
        "stance": flags["stance"],
    }


def _shape_phrase(counts: dict[str, int]) -> str:
    if not counts:
        return "no shapes"
    parts = []
    for label, n in counts.items():
        word = _COUNT_WORDS.get(n, str(n))
        noun = label if n == 1 else (
            "circles" if label == "circle"
            else "squares" if label == "square"
            else "triangles" if label == "triangle"
            else "stars"
        )
        if n == 1:
            noun = label
        parts.append(f"{word} {noun}")
    if len(parts) == 1:
        return parts[0]
    if len(parts) == 2:
        return f"{parts[0]} and {parts[1]}"
    return ", ".join(parts[:-1]) + f", and {parts[-1]}"


def _pose_phrase(arms: str, stance: str) -> str:
    arm_bit = {
        "raised": "arms raised",
        "lowered": "arms lowered",
        "relaxed": "arms relaxed",
    }.get(arms, "arms relaxed")
    stance_bit = {
        "wide": "a wide stance",
        "narrow": "a narrow stance",
        "neutral": "a neutral stance",
    }.get(stance, "a neutral stance")
    return f"the figure stands with {arm_bit} and {stance_bit}"


def describe(gt: dict[str, Any]) -> str:
    """Compose a single factual sentence from gt."""
    scene = gt.get("scene") or "plain"
    scene_bit = _SCENE_PHRASE.get(scene, f"a {scene} background")
    shapes = _shape_phrase(dict(gt.get("shape_counts") or {}))
    pose = _pose_phrase(str(gt.get("arms") or "relaxed"), str(gt.get("stance") or "neutral"))
    if shapes == "no shapes":
        return f"{scene_bit.capitalize()} with no shapes; {pose}."
    return f"{scene_bit.capitalize()} with {shapes}; {pose}."


def describe_from_expected(expected: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    """Return ``(description, gt)`` from a vision expected payload."""
    gt = extract_gt(expected)
    return describe(gt), gt
