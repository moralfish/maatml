"""Deterministic short captions from cleaned vision results."""
from __future__ import annotations

from collections import Counter
from typing import Any

from .constants import KEYPOINT_NAMES, MAX_DESCRIPTION_WORDS
from .linearize import clean_vision_result


def _kp_map(cleaned: dict[str, Any]) -> dict[str, dict[str, Any]]:
    pose = cleaned.get("pose") or {}
    kps = pose.get("keypoints") if isinstance(pose, dict) else []
    out: dict[str, dict[str, Any]] = {}
    if not isinstance(kps, list):
        return out
    for kp in kps:
        if isinstance(kp, dict) and kp.get("name") in KEYPOINT_NAMES:
            out[str(kp["name"])] = kp
    return out


def _pose_phrase(kps: dict[str, dict[str, Any]]) -> str | None:
    """Heuristic pose phrase from wrist/shoulder/hip geometry."""
    needed = ("l_wrist", "r_wrist", "l_shoulder", "r_shoulder", "hip")
    if any(kps.get(n, {}).get("confidence", 0.0) <= 0.0 for n in needed):
        # Soft fallback: still try if majority present.
        present = sum(1 for n in needed if kps.get(n, {}).get("confidence", 0.0) > 0.0)
        if present < 3:
            return None

    def y(name: str, default: float = 0.5) -> float:
        return float(kps.get(name, {}).get("y", default))

    l_up = y("l_wrist") < y("l_shoulder") - 0.02
    r_up = y("r_wrist") < y("r_shoulder") - 0.02
    l_down = y("l_wrist") > y("hip") + 0.02
    r_down = y("r_wrist") > y("hip") + 0.02

    if l_up and r_up:
        return "raising both arms"
    if l_up and not r_up:
        return "raising the left arm"
    if r_up and not l_up:
        return "raising the right arm"
    if l_down and r_down:
        return "arms lowered"
    return "standing upright"


def _count_phrase(counts: Counter[str]) -> str:
    parts: list[str] = []
    for label in sorted(counts):
        n = counts[label]
        if n == 1:
            parts.append(f"a {label}")
        elif n == 2:
            parts.append(f"two {label}s")
        else:
            parts.append(f"{n} {label}s")
    if not parts:
        return "no shapes"
    if len(parts) == 1:
        return parts[0]
    if len(parts) == 2:
        return f"{parts[0]} and {parts[1]}"
    return ", ".join(parts[:-1]) + f", and {parts[-1]}"


def describe_vision_result(payload: dict[str, Any] | str) -> str:
    """Build one factual ≤30-word sentence from a vision payload."""
    cleaned = clean_vision_result(payload)
    scene = str((cleaned.get("scene") or {}).get("label") or "plain")
    dets = cleaned.get("detections") or []
    counts: Counter[str] = Counter()
    for det in dets:
        if isinstance(det, dict) and det.get("label"):
            counts[str(det["label"])] += 1

    objects = _count_phrase(counts)
    pose = _pose_phrase(_kp_map(cleaned))

    if objects == "no shapes":
        base = f"A {scene} scene contains no shapes"
    else:
        base = f"A {scene} scene contains {objects}"

    if pose:
        sentence = f"{base}, with the centered figure {pose}."
    else:
        sentence = f"{base}."

    words = sentence.split()
    if len(words) > MAX_DESCRIPTION_WORDS:
        sentence = " ".join(words[:MAX_DESCRIPTION_WORDS]).rstrip(".,;") + "."
    return sentence


def description_payload(payload: dict[str, Any] | str) -> dict[str, str]:
    return {"description": describe_vision_result(payload)}
