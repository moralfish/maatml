"""Metrics for vision-vlm: scene mention, shape F1, pose phrase, brevity."""
from __future__ import annotations

import json
import re
from typing import Any, Sequence

from .constants import SHAPE_LABELS

_PLURALS = {
    "circle": "circles",
    "square": "squares",
    "triangle": "triangles",
    "star": "stars",
}


def _parse(gen_text: str) -> dict[str, Any] | None:
    try:
        from maatml.validation.base import strip_fences

        return json.loads(strip_fences(gen_text))
    except Exception:  # noqa: BLE001
        return None


def _desc(parsed: dict[str, Any] | None) -> str:
    if not parsed:
        return ""
    d = parsed.get("description")
    return d if isinstance(d, str) else ""


def compute_vision_vlm_metrics(row_results: Sequence[Any]) -> dict[str, float]:
    n = len(row_results)
    if n == 0:
        return {
            "scene_mention_rate": 0.0,
            "shape_mention_f1": 0.0,
            "pose_phrase_rate": 0.0,
            "brevity_rate": 0.0,
            "all_layers_pass_rate": 0.0,
        }

    scene_hits = 0
    pose_hits = 0
    brief_hits = 0
    layers_ok = 0
    f1_sum = 0.0

    for item in row_results:
        row = item.row if hasattr(item, "row") else item.get("row", {})
        gen = item.gen_text if hasattr(item, "gen_text") else item.get("gen_text", "")
        result = item.result if hasattr(item, "result") else None
        if result is not None and getattr(result, "ok", False):
            layers_ok += 1

        gt = row.get("gt") or {}
        text = _desc(_parse(gen)).lower()
        words = text.split()
        if 0 < len(words) <= 40 and "\n" not in text:
            brief_hits += 1

        scene = str(gt.get("scene") or "")
        if scene and scene in text:
            scene_hits += 1

        arms = str(gt.get("arms") or "")
        stance = str(gt.get("stance") or "")
        pose_ok = True
        if arms and arms not in text:
            pose_ok = False
        if stance and stance not in text:
            pose_ok = False
        if pose_ok and (arms or stance):
            pose_hits += 1
        elif not arms and not stance:
            pose_hits += 1

        # Shape mention F1 vs gt shape_counts keys.
        expected_shapes = set((gt.get("shape_counts") or {}).keys())
        predicted = set()
        for label in SHAPE_LABELS:
            if re.search(rf"\b{label}s?\b", text):
                predicted.add(label)
            plural = _PLURALS.get(label)
            if plural and re.search(rf"\b{plural}\b", text):
                predicted.add(label)
        if not expected_shapes and not predicted:
            f1 = 1.0
        elif not expected_shapes or not predicted:
            f1 = 0.0
        else:
            tp = len(expected_shapes & predicted)
            prec = tp / max(1, len(predicted))
            rec = tp / max(1, len(expected_shapes))
            f1 = 0.0 if prec + rec == 0 else 2 * prec * rec / (prec + rec)
        f1_sum += f1

    return {
        "scene_mention_rate": scene_hits / n,
        "shape_mention_f1": f1_sum / n,
        "pose_phrase_rate": pose_hits / n,
        "brevity_rate": brief_hits / n,
        "all_layers_pass_rate": layers_ok / n,
    }
