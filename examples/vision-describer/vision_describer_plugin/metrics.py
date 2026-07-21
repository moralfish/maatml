"""Evaluation metrics for vision-describer."""
from __future__ import annotations

from collections import Counter
from typing import TYPE_CHECKING

from .constants import MAX_DESCRIPTION_WORDS, SCENE_LABELS, SHAPE_LABELS
from .linearize import parse_linearized

if TYPE_CHECKING:
    from maatml.evaluation.harness import RowEval


def _pred_description(item: "RowEval") -> str | None:
    parsed = item.result.parsed if item.result else None
    if isinstance(parsed, dict) and isinstance(parsed.get("description"), str):
        return parsed["description"]
    return None


def _gold_description(row: dict) -> str | None:
    gold = row.get("expected_description")
    if isinstance(gold, dict) and isinstance(gold.get("description"), str):
        return gold["description"]
    if isinstance(gold, str):
        return gold
    return None


def compute_vision_describer_metrics(row_results: list["RowEval"]) -> dict[str, float]:
    n = len(row_results)
    if n == 0:
        return {}

    layer_pass = {i: 0 for i in range(1, 7)}
    all_ok = 0
    exact = 0
    concise = 0
    scene_ok = 0
    scene_total = 0
    object_ok = 0
    object_total = 0
    pose_ok = 0
    pose_total = 0
    word_sum = 0
    word_n = 0

    for item in row_results:
        result = item.result
        for layer in range(1, 7):
            if layer in result.passed_layers:
                layer_pass[layer] += 1
        if result.ok:
            all_ok += 1

        pred = _pred_description(item)
        gold = _gold_description(item.row)
        if pred is not None and gold is not None and pred.strip() == gold.strip():
            exact += 1
        if pred is not None:
            words = pred.split()
            word_sum += len(words)
            word_n += 1
            if len(words) <= MAX_DESCRIPTION_WORDS:
                concise += 1

        request = item.row.get("request")
        cleaned = parse_linearized(request) if isinstance(request, str) else None
        if cleaned is not None and pred is not None:
            scene = (cleaned.get("scene") or {}).get("label")
            if scene in SCENE_LABELS:
                scene_total += 1
                if str(scene).lower() in pred.lower():
                    scene_ok += 1

            counts: Counter[str] = Counter()
            for det in cleaned.get("detections") or []:
                if isinstance(det, dict) and det.get("label") in SHAPE_LABELS:
                    counts[str(det["label"])] += 1
            object_total += 1
            if not counts:
                if "no shapes" in pred.lower() or not any(
                    s in pred.lower() for s in SHAPE_LABELS
                ):
                    object_ok += 1
            elif all(lab in pred.lower() for lab in counts):
                object_ok += 1

            # Pose grounding: if gold mentions arms/standing, pred should share a cue.
            if gold is not None:
                cues = (
                    "raising both arms",
                    "raising the left arm",
                    "raising the right arm",
                    "arms lowered",
                    "standing upright",
                )
                gold_cue = next((c for c in cues if c in gold.lower()), None)
                if gold_cue is not None:
                    pose_total += 1
                    if gold_cue in pred.lower():
                        pose_ok += 1

    return {
        "json_parse_rate": layer_pass[1] / n,
        "schema_conformance_rate": layer_pass[2] / n,
        "field_shape_rate": layer_pass[3] / n,
        "conciseness_rate": layer_pass[4] / n if layer_pass[4] else concise / n,
        "scene_grounding_rate": (
            scene_ok / scene_total if scene_total else layer_pass[5] / n
        ),
        "object_grounding_rate": (
            object_ok / object_total if object_total else layer_pass[6] / n
        ),
        "pose_grounding_rate": pose_ok / pose_total if pose_total else 1.0,
        "all_layers_pass_rate": all_ok / n,
        "exact_match_rate": exact / n,
        "mean_description_words": word_sum / word_n if word_n else 0.0,
    }
