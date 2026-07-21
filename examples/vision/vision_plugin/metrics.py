"""Metrics for multitask vision: scene accuracy, VOC-style mAP@0.5, PCK@0.2."""
from __future__ import annotations

import json
from typing import Any, Sequence

from .constants import KEYPOINT_NAMES, SHAPE_LABELS


def _parse_output(gen_text: str) -> dict[str, Any] | None:
    try:
        from maatml.validation.base import strip_fences

        return json.loads(strip_fences(gen_text))
    except Exception:  # noqa: BLE001
        return None


def _iou(a: Sequence[float], b: Sequence[float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    return inter / (area_a + area_b - inter + 1e-9)


def _ap_for_class(
    preds: list[tuple[float, list[float]]],
    gts: list[list[float]],
    iou_thresh: float = 0.5,
) -> float:
    """Average precision for one class (preds = list of (score, box))."""
    if not gts and not preds:
        return 1.0
    if not gts:
        return 0.0
    if not preds:
        return 0.0
    preds = sorted(preds, key=lambda x: x[0], reverse=True)
    matched = [False] * len(gts)
    tp = []
    fp = []
    for score, box in preds:
        del score
        best_iou = 0.0
        best_j = -1
        for j, gt in enumerate(gts):
            if matched[j]:
                continue
            v = _iou(box, gt)
            if v > best_iou:
                best_iou = v
                best_j = j
        if best_iou >= iou_thresh and best_j >= 0:
            matched[best_j] = True
            tp.append(1)
            fp.append(0)
        else:
            tp.append(0)
            fp.append(1)
    # Precision-recall envelope
    cum_tp = 0
    cum_fp = 0
    precs = []
    recs = []
    n_gt = len(gts)
    for t, f in zip(tp, fp):
        cum_tp += t
        cum_fp += f
        precs.append(cum_tp / max(1, cum_tp + cum_fp))
        recs.append(cum_tp / n_gt)
    # 11-point interpolation
    ap = 0.0
    for t in [i / 10 for i in range(11)]:
        prec_at = [p for p, r in zip(precs, recs) if r >= t]
        ap += max(prec_at) if prec_at else 0.0
    return ap / 11.0


def _pck(
    pred_kps: list[dict[str, Any]],
    gt_kps: list[dict[str, Any]],
    thresh: float = 0.2,
) -> float:
    gt_map = {k["name"]: k for k in gt_kps if isinstance(k, dict) and "name" in k}
    pred_map = {k["name"]: k for k in pred_kps if isinstance(k, dict) and "name" in k}
    # Normalize threshold by person bbox diagonal if possible, else image diagonal=√2.
    xs = [float(k["x"]) for k in gt_kps if "x" in k]
    ys = [float(k["y"]) for k in gt_kps if "y" in k]
    if xs and ys:
        diag = ((max(xs) - min(xs)) ** 2 + (max(ys) - min(ys)) ** 2) ** 0.5
        diag = max(diag, 1e-3)
    else:
        diag = 2**0.5
    hits = 0
    total = 0
    for name in KEYPOINT_NAMES:
        if name not in gt_map or name not in pred_map:
            continue
        g, p = gt_map[name], pred_map[name]
        dist = ((float(g["x"]) - float(p["x"])) ** 2 + (float(g["y"]) - float(p["y"])) ** 2) ** 0.5
        total += 1
        if dist <= thresh * diag:
            hits += 1
    return hits / total if total else 0.0


def compute_vision_metrics(row_results: Sequence[Any]) -> dict[str, float]:
    """Score a list of harness ``RowEval`` (or duck-typed equivalents)."""
    n = len(row_results)
    if n == 0:
        return {
            "scene_accuracy": 0.0,
            "map_50": 0.0,
            "pck_0_2": 0.0,
            "all_layers_pass_rate": 0.0,
        }

    scene_correct = 0
    pck_sum = 0.0
    layers_ok = 0
    # Collect detections per class across the whole set for mAP.
    preds_by_cls: dict[str, list[tuple[float, list[float]]]] = {c: [] for c in SHAPE_LABELS}
    gts_by_cls: dict[str, list[list[float]]] = {c: [] for c in SHAPE_LABELS}
    # For image-level mAP we accumulate all preds/gts with image ids — simple
    # VOC-style: concatenate all boxes per class (approx for synthetic small sets).
    per_image_aps: list[float] = []

    for item in row_results:
        row = item.row if hasattr(item, "row") else item.get("row", {})
        gen = item.gen_text if hasattr(item, "gen_text") else item.get("gen_text", "")
        result = item.result if hasattr(item, "result") else None
        if result is not None and getattr(result, "ok", False):
            layers_ok += 1

        expected = row.get("expected") or {}
        pred = _parse_output(gen) or {}

        exp_scene = (expected.get("scene") or {}).get("label")
        pred_scene = (pred.get("scene") or {}).get("label")
        if exp_scene is not None and exp_scene == pred_scene:
            scene_correct += 1

        exp_dets = expected.get("detections") or []
        pred_dets = pred.get("detections") or []
        # Per-image mAP across classes present
        class_aps = []
        for cls in SHAPE_LABELS:
            gts = [list(d["box"]) for d in exp_dets if d.get("label") == cls]
            preds = [
                (float(d.get("confidence", 0.0)), list(d["box"]))
                for d in pred_dets
                if d.get("label") == cls and isinstance(d.get("box"), list)
            ]
            if gts or preds:
                class_aps.append(_ap_for_class(preds, gts, 0.5))
            preds_by_cls[cls].extend(preds)
            gts_by_cls[cls].extend(gts)
        per_image_aps.append(sum(class_aps) / len(class_aps) if class_aps else 1.0)

        exp_pose = (expected.get("pose") or {}).get("keypoints") or []
        pred_pose = (pred.get("pose") or {}).get("keypoints") or []
        pck_sum += _pck(pred_pose, exp_pose, 0.2)

    # Global class-mean AP (also informative)
    global_aps = []
    for cls in SHAPE_LABELS:
        if gts_by_cls[cls] or preds_by_cls[cls]:
            global_aps.append(_ap_for_class(preds_by_cls[cls], gts_by_cls[cls], 0.5))

    return {
        "scene_accuracy": scene_correct / n,
        "map_50": (sum(per_image_aps) / n) if n else 0.0,
        "map_50_global": (sum(global_aps) / len(global_aps)) if global_aps else 0.0,
        "pck_0_2": pck_sum / n,
        "all_layers_pass_rate": layers_ok / n,
    }
