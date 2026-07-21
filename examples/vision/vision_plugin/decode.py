"""Pure-numpy decode for CenterNet-style detection + pose + scene heads.

Shared by the maatml predictor and the Jetson/deploy client so ONNX graphs
stay free of NonMaxSuppression.
"""
from __future__ import annotations

from typing import Any, Optional

import math


def _softmax(logits: list[float]) -> list[float]:
    m = max(logits) if logits else 0.0
    exps = [math.exp(x - m) for x in logits]
    s = sum(exps) or 1.0
    return [e / s for e in exps]


def decode_scene(
    logits: Any,
    scene_labels: list[str],
) -> dict[str, Any]:
    """``logits``: 1D array-like of length C."""
    vals = [float(x) for x in list(logits)]
    probs = _softmax(vals)
    idx = max(range(len(probs)), key=lambda i: probs[i]) if probs else 0
    label = scene_labels[idx] if idx < len(scene_labels) else str(idx)
    return {"label": label, "confidence": float(probs[idx]) if probs else 0.0}


def _nms_xyxy(boxes: list[list[float]], scores: list[float], iou_thresh: float) -> list[int]:
    """Greedy NMS on xyxy boxes in [0,1]. Returns kept indices."""
    if not boxes:
        return []
    order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    keep: list[int] = []

    def iou(a: list[float], b: list[float]) -> float:
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        ix1, iy1 = max(ax1, bx1), max(ay1, by1)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
        inter = iw * ih
        area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
        area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
        union = area_a + area_b - inter + 1e-9
        return inter / union

    while order:
        i = order.pop(0)
        keep.append(i)
        order = [j for j in order if iou(boxes[i], boxes[j]) < iou_thresh]
    return keep


def decode_detections(
    heatmaps: Any,
    sizes: Any,
    offsets: Any,
    shape_labels: list[str],
    *,
    score_thresh: float = 0.25,
    top_k: int = 50,
    nms_iou: float = 0.4,
) -> list[dict[str, Any]]:
    """Decode CenterNet outputs.

    heatmaps: (C, H, W) sigmoid-ready logits or probabilities
    sizes:    (2, H, W) width/height in heatmap cell units
    offsets:  (2, H, W) center offsets in heatmap cell units
    """
    import numpy as np

    hm = np.asarray(heatmaps, dtype=np.float64)
    sz = np.asarray(sizes, dtype=np.float64)
    off = np.asarray(offsets, dtype=np.float64)
    if hm.ndim != 3:
        raise ValueError(f"heatmaps must be (C,H,W); got {hm.shape}")
    # Sigmoid if looks like logits.
    if hm.min() < 0.0 or hm.max() > 1.0:
        hm = 1.0 / (1.0 + np.exp(-np.clip(hm, -50.0, 50.0)))

    c, h, w = hm.shape
    # 3×3 max-pool peak picking
    peaks: list[tuple[float, int, int, int]] = []
    for ci in range(c):
        plane = hm[ci]
        for y in range(h):
            for x in range(w):
                v = float(plane[y, x])
                if v < score_thresh:
                    continue
                y0, y1 = max(0, y - 1), min(h, y + 2)
                x0, x1 = max(0, x - 1), min(w, x + 2)
                if v >= float(plane[y0:y1, x0:x1].max()) - 1e-12:
                    peaks.append((v, ci, y, x))
    peaks.sort(reverse=True)
    peaks = peaks[:top_k]

    boxes: list[list[float]] = []
    scores: list[float] = []
    labels: list[str] = []
    for score, ci, y, x in peaks:
        ox = float(off[0, y, x])
        oy = float(off[1, y, x])
        bw = float(sz[0, y, x])
        bh = float(sz[1, y, x])
        cx = (x + ox) / w
        cy = (y + oy) / h
        # size is in heatmap cells
        ww = max(1e-4, bw / w)
        hh = max(1e-4, bh / h)
        x1 = max(0.0, min(1.0, cx - ww / 2))
        y1 = max(0.0, min(1.0, cy - hh / 2))
        x2 = max(0.0, min(1.0, cx + ww / 2))
        y2 = max(0.0, min(1.0, cy + hh / 2))
        boxes.append([x1, y1, x2, y2])
        scores.append(score)
        labels.append(shape_labels[ci] if ci < len(shape_labels) else str(ci))

    keep = _nms_xyxy(boxes, scores, nms_iou)
    return [
        {"label": labels[i], "box": boxes[i], "confidence": float(scores[i])}
        for i in keep
    ]


def decode_pose(
    coords: Any,
    keypoint_names: list[str],
    *,
    confidences: Optional[Any] = None,
) -> dict[str, Any]:
    """``coords``: flat (K*2,) or (K,2) normalized xy in [0,1] (or logits clipped)."""
    import numpy as np

    arr = np.asarray(coords, dtype=np.float64).reshape(-1)
    k = len(keypoint_names)
    if arr.size < k * 2:
        arr = np.pad(arr, (0, k * 2 - arr.size))
    pts = arr[: k * 2].reshape(k, 2)
    # Soft-clip to [0,1] if slightly out of range.
    pts = np.clip(pts, -0.05, 1.05)
    confs: list[float]
    if confidences is not None:
        confs = [float(c) for c in list(np.asarray(confidences).reshape(-1)[:k])]
        while len(confs) < k:
            confs.append(1.0)
    else:
        confs = [1.0] * k
    keypoints = []
    for i, name in enumerate(keypoint_names):
        keypoints.append(
            {
                "name": name,
                "x": float(pts[i, 0]),
                "y": float(pts[i, 1]),
                "confidence": float(confs[i]),
            }
        )
    return {"keypoints": keypoints}


def decode_multitask_outputs(
    *,
    scene_logits: Any,
    heatmaps: Any,
    sizes: Any,
    offsets: Any,
    pose_coords: Any,
    scene_labels: list[str],
    shape_labels: list[str],
    keypoint_names: list[str],
    score_thresh: float = 0.25,
) -> dict[str, Any]:
    return {
        "scene": decode_scene(scene_logits, scene_labels),
        "detections": decode_detections(
            heatmaps,
            sizes,
            offsets,
            shape_labels,
            score_thresh=score_thresh,
        ),
        "pose": decode_pose(pose_coords, keypoint_names),
    }
