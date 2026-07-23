"""Torch Dataset over prepared JSONL + image paths + CenterNet target encoding."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from .model import MultitaskConfig


def _gaussian2d(h: int, w: int, cx: float, cy: float, sigma: float):
    import numpy as np

    ys, xs = np.ogrid[0:h, 0:w]
    g = np.exp(-((xs - cx) ** 2 + (ys - cy) ** 2) / (2 * sigma**2))
    return g


def encode_targets(
    expected: dict[str, Any],
    cfg: MultitaskConfig,
) -> dict[str, Any]:
    """Encode expected multitask labels into CenterNet / CE / pose tensors (numpy)."""
    import numpy as np

    hm_size = cfg.heatmap_size
    n_shape = len(cfg.shape_labels)
    heatmaps = np.zeros((n_shape, hm_size, hm_size), dtype=np.float32)
    sizes = np.zeros((2, hm_size, hm_size), dtype=np.float32)
    offsets = np.zeros((2, hm_size, hm_size), dtype=np.float32)
    center_mask = np.zeros((1, hm_size, hm_size), dtype=np.float32)

    label_to_idx = {label: i for i, label in enumerate(cfg.shape_labels)}
    for det in expected.get("detections") or []:
        label = det.get("label")
        box = det.get("box")
        if label not in label_to_idx or not isinstance(box, list) or len(box) != 4:
            continue
        x1, y1, x2, y2 = [float(v) for v in box]
        cx = (x1 + x2) / 2.0 * hm_size
        cy = (y1 + y2) / 2.0 * hm_size
        bw = max(1e-3, (x2 - x1) * hm_size)
        bh = max(1e-3, (y2 - y1) * hm_size)
        ix, iy = int(cx), int(cy)
        if not (0 <= ix < hm_size and 0 <= iy < hm_size):
            continue
        ci = label_to_idx[label]
        sigma = max(1.0, (bw + bh) / 6.0)
        g = _gaussian2d(hm_size, hm_size, cx, cy, sigma)
        heatmaps[ci] = np.maximum(heatmaps[ci], g.astype(np.float32))
        sizes[0, iy, ix] = bw
        sizes[1, iy, ix] = bh
        offsets[0, iy, ix] = cx - ix
        offsets[1, iy, ix] = cy - iy
        center_mask[0, iy, ix] = 1.0

    scene_label = (expected.get("scene") or {}).get("label")
    scene_idx = (
        cfg.scene_labels.index(scene_label)
        if scene_label in cfg.scene_labels
        else 0
    )

    kps = {k["name"]: k for k in (expected.get("pose") or {}).get("keypoints") or []}
    pose = []
    for name in cfg.keypoint_names:
        k = kps.get(name) or {}
        pose.extend([float(k.get("x", 0.0)), float(k.get("y", 0.0))])

    return {
        "scene_idx": scene_idx,
        "heatmaps": heatmaps,
        "sizes": sizes,
        "offsets": offsets,
        "center_mask": center_mask,
        "pose_coords": np.asarray(pose, dtype=np.float32),
    }


def _load_image(path: Path, size: int):
    from PIL import Image
    import numpy as np

    img = Image.open(path).convert("RGB")
    if img.size != (size, size):
        img = img.resize((size, size), Image.BILINEAR)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    # ImageNet normalize
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    arr = (arr - mean) / std
    return arr.transpose(2, 0, 1)  # CHW


def resolve_image_bytes_or_path(
    value: Any,
    *,
    model_dir: Optional[Path] = None,
) -> bytes:
    """Accept path, data-URI, or raw base64; return PNG/JPEG bytes."""
    import base64
    import re

    if isinstance(value, (bytes, bytearray)):
        return bytes(value)
    if not isinstance(value, str):
        raise TypeError(f"image must be str/bytes; got {type(value)}")
    if value.startswith("data:"):
        m = re.match(r"data:image/[^;]+;base64,(.+)$", value, re.DOTALL)
        if not m:
            raise ValueError("Invalid data-URI image")
        return base64.b64decode(m.group(1))
    # Heuristic: base64 blob (no path separators, long)
    if "/" not in value and "\\" not in value and len(value) > 200:
        try:
            return base64.b64decode(value, validate=False)
        except Exception:  # noqa: BLE001
            pass
    # Otherwise treat as a filesystem path, confined to model_dir. This runs on
    # client-controlled input at serve time, so reject absolute and '..' paths.
    if model_dir is None:
        raise ValueError(
            f"image path {value!r} given but no model_dir to resolve it against"
        )
    rel = Path(value)
    if rel.is_absolute() or ".." in rel.parts:
        raise ValueError(
            f"unsafe image path {value!r}: must be relative to the model "
            "directory with no '..' segments"
        )
    root = Path(model_dir).resolve()
    resolved = (root / rel).resolve()
    if resolved != root and root not in resolved.parents:
        raise ValueError(f"unsafe image path {value!r}: escapes the model directory")
    if not resolved.is_file():
        raise FileNotFoundError(f"Image not found: {value}")
    return resolved.read_bytes()


def image_bytes_to_tensor(data: bytes, size: int):
    import io
    from PIL import Image
    import numpy as np
    import torch

    img = Image.open(io.BytesIO(data)).convert("RGB")
    if img.size != (size, size):
        img = img.resize((size, size), Image.BILINEAR)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    arr = (arr - mean) / std
    return torch.from_numpy(arr.transpose(2, 0, 1))


class VisionSceneDataset:
    """Iterable-style torch Dataset factory (returns nn Dataset when torch present)."""

    @staticmethod
    def build(
        rows: list[dict[str, Any]],
        *,
        model_dir: Path,
        cfg: MultitaskConfig,
        limit: Optional[int] = None,
    ):
        import torch
        from torch.utils.data import Dataset

        class _DS(Dataset):
            def __init__(self) -> None:
                self.rows = rows[:limit] if limit else list(rows)
                self.model_dir = Path(model_dir)
                self.cfg = cfg

            def __len__(self) -> int:
                return len(self.rows)

            def __getitem__(self, idx: int) -> dict[str, Any]:
                row = self.rows[idx]
                rel = row.get("image")
                path = self.model_dir / rel if not Path(rel).is_absolute() else Path(rel)
                image = torch.from_numpy(_load_image(path, self.cfg.image_size))
                targets = encode_targets(row.get("expected") or {}, self.cfg)
                return {
                    "image": image,
                    "scene_idx": torch.tensor(targets["scene_idx"], dtype=torch.long),
                    "heatmaps": torch.from_numpy(targets["heatmaps"]),
                    "sizes": torch.from_numpy(targets["sizes"]),
                    "offsets": torch.from_numpy(targets["offsets"]),
                    "center_mask": torch.from_numpy(targets["center_mask"]),
                    "pose_coords": torch.from_numpy(targets["pose_coords"]),
                    "sample_id": row.get("sample_id", str(idx)),
                }

        return _DS()


def collate_vision(batch: list[dict[str, Any]]) -> dict[str, Any]:
    import torch

    return {
        "image": torch.stack([b["image"] for b in batch], dim=0),
        "scene_idx": torch.stack([b["scene_idx"] for b in batch], dim=0),
        "heatmaps": torch.stack([b["heatmaps"] for b in batch], dim=0),
        "sizes": torch.stack([b["sizes"] for b in batch], dim=0),
        "offsets": torch.stack([b["offsets"] for b in batch], dim=0),
        "center_mask": torch.stack([b["center_mask"] for b in batch], dim=0),
        "pose_coords": torch.stack([b["pose_coords"] for b in batch], dim=0),
        "sample_id": [b["sample_id"] for b in batch],
    }
