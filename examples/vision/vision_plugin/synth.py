"""Deterministic synthetic scene renderer with exact multitask ground truth.

Each scene yields independent supervision for:
  - scene classification (background style)
  - object detection (0–3 colored shapes with boxes)
  - single-person pose (one stick figure, 12 keypoints)
"""
from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from .constants import KEYPOINT_NAMES, SCENE_LABELS, SHAPE_LABELS


def _rng(seed: int) -> Any:
    """Return a seeded random.Random (stdlib, no numpy required for synth)."""
    import random

    return random.Random(seed)


def _stable_seed(*parts: Any) -> int:
    h = hashlib.sha256("|".join(str(p) for p in parts).encode("utf-8")).hexdigest()
    return int(h[:12], 16)


@dataclass(frozen=True)
class SceneSpec:
    sample_id: str
    family: str
    scene: str
    shapes: list[dict[str, Any]]
    pose: dict[str, Any]
    seed: int


def _draw_background(img: Any, draw: Any, scene: str, size: int, rng: Any) -> None:

    if scene == "plain":
        color = tuple(rng.randint(40, 220) for _ in range(3))
        draw.rectangle([0, 0, size, size], fill=color)
    elif scene == "gradient":
        for y in range(size):
            t = y / max(1, size - 1)
            c = (
                int(30 + 180 * t),
                int(80 + 100 * (1 - t)),
                int(200 - 120 * t),
            )
            draw.line([(0, y), (size, y)], fill=c)
    elif scene == "striped":
        band = max(8, size // 16)
        c1 = (rng.randint(20, 100), rng.randint(20, 100), rng.randint(20, 100))
        c2 = (rng.randint(150, 240), rng.randint(150, 240), rng.randint(150, 240))
        for y in range(0, size, band):
            draw.rectangle([0, y, size, min(size, y + band)], fill=c1 if (y // band) % 2 == 0 else c2)
    elif scene == "noisy":
        # Coarse noise tiles (fast + deterministic).
        tile = max(4, size // 40)
        for y in range(0, size, tile):
            for x in range(0, size, tile):
                c = tuple(rng.randint(0, 255) for _ in range(3))
                draw.rectangle([x, y, min(size, x + tile), min(size, y + tile)], fill=c)
    elif scene == "checker":
        cell = max(10, size // 8)
        c1 = (30, 30, 30)
        c2 = (220, 220, 220)
        for y in range(0, size, cell):
            for x in range(0, size, cell):
                fill = c1 if ((x // cell) + (y // cell)) % 2 == 0 else c2
                draw.rectangle([x, y, min(size, x + cell), min(size, y + cell)], fill=fill)
    else:
        draw.rectangle([0, 0, size, size], fill=(128, 128, 128))


def _draw_shape(
    draw: Any,
    label: str,
    cx: float,
    cy: float,
    radius: float,
    color: tuple[int, int, int],
) -> tuple[float, float, float, float]:
    """Draw shape; return axis-aligned box as xyxy in pixel coords."""
    if label == "circle":
        box = (cx - radius, cy - radius, cx + radius, cy + radius)
        draw.ellipse(box, fill=color)
        return box
    if label == "square":
        box = (cx - radius, cy - radius, cx + radius, cy + radius)
        draw.rectangle(box, fill=color)
        return box
    if label == "triangle":
        pts = [
            (cx, cy - radius),
            (cx - radius, cy + radius),
            (cx + radius, cy + radius),
        ]
        draw.polygon(pts, fill=color)
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        return (min(xs), min(ys), max(xs), max(ys))
    # star
    pts = []
    for i in range(10):
        ang = -math.pi / 2 + i * math.pi / 5
        r = radius if i % 2 == 0 else radius * 0.45
        pts.append((cx + r * math.cos(ang), cy + r * math.sin(ang)))
    draw.polygon(pts, fill=color)
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    return (min(xs), min(ys), max(xs), max(ys))


def _make_pose(rng: Any, size: int) -> dict[str, Any]:
    """Generate a posable stick figure with 12 keypoints in pixel coords."""
    # Root near center-ish.
    hip_x = rng.uniform(size * 0.3, size * 0.7)
    hip_y = rng.uniform(size * 0.45, size * 0.65)
    scale = rng.uniform(size * 0.12, size * 0.22)
    lean = rng.uniform(-0.25, 0.25)

    neck_x = hip_x + lean * scale
    neck_y = hip_y - 1.4 * scale
    head_x = neck_x + lean * scale * 0.3
    head_y = neck_y - 0.55 * scale

    l_sh_x = neck_x - 0.55 * scale
    r_sh_x = neck_x + 0.55 * scale
    sh_y = neck_y + 0.05 * scale

    arm_spread = rng.uniform(0.2, 1.0)
    l_el_x = l_sh_x - arm_spread * 0.45 * scale
    r_el_x = r_sh_x + arm_spread * 0.45 * scale
    el_y = sh_y + 0.7 * scale
    l_wr_x = l_el_x - arm_spread * 0.35 * scale
    r_wr_x = r_el_x + arm_spread * 0.35 * scale
    wr_y = el_y + 0.65 * scale

    stance = rng.uniform(0.3, 0.9)
    l_kn_x = hip_x - stance * 0.35 * scale
    r_kn_x = hip_x + stance * 0.35 * scale
    kn_y = hip_y + 0.9 * scale
    feet_x = hip_x
    feet_y = kn_y + 0.85 * scale

    coords = {
        "head": (head_x, head_y),
        "neck": (neck_x, neck_y),
        "l_shoulder": (l_sh_x, sh_y),
        "r_shoulder": (r_sh_x, sh_y),
        "l_elbow": (l_el_x, el_y),
        "r_elbow": (r_el_x, el_y),
        "l_wrist": (l_wr_x, wr_y),
        "r_wrist": (r_wr_x, wr_y),
        "hip": (hip_x, hip_y),
        "l_knee": (l_kn_x, kn_y),
        "r_knee": (r_kn_x, kn_y),
        "feet": (feet_x, feet_y),
    }
    keypoints = []
    for name in KEYPOINT_NAMES:
        x, y = coords[name]
        x = float(min(size - 1, max(0, x)))
        y = float(min(size - 1, max(0, y)))
        keypoints.append({"name": name, "x": x, "y": y, "confidence": 1.0})
    return {"keypoints": keypoints}


def _draw_pose(draw: Any, pose: dict[str, Any], color: tuple[int, int, int] = (255, 40, 40)) -> None:
    kp = {k["name"]: (k["x"], k["y"]) for k in pose["keypoints"]}
    bones = [
        ("head", "neck"),
        ("neck", "l_shoulder"),
        ("neck", "r_shoulder"),
        ("l_shoulder", "l_elbow"),
        ("l_elbow", "l_wrist"),
        ("r_shoulder", "r_elbow"),
        ("r_elbow", "r_wrist"),
        ("neck", "hip"),
        ("hip", "l_knee"),
        ("hip", "r_knee"),
        ("l_knee", "feet"),
        ("r_knee", "feet"),
    ]
    for a, b in bones:
        draw.line([kp[a], kp[b]], fill=color, width=3)
    for name in KEYPOINT_NAMES:
        x, y = kp[name]
        r = 4
        draw.ellipse((x - r, y - r, x + r, y + r), fill=(255, 220, 0))


def make_scene_spec(index: int, *, base_seed: int = 0, size: int = 320) -> SceneSpec:
    seed = _stable_seed(base_seed, index, "scene")
    rng = _rng(seed)
    scene = SCENE_LABELS[index % len(SCENE_LABELS)]
    # Mix scene labels within each family so group-aware splits don't
    # quarantine an entire class into train or test.
    family = f"batch_{index // 10}"
    n_shapes = rng.randint(0, 3)
    shapes: list[dict[str, Any]] = []
    for i in range(n_shapes):
        label = SHAPE_LABELS[rng.randrange(len(SHAPE_LABELS))]
        radius = rng.uniform(size * 0.06, size * 0.14)
        # Keep shapes away from the stick figure center band.
        cx = rng.uniform(radius + 4, size - radius - 4)
        cy = rng.uniform(radius + 4, size * 0.4)
        color = (rng.randint(20, 255), rng.randint(20, 255), rng.randint(20, 255))
        shapes.append(
            {
                "label": label,
                "cx": cx,
                "cy": cy,
                "radius": radius,
                "color": color,
            }
        )
    pose = _make_pose(rng, size)
    sample_id = f"syn-{scene}-{index:04d}-{seed & 0xFFFF:04x}"
    return SceneSpec(
        sample_id=sample_id,
        family=family,
        scene=scene,
        shapes=shapes,
        pose=pose,
        seed=seed,
    )


def render_scene(
    spec: SceneSpec,
    *,
    size: int = 320,
    out_path: Optional[Path] = None,
) -> tuple[Any, dict[str, Any]]:
    """Render scene PNG and return ``(PIL.Image, expected_dict)``."""
    from PIL import Image, ImageDraw

    rng = _rng(spec.seed ^ 0xA5A5)
    img = Image.new("RGB", (size, size), (0, 0, 0))
    draw = ImageDraw.Draw(img)
    _draw_background(img, draw, spec.scene, size, rng)

    detections: list[dict[str, Any]] = []
    for shape in spec.shapes:
        box = _draw_shape(
            draw,
            shape["label"],
            shape["cx"],
            shape["cy"],
            shape["radius"],
            tuple(shape["color"]),
        )
        x1, y1, x2, y2 = box
        # Clamp + normalize to [0,1] xyxy for the expected payload.
        x1 = max(0.0, min(float(size), float(x1)))
        y1 = max(0.0, min(float(size), float(y1)))
        x2 = max(0.0, min(float(size), float(x2)))
        y2 = max(0.0, min(float(size), float(y2)))
        detections.append(
            {
                "label": shape["label"],
                "box": [x1 / size, y1 / size, x2 / size, y2 / size],
                "confidence": 1.0,
            }
        )

    # Pose keypoints stored normalized in expected payload.
    pose_norm = {
        "keypoints": [
            {
                "name": k["name"],
                "x": k["x"] / size,
                "y": k["y"] / size,
                "confidence": 1.0,
            }
            for k in spec.pose["keypoints"]
        ]
    }
    _draw_pose(draw, spec.pose)

    expected = {
        "scene": {"label": spec.scene, "confidence": 1.0},
        "detections": detections,
        "pose": pose_norm,
    }
    if out_path is not None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        img.save(out_path)
    return img, expected


def build_sample_row(
    index: int,
    *,
    base_seed: int = 0,
    size: int = 320,
    image_rel: str,
    images_dir: Path,
) -> dict[str, Any]:
    """Render one sample and return a seed JSONL row."""
    spec = make_scene_spec(index, base_seed=base_seed, size=size)
    out_path = Path(images_dir) / f"{spec.sample_id}.png"
    _, expected = render_scene(spec, size=size, out_path=out_path)
    return {
        "sample_id": spec.sample_id,
        "source": "synthetic:scene",
        "family": spec.family,
        "category": spec.scene,
        "image": image_rel.replace("{id}", spec.sample_id)
        if "{id}" in image_rel
        else f"{image_rel.rstrip('/')}/{spec.sample_id}.png",
        "expected": expected,
    }
