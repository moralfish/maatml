"""Shared constants for the vision-vlm example."""
from __future__ import annotations

SCENE_LABELS: list[str] = ["plain", "gradient", "striped", "noisy", "checker"]
SHAPE_LABELS: list[str] = ["circle", "square", "triangle", "star"]
KEYPOINT_NAMES: list[str] = [
    "head",
    "neck",
    "l_shoulder",
    "r_shoulder",
    "l_elbow",
    "r_elbow",
    "l_wrist",
    "r_wrist",
    "hip",
    "l_knee",
    "r_knee",
    "feet",
]

DEFAULT_IMAGE_SIZE = 320
