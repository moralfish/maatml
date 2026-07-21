"""Standalone label/keypoint vocab (mirrors examples/vision; no cross-import)."""
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

# Drop detections / keypoints below this confidence when linearizing.
DEFAULT_SCORE_THRESH: float = 0.3
# Round noisy floats so train/serve share the same compact token distribution.
DEFAULT_NDIGITS: int = 2
# Hard cap for the emitted description (validator + metrics).
MAX_DESCRIPTION_WORDS: int = 30
