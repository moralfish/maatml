"""Scaffold defaults for the plugin-owned ``vision_multitask`` architecture.

Core cannot know this architecture's config shape (heads, image size, loss
weights), so the plugin supplies it. ``maatml scaffold DIR --architecture
vision_multitask --plugin <this folder>`` then produces a folder that
``maatml validate`` accepts and ``maatml datagen`` can fill.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from maatml.registry import register_scaffold_hook

SCENE_LABELS = ["plain", "gradient", "striped", "noisy", "checker"]
SHAPE_LABELS = ["circle", "square", "triangle", "star"]
KEYPOINT_NAMES = [
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

# The canonical schema lives with the example corpus; a scaffolded folder must
# use exactly that document or `maatml datagen` generates rows the validator
# then rejects. The inline copy is the fallback when the plugin is installed
# without the example folder around it (tests keep the two in step).
_CANONICAL_SCHEMA = Path(__file__).resolve().parents[1] / "datasets" / "schema.json"

_FALLBACK_SCHEMA: dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "title": "VisionMultitaskResult",
    "type": "object",
    "required": ["scene", "detections", "pose"],
    "additionalProperties": True,
    "properties": {
        "scene": {
            "type": "object",
            "required": ["label"],
            "properties": {
                "label": {"type": "string", "enum": SCENE_LABELS},
                "confidence": {"type": "number"},
            },
        },
        "detections": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["label", "box"],
                "properties": {
                    "label": {"type": "string", "enum": SHAPE_LABELS},
                    "box": {
                        "type": "array",
                        "minItems": 4,
                        "maxItems": 4,
                        "items": {"type": "number"},
                    },
                    "confidence": {"type": "number"},
                },
            },
        },
        "pose": {
            "type": "object",
            "required": ["keypoints"],
            "properties": {
                "keypoints": {
                    "type": "array",
                    "minItems": len(KEYPOINT_NAMES),
                    "items": {
                        "type": "object",
                        "required": ["name", "x", "y"],
                        "properties": {
                            "name": {"type": "string"},
                            "x": {"type": "number"},
                            "y": {"type": "number"},
                            "confidence": {"type": "number"},
                        },
                    },
                }
            },
        },
    },
}


def scaffold_schema() -> str:
    """The schema text a scaffolded folder should carry."""
    if _CANONICAL_SCHEMA.is_file():
        return _CANONICAL_SCHEMA.read_text(encoding="utf-8")
    return json.dumps(_FALLBACK_SCHEMA, indent=2) + "\n"


@register_scaffold_hook("vision_multitask")
def scaffold_vision_multitask(
    target_dir: Path, *, architecture: str, name: str
) -> dict[str, Any]:
    """Contribute the vision sections, a schema, and an empty seed corpus."""
    del target_dir, architecture, name
    return {
        "model_yml": {
            "base_model": "torchvision/mobilenet_v3_large",
            "dataset": {
                "format": "jsonl_seed",
                "request_field": "image",
                "target_field": "expected",
                "group_by": "family",
                "schema": "datasets/schema.json",
                "seed_samples": "datasets/samples/seed_samples.jsonl",
                "split_ratios": [0.7, 0.15, 0.15],
                "generator": "synthetic_scenes",
                "seed": 42,
            },
            "training": {
                "image_size": 320,
                "backbone": "mobilenet_v3_large",
                "pretrained": True,
                "batch_size": 4,
                "learning_rate": 1.0e-3,
                "weight_decay": 0.0001,
                "epochs": 8,
                "seed": 42,
                "logging_steps": 10,
                "max_steps": -1,
                "score_thresh": 0.25,
                "loss_weights": {"scene": 1.0, "detect": 2.0, "pose": 5.0},
                "heads": {
                    "scene_labels": SCENE_LABELS,
                    "shape_labels": SHAPE_LABELS,
                    "keypoint_names": KEYPOINT_NAMES,
                },
            },
            "smoke": {
                "pretrained": False,
                "batch_size": 2,
                "epochs": 1,
                "max_steps": 4,
                "logging_steps": 1,
            },
            "evaluation": {
                "predictor": "vision_multitask",
                "validator": "vision_scene",
                "metrics": "vision_scene",
                "gates": {"scene_accuracy": 0.85, "map_50": 0.15, "pck_0_2": 0.15},
            },
            "packaging": {
                "max_input_tokens": 1,
                "expected_latency_ms": 30,
                "weights_dtype": "f16",
            },
        },
        # Rows carry rendered images, so the corpus starts empty on purpose:
        # `maatml datagen <dir>` renders scenes and writes validator-gated rows.
        "seed_rows": [],
        "files": {
            "datasets/schema.json": scaffold_schema(),
            "GENERATE.md": (
                "# Next steps\n\n"
                "This folder has no seed rows yet: images are rendered, not\n"
                "hand-written. Generate a validator-gated corpus with\n\n"
                "```bash\n"
                "maatml datagen . --target 200\n"
                "maatml prepare .\n"
                "maatml train . --smoke\n"
                "```\n"
            ),
        },
    }
