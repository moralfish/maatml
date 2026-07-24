"""Vision-VLM example plugin, SmolVLM SFT + vLLM-oriented serving."""
from __future__ import annotations

from maatml.registry import (
    register_generator,
    register_metrics,
    register_predictor,
    register_trainer,
    register_validator,
)

from .datagen import described_scenes_generator
from .metrics import compute_vision_vlm_metrics
from .predictor import VisionVlmPredictor
from .scaffold import scaffold_vlm_sft  # noqa: F401  registers @register_scaffold_hook
from .trainer import train_vlm_sft
from .validator import validate_vision_vlm

register_trainer("vlm_sft")(train_vlm_sft)
register_predictor("vision_vlm")(VisionVlmPredictor)
register_validator("vision_vlm")(validate_vision_vlm)
register_metrics("vision_vlm")(compute_vision_vlm_metrics)
register_generator("described_scenes")(described_scenes_generator)

__all__ = [
    "train_vlm_sft",
    "VisionVlmPredictor",
    "validate_vision_vlm",
    "compute_vision_vlm_metrics",
    "described_scenes_generator",
    "scaffold_vlm_sft",
]
