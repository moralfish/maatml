"""Vision multitask example plugin, trainer, eval, datagen, ONNX export."""
from __future__ import annotations

from maatml.registry import (
    register_generator,
    register_metrics,
    register_predictor,
    register_trainer,
    register_validator,
)

from .datagen import synthetic_scenes_generator
from .export_onnx import export_onnx  # noqa: F401  registers @register_exporter("onnx")
from .metrics import compute_vision_metrics
from .predictor import VisionMultitaskPredictor
from .trainer import train_vision_multitask
from .validator import validate_vision_scene

register_trainer("vision_multitask")(train_vision_multitask)
register_predictor("vision_multitask")(VisionMultitaskPredictor)
register_validator("vision_scene")(validate_vision_scene)
register_metrics("vision_scene")(compute_vision_metrics)
register_generator("synthetic_scenes")(synthetic_scenes_generator)

__all__ = [
    "train_vision_multitask",
    "VisionMultitaskPredictor",
    "validate_vision_scene",
    "compute_vision_metrics",
    "synthetic_scenes_generator",
    "export_onnx",
]
