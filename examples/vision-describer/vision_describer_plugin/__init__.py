"""Vision-describer example plugin, validator, metrics, generator."""
from __future__ import annotations

from maatml.registry import register_generator, register_metrics, register_validator

from .generator import vision_describer_generator
from .metrics import compute_vision_describer_metrics
from .validator import validate_vision_describer

register_validator("vision_describer")(validate_vision_describer)
register_metrics("vision_describer")(compute_vision_describer_metrics)
register_generator("vision_describer")(vision_describer_generator)

__all__ = [
    "compute_vision_describer_metrics",
    "validate_vision_describer",
    "vision_describer_generator",
]
