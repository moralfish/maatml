"""JCL example plugin, validators, metrics, predictor, sanitizer, transform."""
from __future__ import annotations

from pathlib import Path

from maatml.data.sanitizer import make_tag_sanitizer
from maatml.registry import (
    register_generator,
    register_metrics,
    register_predictor,
    register_sanitizer,
    register_transform,
    register_validator,
)

from .datagen import jcl_generator
from .metrics import compute_jcl_metrics
from .predictor import JclClassifierPredictor
from .tokenizer import pre_tokenize_jcl
from .validator import validate_jcl_result

_RULES = Path(__file__).resolve().parent / "sanitization.yaml"

register_validator("jcl")(validate_jcl_result)
register_metrics("jcl")(compute_jcl_metrics)
register_predictor("jcl_classifier")(JclClassifierPredictor)
register_generator("jcl")(jcl_generator)
register_sanitizer("jcl")(
    make_tag_sanitizer(_RULES, tag="jcl", length_preserving_only=True)
)
register_transform("jcl_columns")(pre_tokenize_jcl)

__all__ = [
    "compute_jcl_metrics",
    "validate_jcl_result",
    "pre_tokenize_jcl",
    "JclClassifierPredictor",
]
