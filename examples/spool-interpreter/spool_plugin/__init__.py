"""Spool example plugin — validator, metrics, sanitizer (uses core seq2seq)."""
from __future__ import annotations

from pathlib import Path

from maatml.data.sanitizer import make_tag_sanitizer
from maatml.registry import (
    register_generator,
    register_metrics,
    register_sanitizer,
    register_validator,
)

from .generator import spool_generator
from .metrics import compute_spool_metrics
from .validator import validate_spool_result

_RULES = Path(__file__).resolve().parent / "sanitization.yaml"

register_validator("spool")(validate_spool_result)
register_metrics("spool")(compute_spool_metrics)
register_sanitizer("spool")(make_tag_sanitizer(_RULES, tag="spool"))
register_generator("spool")(spool_generator)

__all__ = ["compute_spool_metrics", "validate_spool_result"]
