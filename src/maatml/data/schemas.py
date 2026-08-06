"""Shared data enums used by core + example plugins.

Task-specific sample / result schemas live in example plugins.
"""

from __future__ import annotations

from enum import Enum


class Split(str, Enum):
    train = "train"
    val = "val"
    test = "test"
