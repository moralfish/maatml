"""Shared data enums used by core + example plugins.

Task-specific sample / result schemas live in example plugins
(``jcl_plugin.schemas``, ``spool_plugin.schemas``).
"""
from __future__ import annotations

from enum import Enum


class Split(str, Enum):
    train = "train"
    val = "val"
    test = "test"
