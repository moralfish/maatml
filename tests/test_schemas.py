"""Core schema smoke tests (Split enum only)."""
from __future__ import annotations

from maatml.data.schemas import Split


def test_split_values() -> None:
    assert {s.value for s in Split} == {"train", "val", "test"}
