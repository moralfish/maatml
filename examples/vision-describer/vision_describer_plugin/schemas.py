"""Pydantic sample + result types for the vision describer."""
from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, ConfigDict, Field

from maatml.data.schemas import Split

from .constants import MAX_DESCRIPTION_WORDS


class DescriptionResult(BaseModel):
    """Model output shape: a single short factual description."""

    model_config = ConfigDict(extra="forbid")

    description: str = Field(min_length=1, max_length=240)


class VisionDescriberSample(BaseModel):
    """Training / eval sample row."""

    model_config = ConfigDict(extra="forbid")

    sample_id: str
    source: str
    category: str
    request: str
    expected_description: DescriptionResult
    split: Optional[Split] = None
    family: Optional[str] = None

    def word_count(self) -> int:
        return len(self.expected_description.description.split())

    def within_word_limit(self) -> bool:
        return self.word_count() <= MAX_DESCRIPTION_WORDS
