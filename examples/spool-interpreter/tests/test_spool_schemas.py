from __future__ import annotations

import pytest
from pydantic import ValidationError

from maatml.data.schemas import Split
from spool_plugin.schemas import FailureCategory, SpoolInterpretation, SpoolSample


def test_spool_sample_round_trip() -> None:
    sample = SpoolSample(
        sample_id="s1",
        source="fixture",
        category="dataset_resolution_failure",
        request="JOB ENDED\nIEF212I MYJOB STEP1 - DATA SET NOT FOUND",
        expected_interpretation={
            "summary": "Dataset MY.DATA not found.",
            "status": "failed",
            "returnCode": "0008",
            "rootCause": "Catalog has no entry for MY.DATA.",
            "suggestedFix": "Verify dataset name; allocate or restore from backup.",
            "failureCategory": FailureCategory.dataset_resolution_failure.value,
            "confidence": 0.92,
        },
        split=Split.test,
    )
    payload = sample.model_dump(mode="json")
    again = SpoolSample.model_validate(payload)
    assert again == sample


def test_spool_interpretation_confidence_bounds() -> None:
    with pytest.raises(ValidationError):
        SpoolInterpretation(
            summary="x",
            status="failed",
            rootCause="y",
            suggestedFix="z",
            confidence=1.5,
        )
