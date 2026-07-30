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


def test_validator_rejects_non_object_json_without_raising() -> None:
    """A valid-JSON non-object has no fields to check. It must score as a failed
    row, not raise AttributeError and abort the whole evaluate/distill run."""
    from pathlib import Path

    from spool_plugin.validator import validate_spool_result

    root = Path(__file__).resolve().parents[1]
    schema = root / "datasets" / "spool_interpretation_schema.json"
    contracts = root / "datasets" / "node_contracts.json"

    for payload in ('[]', '"a string"', "42", "null", "true"):
        result = validate_spool_result(
            payload, schema_path=schema, contracts_path=contracts
        )
        assert result.ok is False, payload
        assert any(e.code == "not_an_object" for e in result.errors), payload
