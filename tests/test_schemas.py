from __future__ import annotations

import pytest
from pydantic import ValidationError

from flow_ml.data.schemas import (
    ErrorCategory,
    FailureCategory,
    JclError,
    JclSample,
    JclValidationResult,
    Severity,
    SpoolInterpretation,
    SpoolSample,
    Split,
)


def test_jcl_sample_round_trip() -> None:
    sample = JclSample(
        sample_id="abc",
        source="syn",
        sanitized_jcl="//J JOB\n",
        is_valid=False,
        error_category=ErrorCategory.missing_dd,
        error_line=2,
        error_column=1,
        suggestion="add DD",
        split=Split.train,
    )
    payload = sample.model_dump(mode="json")
    again = JclSample.model_validate(payload)
    assert again == sample


def test_jcl_sample_rejects_unknown_category() -> None:
    with pytest.raises(ValidationError):
        JclSample(
            sample_id="abc",
            source="syn",
            sanitized_jcl="x",
            is_valid=False,
            error_category="not_a_category",  # type: ignore[arg-type]
            split=Split.train,
        )


def test_jcl_sample_error_line_must_be_positive() -> None:
    with pytest.raises(ValidationError):
        JclSample(
            sample_id="x",
            source="s",
            sanitized_jcl="y",
            is_valid=False,
            error_line=0,
            split=Split.val,
        )


def test_spool_sample_round_trip() -> None:
    sample = SpoolSample(
        sample_id="s1",
        source="fixture",
        sanitized_spool="JOB ENDED",
        status="failed",
        return_code="0008",
        failure_category=FailureCategory.dataset_resolution_failure,
        root_cause="missing dataset",
        suggested_fix="check catalog",
        split=Split.test,
    )
    payload = sample.model_dump(mode="json")
    again = SpoolSample.model_validate(payload)
    assert again == sample


def test_jcl_validation_result_round_trip() -> None:
    result = JclValidationResult(
        valid=False,
        errors=[
            JclError(
                line=14,
                column=3,
                severity=Severity.error,
                code=ErrorCategory.missing_dd,
                message="missing DD",
                suggestion="add SORTIN",
            )
        ],
        confidence=0.94,
    )
    j = result.model_dump_json()
    assert JclValidationResult.model_validate_json(j) == result


def test_spool_interpretation_confidence_bounds() -> None:
    with pytest.raises(ValidationError):
        SpoolInterpretation(
            summary="x",
            status="failed",
            rootCause="y",
            suggestedFix="z",
            confidence=1.5,
        )
