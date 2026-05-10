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
        source="hand:starter",
        category="missing_dd",
        request="//RUNJOB JOB (ACCT)\n//STEP01 EXEC PGM=IEBGENER\n",
        expected_validation_result=JclValidationResult(
            valid=False,
            errors=[
                JclError(
                    line=2,
                    severity=Severity.error,
                    code=ErrorCategory.missing_dd,
                    message="IEBGENER step missing required DD statements.",
                    suggestion="Add SYSUT1 and SYSUT2.",
                )
            ],
            confidence=0.91,
        ),
        split=Split.train,
    )
    payload = sample.model_dump(mode="json")
    again = JclSample.model_validate(payload)
    assert again == sample


def test_jcl_sample_rejects_invalid_validation_result() -> None:
    with pytest.raises(ValidationError):
        JclSample(
            sample_id="abc",
            source="hand:starter",
            category="missing_dd",
            request="x",
            expected_validation_result={"valid": True, "errors": [], "confidence": 1.5},  # bad confidence
            split=Split.train,
        )


def test_jcl_error_line_must_be_positive() -> None:
    with pytest.raises(ValidationError):
        JclError(
            line=0,
            severity=Severity.error,
            code=ErrorCategory.missing_dd,
            message="x",
        )


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
