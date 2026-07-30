from __future__ import annotations

import pytest
from pydantic import ValidationError

from jcl_plugin.schemas import (
    ErrorCategory,
    JclError,
    JclSample,
    JclValidationResult,
    Severity,
)
from maatml.data.schemas import Split


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
            expected_validation_result={"valid": True, "errors": [], "confidence": 1.5},
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


def test_validator_rejects_non_object_json_without_raising() -> None:
    """A valid-JSON non-object has no fields to check. It must score as a failed
    row, not raise AttributeError and abort the whole evaluate/distill run."""
    from pathlib import Path

    from jcl_plugin.validator import validate_jcl_result

    root = Path(__file__).resolve().parents[1]
    schema = root / "datasets" / "jcl_validation_schema.json"
    contracts = root / "datasets" / "node_contracts.json"

    for payload in ('[]', '"a string"', "42", "null", "true"):
        result = validate_jcl_result(
            payload, schema_path=schema, contracts_path=contracts
        )
        assert result.ok is False, payload
        assert any(e.code == "not_an_object" for e in result.errors), payload
