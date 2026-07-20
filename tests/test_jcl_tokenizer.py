"""Fixture-driven tests for the JCL pre-tokenizer.

Uses `jcl_pretokenize_fixtures.json` as the normative expected output for
`pre_tokenize_jcl`. Downstream inference pre-tokenizers should match these
fixtures byte-for-byte.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from flow_ml.tokenization import pre_tokenize_jcl


FIXTURES = (
    Path(__file__).resolve().parent.parent
    / "src"
    / "flow_ml"
    / "tokenization"
    / "fixtures"
    / "jcl_pretokenize_fixtures.json"
)


def _load_fixtures() -> list[dict]:
    data = json.loads(FIXTURES.read_text(encoding="utf-8"))
    return data["fixtures"]


@pytest.mark.parametrize("fixture", _load_fixtures(), ids=lambda f: f["name"])
def test_pretokenize_matches_fixture(fixture: dict) -> None:
    got = pre_tokenize_jcl(fixture["input"])
    assert got == fixture["expected"], (
        f"\n--- input ---\n{fixture['input']!r}\n"
        f"--- expected ---\n{fixture['expected']!r}\n"
        f"--- got ---\n{got!r}\n"
    )
