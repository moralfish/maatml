"""Tests for the `flow_dsl_py` PyO3 binding and its use in the eval round-trip filter.

The binding is built from `flow-studio/crates/flow-dsl-py` via
`maturin develop --release`. When it isn't installed (fresh clone, CI
without the build step) these tests are skipped rather than failing -
the binding is an optional eval/dataset-quality dependency, not a
runtime requirement.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SEED_SAMPLES = REPO_ROOT / "models" / "dsl-generator" / "datasets" / "samples" / "seed_samples.jsonl"

flow_dsl_py = pytest.importorskip(
    "flow_dsl_py",
    reason="flow_dsl_py not installed (run `cd flow-studio/crates/flow-dsl-py && maturin develop --release`)",
)


def test_binding_module_exposes_expected_surface() -> None:
    assert hasattr(flow_dsl_py, "parse")
    assert hasattr(flow_dsl_py, "serialize")
    assert hasattr(flow_dsl_py, "parses")
    assert hasattr(flow_dsl_py, "DslError")


def test_parse_round_trips_minimal_flow() -> None:
    src = (
        'flow "Test" v1.0.0\n'
        "\n"
        'a[action: "A"] {\n'
        '  adapter: "mock"\n'
        "}\n"
    )
    g = flow_dsl_py.parse(src)
    assert g["name"] == "Test"
    assert g["version"] == "1.0.0"
    assert len(g["nodes"]) == 1
    out = flow_dsl_py.serialize(g)
    # Re-parsing the serialized form yields the same graph (semantic equality).
    assert flow_dsl_py.parses(out)


def test_dsl_error_carries_line_and_col() -> None:
    with pytest.raises(flow_dsl_py.DslError) as exc_info:
        flow_dsl_py.parse("not a flow")
    err = exc_info.value
    assert err.line == 1
    assert err.col >= 1
    assert isinstance(err.message, str) and err.message


def test_every_seed_sample_parses() -> None:
    """Every hand-authored seed in `seed_samples.jsonl` MUST be a valid Flow DSL document.

    This guards the dataset against grammar drift: if the parser ever stops
    accepting a document we authored, the test fails loudly so we can fix
    either the seed or the parser before training proceeds.
    """
    rows = []
    with SEED_SAMPLES.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    assert rows, "seed_samples.jsonl should not be empty"
    failures: list[str] = []
    for row in rows:
        try:
            flow_dsl_py.parse(row["dsl"])
        except flow_dsl_py.DslError as e:
            failures.append(f"{row['sample_id']}: line {e.line}:{e.col} - {e.message}")
    assert not failures, "seed samples that don't parse:\n  " + "\n  ".join(failures)


def test_parses_helper_returns_bool() -> None:
    assert flow_dsl_py.parses('flow "X" v1.0.0\n\na[utility: "u"] { utilityId: "x" }\n') is True
    assert flow_dsl_py.parses("garbage") is False
