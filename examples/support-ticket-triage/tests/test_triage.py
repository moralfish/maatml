"""Support-ticket-triage tests, dependency-free (no torch required)."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = ROOT / "datasets/schema.json"
SEEDS = ROOT / "datasets/samples/seed_samples.jsonl"
BENCH = ROOT / "datasets/samples/test_prompt_set.jsonl"


def _rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


@pytest.fixture(scope="module")
def plugin():
    from triage_plugin import compute_triage_metrics, validate_triage  # noqa: F401

    return True


def test_validate_model_dir() -> None:
    from maatml.scaffold import validate_model_dir

    errors = validate_model_dir(ROOT)
    assert errors == [], errors


def test_routing_contract_covers_every_category(plugin) -> None:
    from triage_plugin.constants import CATEGORIES, ROUTING, TEAMS

    assert set(ROUTING) == set(CATEGORIES)
    assert set(ROUTING.values()) <= set(TEAMS)


@pytest.mark.parametrize("path", [SEEDS, BENCH], ids=["seeds", "benchmark"])
def test_validator_accepts_gold(plugin, path: Path) -> None:
    from triage_plugin.validator import validate_triage

    for row in _rows(path):
        result = validate_triage(json.dumps(row["expected_output"]), schema_path=SCHEMA)
        assert result.ok, (row["sample_id"], [e.__dict__ for e in result.errors])


def test_validator_rejects_bad_json(plugin) -> None:
    from triage_plugin.validator import validate_triage

    result = validate_triage("not json at all", schema_path=SCHEMA)
    assert not result.ok
    assert any(e.code == "invalid_json" for e in result.errors)


def test_validator_rejects_bad_enum(plugin) -> None:
    from triage_plugin.validator import validate_triage

    bad = {"priority": "p9", "category": "billing", "team": "payments", "summary": "x"}
    result = validate_triage(json.dumps(bad), schema_path=SCHEMA)
    assert not result.ok
    assert any(e.code == "schema_error" for e in result.errors)


def test_validator_rejects_misrouted(plugin) -> None:
    from triage_plugin.validator import validate_triage

    # Well-formed and schema-valid, but billing must route to payments.
    misrouted = {"priority": "p2", "category": "billing", "team": "platform", "summary": "x"}
    result = validate_triage(json.dumps(misrouted), schema_path=SCHEMA)
    assert not result.ok
    assert any(e.code == "misrouted" for e in result.errors)
    # The schema layer still passes, this is a contract failure, not a shape one.
    assert 2 in result.passed_layers


def test_validator_rejects_long_summary(plugin) -> None:
    from triage_plugin.constants import MAX_SUMMARY_WORDS
    from triage_plugin.validator import validate_triage

    long_summary = " ".join(["word"] * (MAX_SUMMARY_WORDS + 5))
    row = {"priority": "p3", "category": "other", "team": "general", "summary": long_summary}
    result = validate_triage(json.dumps(row), schema_path=SCHEMA)
    assert not result.ok
    assert any(e.code == "summary_shape" for e in result.errors)


def test_validator_strips_fences(plugin) -> None:
    from triage_plugin.validator import validate_triage

    row = {"priority": "p1", "category": "bug", "team": "platform", "summary": "outage"}
    fenced = "```json\n" + json.dumps(row) + "\n```"
    result = validate_triage(fenced, schema_path=SCHEMA)
    assert result.ok, [e.__dict__ for e in result.errors]


class _Item:
    """Minimal RowEval stand-in for metrics (uses .row and .result only)."""

    def __init__(self, row: dict, result) -> None:
        self.row = row
        self.result = result
        self.gen_text = ""


def test_metrics_perfect_on_gold(plugin) -> None:
    from triage_plugin.metrics import compute_triage_metrics
    from triage_plugin.validator import validate_triage

    items = []
    for row in _rows(BENCH):
        result = validate_triage(json.dumps(row["expected_output"]), schema_path=SCHEMA)
        items.append(_Item(row, result))

    metrics = compute_triage_metrics(items)
    assert metrics["json_parse_rate"] == 1.0
    assert metrics["schema_conformance_rate"] == 1.0
    assert metrics["routing_consistency_rate"] == 1.0
    assert metrics["all_layers_pass_rate"] == 1.0
    assert metrics["category_accuracy"] == 1.0
    assert metrics["team_accuracy"] == 1.0
    assert metrics["priority_accuracy"] == 1.0
    assert metrics["exact_match_rate"] == 1.0


def test_metrics_counts_parse_failure_as_wrong(plugin) -> None:
    from triage_plugin.metrics import compute_triage_metrics
    from triage_plugin.validator import validate_triage

    gold_row = _rows(BENCH)[0]
    bad = _Item(gold_row, validate_triage("garbage", schema_path=SCHEMA))
    metrics = compute_triage_metrics([bad])
    assert metrics["json_parse_rate"] == 0.0
    assert metrics["category_accuracy"] == 0.0
    assert metrics["exact_match_rate"] == 0.0
