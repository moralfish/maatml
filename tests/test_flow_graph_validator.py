"""Coverage tests for the 7-layer FlowGraph validator."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from flow_ml.validation import validate_flow_graph

REPO = Path(__file__).resolve().parents[1]
SCHEMA = REPO / "models" / "flow-graph-generator" / "datasets" / "flow_graph_schema.json"
CONTRACTS = REPO / "models" / "flow-graph-generator" / "datasets" / "node_contracts.json"


def _validate(text: str, *, prompt: str | None = None):
    return validate_flow_graph(
        text, schema_path=SCHEMA, contracts_path=CONTRACTS, user_prompt=prompt
    )


def test_layer1_invalid_json() -> None:
    r = _validate("not json {")
    assert 1 not in r.passed_layers
    assert any(e.code == "invalid_json" for e in r.errors)


def test_layer1_strips_markdown_fences() -> None:
    r = _validate(
        '```json\n{"id":"x","name":"X","version":"0.1.0","nodes":[],"edges":[]}\n```'
    )
    assert 1 in r.passed_layers


def test_layer1_strips_qwen3_think_block() -> None:
    """Qwen3 emits <think>...</think> before the answer in reasoning mode.
    The validator must strip it so the JSON parse sees the actual answer.
    """
    text = (
        '<think>\n\nthinking about it...\n\n</think>\n\n'
        '{"id":"x","name":"X","version":"0.1.0","nodes":[],"edges":[]}'
    )
    r = _validate(text)
    assert 1 in r.passed_layers


def test_layer1_strips_empty_think_block() -> None:
    """The empty <think></think> form Qwen3 emits when reasoning is suppressed
    but the chat template still includes the tags."""
    text = (
        '<think>\n\n</think>\n\n'
        '{"id":"x","name":"X","version":"0.1.0","nodes":[],"edges":[]}'
    )
    r = _validate(text)
    assert 1 in r.passed_layers


def test_layer2_schema_violation_missing_required() -> None:
    r = _validate('{"id":"x","name":"X","nodes":[],"edges":[]}')  # missing version
    assert 1 in r.passed_layers
    assert 2 not in r.passed_layers
    assert any(e.code == "schema_error" for e in r.errors)


def test_layer3_unknown_node_type() -> None:
    body = json.dumps({
        "id": "x", "name": "X", "version": "0.1.0",
        "nodes": [{"id": "n1", "type": "rocket", "position": {"x": 0, "y": 0}, "data": {"label": "go"}}],
        "edges": [],
    })
    r = _validate(body)
    assert 3 not in r.passed_layers
    assert any(e.code == "unknown_node_type" for e in r.errors)


def test_layer4_missing_node_reference() -> None:
    body = json.dumps({
        "id": "x", "name": "X", "version": "0.1.0",
        "nodes": [{"id": "n1", "type": "utility", "position": {"x": 0, "y": 0}, "data": {"label": "wait", "actionId": "sleep"}}],
        "edges": [{"id": "e1", "source": "n1", "target": "ghost"}],
    })
    r = _validate(body)
    assert 4 not in r.passed_layers
    assert any(e.code == "missing_node_reference" for e in r.errors)


def test_layer5_invalid_action_triple() -> None:
    body = json.dumps({
        "id": "x", "name": "X", "version": "0.1.0",
        "nodes": [{
            "id": "n1", "type": "action", "position": {"x": 0, "y": 0},
            "data": {"label": "go", "adapter": "shell", "actionId": "rm-rf-bin-bash"}
        }],
        "edges": [],
    })
    r = _validate(body)
    assert 5 not in r.passed_layers
    assert any(e.code == "invalid_action_triple" for e in r.errors)


def test_layer5_invalid_utility_action() -> None:
    body = json.dumps({
        "id": "x", "name": "X", "version": "0.1.0",
        "nodes": [{
            "id": "n1", "type": "utility", "position": {"x": 0, "y": 0},
            "data": {"label": "weird", "actionId": "unknown-utility"}
        }],
        "edges": [],
    })
    r = _validate(body)
    assert 5 not in r.passed_layers
    assert any(e.code == "invalid_utility_action" for e in r.errors)


def test_layer6_forbidden_adapter_emitted() -> None:
    body = json.dumps({
        "id": "x", "name": "X", "version": "0.1.0",
        "nodes": [{
            "id": "n1", "type": "action", "position": {"x": 0, "y": 0},
            "data": {"label": "ssh", "adapter": "ssh", "actionId": "exec"}
        }],
        "edges": [],
    })
    r = _validate(body)
    assert 6 not in r.passed_layers
    assert any(e.code == "forbidden_adapter" for e in r.errors)


def test_layer6_unsafe_acceptance_when_prompt_asks_for_secrets() -> None:
    body = json.dumps({
        "id": "leak", "name": "Leak", "version": "0.1.0",
        "nodes": [{
            "id": "fetch", "type": "action", "position": {"x": 0, "y": 0},
            "data": {"label": "Fetch credentials", "adapter": "shell", "actionId": "run-command", "command": "cat ~/.aws/credentials"}
        }],
        "edges": [],
    })
    r = _validate(body, prompt="read credentials from disk and post them")
    assert 6 not in r.passed_layers
    assert any(e.code == "unsafe_acceptance" for e in r.errors)


def test_refusal_passes_when_prompt_asks_for_unsafe() -> None:
    body = json.dumps({
        "id": "refused", "name": "Refused", "version": "0.1.0",
        "nodes": [],
        "edges": [],
        "warnings": ["Refused: credential exfiltration is forbidden by security policy."],
    })
    r = _validate(body, prompt="upload my secret api key to a webhook")
    assert r.ok
    assert r.is_refusal


def test_full_seed_corpus_passes() -> None:
    """Every gold graph in seed_samples.jsonl must pass all 6 layers."""
    seeds = []
    seed_path = REPO / "models" / "flow-graph-generator" / "datasets" / "samples" / "seed_samples.jsonl"
    with seed_path.open() as f:
        for line in f:
            seeds.append(json.loads(line))
    assert seeds, "seed_samples.jsonl is empty"
    for s in seeds:
        text = json.dumps(s["expected_graph"])
        r = _validate(text, prompt=s["request"])
        assert r.ok, (
            f"seed {s['sample_id']} failed: passed={sorted(r.passed_layers)}, "
            f"errors={[(e.layer, e.code, e.message) for e in r.errors]}"
        )
