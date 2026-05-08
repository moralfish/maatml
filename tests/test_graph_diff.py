"""Pin the structural-equivalence comparator used by the dsl-generator
three-tier eval metric."""

from __future__ import annotations

from flow_ml.evaluation.graph_diff import (
    graph_signature,
    score_pair,
    semantic_tag_set,
    structural_match,
    summarise,
)


def _node(id_: str, kind: str, **data) -> dict:
    return {
        "id": id_,
        "type": kind,
        "position": {"x": 0.0, "y": 0.0},
        "data": data,
    }


def _edge(source: str, outcome: str, target: str) -> dict:
    return {
        "id": f"e-{source}-{outcome}-{target}",
        "source": source,
        "target": target,
        "outcome": outcome,
        "label": None,
        "condition": None,
    }


def _graph(nodes: list[dict], edges: list[dict]) -> dict:
    return {
        "id": "g",
        "name": "g",
        "version": "1",
        "nodes": nodes,
        "edges": edges,
    }


def test_identical_graphs_match_structurally():
    g = _graph(
        [_node("a", "action", adapter="shell", actionId="run-command", command="echo")],
        [],
    )
    assert structural_match(g, g)


def test_node_order_is_irrelevant():
    a = _graph(
        [
            _node("a", "action", adapter="shell", actionId="run-command", command="echo a"),
            _node("b", "action", adapter="shell", actionId="run-command", command="echo b"),
        ],
        [_edge("a", "always", "b")],
    )
    b = _graph(
        [
            _node("b", "action", adapter="shell", actionId="run-command", command="echo b"),
            _node("a", "action", adapter="shell", actionId="run-command", command="echo a"),
        ],
        [_edge("a", "always", "b")],
    )
    assert structural_match(a, b)


def test_label_differences_do_not_break_structural_match():
    # `label` is excluded from the node signature; the auto-layout +
    # canvas often rewrite it without changing topology.
    a = _graph([_node("a", "action", adapter="shell", actionId="git", args="status", label="Old")], [])
    b = _graph([_node("a", "action", adapter="shell", actionId="git", args="status", label="New")], [])
    assert structural_match(a, b)


def test_data_value_difference_breaks_structural_match():
    a = _graph([_node("a", "action", adapter="shell", actionId="git", args="status")], [])
    b = _graph([_node("a", "action", adapter="shell", actionId="git", args="diff")], [])
    assert not structural_match(a, b)


def test_extra_edge_breaks_structural_match():
    a = _graph(
        [_node("a", "action"), _node("b", "action")],
        [_edge("a", "always", "b")],
    )
    b = _graph(
        [_node("a", "action"), _node("b", "action")],
        [_edge("a", "always", "b"), _edge("a", "fail", "b")],
    )
    assert not structural_match(a, b)


def test_outcome_difference_breaks_structural_match():
    a = _graph(
        [_node("a", "action"), _node("b", "action")],
        [_edge("a", "pass", "b")],
    )
    b = _graph(
        [_node("a", "action"), _node("b", "action")],
        [_edge("a", "fail", "b")],
    )
    assert not structural_match(a, b)


def test_semantic_tag_match_ignores_data_values_and_topology():
    # Two graphs that use the same toolset but wire it differently still
    # tag-match: tier-3 is "you picked the right tools".
    a = _graph(
        [
            _node("x", "action", adapter="shell", actionId="cargo", args="test"),
            _node("y", "action", adapter="shell", actionId="kubectl", args="rollout"),
        ],
        [_edge("x", "pass", "y")],
    )
    b = _graph(
        [
            _node("p", "action", adapter="shell", actionId="cargo", args="build"),
            _node("q", "action", adapter="shell", actionId="kubectl", args="status"),
        ],
        [_edge("q", "pass", "p")],
    )
    assert semantic_tag_set(a) == semantic_tag_set(b)
    # ... but they are not structurally equal.
    assert not structural_match(a, b)


def test_score_pair_returns_both_tiers():
    g = _graph([_node("a", "utility", actionId="sleep", durationMs=100)], [])
    s = score_pair(g, g)
    assert s == {"structural_match": True, "semantic_tag_match": True}


def test_summarise_handles_empty_input():
    out = summarise([])
    assert out == {"structural_rate": 0.0, "semantic_tag_rate": 0.0, "n": 0.0}


def test_summarise_aggregates_rates():
    rows = [
        {"structural_match": True, "semantic_tag_match": True},
        {"structural_match": False, "semantic_tag_match": True},
        {"structural_match": False, "semantic_tag_match": False},
    ]
    out = summarise(rows)
    assert out["n"] == 3
    assert abs(out["structural_rate"] - 1 / 3) < 1e-9
    assert abs(out["semantic_tag_rate"] - 2 / 3) < 1e-9


def test_graph_signature_is_hashable():
    # The signature has to be a tuple-of-tuples so it can be used as a
    # dict key during deduplication.
    g = _graph([_node("a", "action", adapter="shell", actionId="git", args="status")], [])
    sig = graph_signature(g)
    bucket: dict[object, int] = {}
    bucket[sig] = 1
    assert bucket[sig] == 1
