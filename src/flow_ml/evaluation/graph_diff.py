"""Structural-equivalence comparator for parsed Flow DSL graphs.

Used by the dsl-generator three-tier eval metric. Inputs are dicts in the
shape produced by `flow_dsl::parse` -> serde_json -> Python (whether
parsed via the `flow_dsl_py` PyO3 binding or shelled out through
`flow-dsl-validate --json`).

The three tiers, in increasing strictness:

    semantic_tag_match(a, b)
        Same multiset of (kind, adapter, actionId) triples. Catches
        "right tools, wrong wiring" and lets us measure "the model picked
        the right adapter mix" without penalising every cosmetic
        rearrangement.

    graph_signature(a) == graph_signature(b)
        Strict topology match: same node ids, same per-node (kind +
        sorted data items), same edge multiset. Position is excluded
        because the parser always sets {0, 0}; label is excluded because
        the auto-layout overwrites it on the canvas.

    byte_equal(a, b)
        Trivial; not used as a metric (the serializer reorders body keys
        alphabetically, so byte-equality is too brittle for a learned
        model).

Position and edge ids are intentionally not part of any signature: ids
are auto-generated as `e-{source}-{outcome}-{target}` and would re-derive
under any structural match anyway.

Pure stdlib. No external deps. Stays importable in environments that do
not have torch installed (training-time dep) so the lighter-weight eval
report can still run on CI machines.
"""

from __future__ import annotations

from typing import Any, Iterable

GraphDict = dict[str, Any]
NodeDict = dict[str, Any]
EdgeDict = dict[str, Any]


# --- Tier 3: semantic tags ----------------------------------------------

def semantic_tag_set(graph: GraphDict) -> tuple[tuple[str, str | None, str | None], ...]:
    """Multiset of (kind, adapter, actionId) triples used in `graph`.

    Returned as a sorted tuple of triples so two semantically-equivalent
    graphs hash the same regardless of node order. `adapter` and
    `actionId` are `None` when the kind doesn't carry them (ai,
    cloud_ai, utility may have their own identifiers but at the tier-3
    level we only check the kind plus anything the action node would
    use).
    """
    triples: list[tuple[str, str | None, str | None]] = []
    for node in graph.get("nodes", []):
        kind = str(node.get("type", "")) or ""
        data = node.get("data") or {}
        adapter = data.get("adapter") if isinstance(data, dict) else None
        action_id = data.get("actionId") if isinstance(data, dict) else None
        triples.append((kind, _opt_str(adapter), _opt_str(action_id)))
    triples.sort()
    return tuple(triples)


def semantic_tag_match(a: GraphDict, b: GraphDict) -> bool:
    return semantic_tag_set(a) == semantic_tag_set(b)


# --- Tier 2: structural equivalence -------------------------------------

NodeSig = tuple[str, str, tuple[tuple[str, Any], ...]]
EdgeSig = tuple[str, str, str]


def graph_signature(graph: GraphDict) -> tuple[tuple[NodeSig, ...], tuple[EdgeSig, ...]]:
    """Canonical structural signature: sorted node sigs + sorted edge sigs.

    A `NodeSig` is `(id, kind, sorted-data-items)`. `data.label` is
    excluded because the parser injects it from the header and the
    serializer lifts it back; including it would amount to comparing the
    label twice (once in the kind header sense, once as a data field) and
    a hand-edited label cosmetic change should not count as a structural
    diff.

    An `EdgeSig` is `(source, outcome, target)`. The auto-derived edge
    id is excluded because it is a pure function of the other three.
    """
    nodes = tuple(sorted(_node_sig(n) for n in graph.get("nodes", [])))
    edges = tuple(sorted(_edge_sig(e) for e in graph.get("edges", [])))
    return (nodes, edges)


def structural_match(a: GraphDict, b: GraphDict) -> bool:
    return graph_signature(a) == graph_signature(b)


# --- helpers ------------------------------------------------------------

def _node_sig(node: NodeDict) -> NodeSig:
    nid = str(node.get("id", ""))
    kind = str(node.get("type", ""))
    data = node.get("data") or {}
    if not isinstance(data, dict):
        # Defensive: parser always emits an object, but be liberal.
        return (nid, kind, ())
    items = sorted(
        (str(k), _normalise(v)) for k, v in data.items() if k != "label"
    )
    return (nid, kind, tuple(items))


def _edge_sig(edge: EdgeDict) -> EdgeSig:
    source = str(edge.get("source", ""))
    target = str(edge.get("target", ""))
    outcome = str(edge.get("outcome", "always"))
    return (source, outcome, target)


def _normalise(value: Any) -> Any:
    """Make a value hashable + comparable. The DSL grammar accepts only
    scalars in body fields, so dict/list values would mean either the
    canvas wrote them post-parse or someone authored a non-DSL document.
    Either way we collapse them to a stable string."""
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    try:
        import json
        return ("json", json.dumps(value, sort_keys=True))
    except (TypeError, ValueError):
        return ("repr", repr(value))


def _opt_str(value: Any) -> str | None:
    return str(value) if isinstance(value, str) else None


# --- summary report -----------------------------------------------------

def score_pair(predicted: GraphDict, gold: GraphDict) -> dict[str, bool]:
    """Run all three tiers on a single (predicted, gold) pair. Returns a
    dict suitable for direct merging into the per-row eval report."""
    return {
        "structural_match": structural_match(predicted, gold),
        "semantic_tag_match": semantic_tag_match(predicted, gold),
    }


def summarise(rows: Iterable[dict[str, bool]]) -> dict[str, float]:
    """Aggregate per-row scores into headline rates."""
    rows = list(rows)
    if not rows:
        return {
            "structural_rate": 0.0,
            "semantic_tag_rate": 0.0,
            "n": 0.0,
        }
    n = len(rows)
    structural = sum(1 for r in rows if r.get("structural_match")) / n
    semantic = sum(1 for r in rows if r.get("semantic_tag_match")) / n
    return {
        "structural_rate": structural,
        "semantic_tag_rate": semantic,
        "n": float(n),
    }
