"""Evaluation metrics for support-ticket triage.

Reports validator-layer pass rates (the contract) plus field accuracy against
the gold ``expected_output`` (the quality). Parse failures count against
accuracy: an unparseable prediction is a wrong answer, not a skipped one.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from maatml.evaluation.harness import RowEval

_STRUCT_FIELDS = ("priority", "category", "team")


def _pred(item: "RowEval") -> Optional[dict]:
    parsed = item.result.parsed if item.result else None
    return parsed if isinstance(parsed, dict) else None


def _gold(row: dict) -> Optional[dict]:
    gold = row.get("expected_output") or row.get("expected")
    return gold if isinstance(gold, dict) else None


def compute_triage_metrics(row_results: list["RowEval"]) -> dict[str, float]:
    n = len(row_results)
    if n == 0:
        return {}

    layer_pass = {i: 0 for i in range(1, 5)}
    all_ok = 0
    field_ok = {f: 0 for f in _STRUCT_FIELDS}
    field_total = {f: 0 for f in _STRUCT_FIELDS}
    exact = 0
    exact_total = 0

    for item in row_results:
        for layer in range(1, 5):
            if layer in item.result.passed_layers:
                layer_pass[layer] += 1
        if item.result.ok:
            all_ok += 1

        gold = _gold(item.row)
        pred = _pred(item)
        if gold is None:
            continue

        exact_total += 1
        all_match = True
        for f in _STRUCT_FIELDS:
            if f not in gold:
                continue
            field_total[f] += 1
            if pred is not None and pred.get(f) == gold.get(f):
                field_ok[f] += 1
            else:
                all_match = False
        if all_match:
            exact += 1

    metrics = {
        "json_parse_rate": layer_pass[1] / n,
        "schema_conformance_rate": layer_pass[2] / n,
        "routing_consistency_rate": layer_pass[3] / n,
        "summary_quality_rate": layer_pass[4] / n,
        "all_layers_pass_rate": all_ok / n,
    }
    for f in _STRUCT_FIELDS:
        metrics[f"{f}_accuracy"] = (
            field_ok[f] / field_total[f] if field_total[f] else 0.0
        )
    metrics["exact_match_rate"] = exact / exact_total if exact_total else 0.0
    return metrics
