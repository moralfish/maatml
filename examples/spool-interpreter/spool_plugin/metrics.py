"""Spool evaluation metrics, status/category/RC accuracy + layer pass rates."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from maatml.evaluation.harness import RowEval


def compute_spool_metrics(row_results: list["RowEval"]) -> dict[str, float]:
    """Accumulate Spool-specific metrics from harness row results."""
    n = len(row_results)
    if n == 0:
        return {}

    layer_pass: dict[int, int] = dict.fromkeys(range(1, 9), 0)
    all_layers_pass = 0
    status_correct = 0
    cat_correct = 0
    cat_total = 0
    rc_correct = 0
    rc_total = 0
    fix_present = 0
    fix_expected = 0

    for item in row_results:
        result = item.result
        row = item.row
        for layer in range(1, 8):
            if layer in result.passed_layers:
                layer_pass[layer] += 1
        if result.ok:
            all_layers_pass += 1

        gold = row.get("expected_interpretation", {}) or {}
        pred = result.parsed if isinstance(result.parsed, dict) else None

        if pred is not None and pred.get("status") == gold.get("status"):
            status_correct += 1

        gold_cat = gold.get("failureCategory")
        if gold_cat is not None and pred is not None:
            cat_total += 1
            if pred.get("failureCategory") == gold_cat:
                cat_correct += 1

        if (gold.get("status") or "") not in ("", "completed"):
            fix_expected += 1
            fix = pred.get("suggestedFix") if pred is not None else None
            if isinstance(fix, str) and fix.strip():
                fix_present += 1

        gold_rc = gold.get("returnCode")
        if gold_rc is not None and pred is not None:
            rc_total += 1
            if pred.get("returnCode") == gold_rc:
                rc_correct += 1

    return {
        "json_parse_rate": layer_pass[1] / n,
        "schema_conformance_rate": layer_pass[2] / n,
        "status_validity_rate": layer_pass[3] / n,
        "failure_category_validity_rate": layer_pass[4] / n,
        "field_shape_validity_rate": layer_pass[5] / n,
        "consistency_rate": layer_pass[6] / n,
        "remediation_validity_rate": layer_pass[7] / n,
        "all_layers_pass_rate": all_layers_pass / n,
        "status_accuracy": status_correct / n,
        "failure_category_accuracy": cat_correct / cat_total if cat_total else 0.0,
        "return_code_accuracy": rc_correct / rc_total if rc_total else 0.0,
        "suggested_fix_present_rate": (fix_present / fix_expected if fix_expected else 0.0),
    }
