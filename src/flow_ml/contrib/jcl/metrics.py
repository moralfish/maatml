"""JCL evaluation metrics — severity/code/line accuracy + layer pass rates."""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from flow_ml.evaluation.harness import RowEval


def compute_jcl_metrics(row_results: list["RowEval"]) -> dict[str, float]:
    """Accumulate JCL-specific metrics from harness row results."""
    n = len(row_results)
    if n == 0:
        return {}

    layer_pass: dict[int, int] = {i: 0 for i in range(1, 7)}
    all_layers_pass = 0
    severity_correct = 0
    severity_total = 0
    code_correct = 0
    code_total = 0
    valid_flag_correct = 0
    line_within_3 = 0
    line_total = 0

    for item in row_results:
        result = item.result
        row = item.row
        for layer in range(1, 7):
            if layer in result.passed_layers:
                layer_pass[layer] += 1
        if result.ok:
            all_layers_pass += 1

        gold = row.get("expected_validation_result", {}) or {}
        pred = result.parsed if isinstance(result.parsed, dict) else None

        if pred is not None and isinstance(pred.get("valid"), bool):
            if pred["valid"] == bool(gold.get("valid")):
                valid_flag_correct += 1

        gold_errors = gold.get("errors") or []
        pred_errors = (pred or {}).get("errors") or []
        if gold_errors and pred_errors:
            ge = gold_errors[0]
            pe = pred_errors[0]
            if isinstance(pe.get("severity"), str):
                severity_total += 1
                if pe["severity"] == ge.get("severity"):
                    severity_correct += 1
            if isinstance(pe.get("code"), str):
                code_total += 1
                if pe["code"] == ge.get("code"):
                    code_correct += 1
            gold_line = ge.get("line")
            pred_line = pe.get("line")
            if isinstance(gold_line, int) and isinstance(pred_line, int):
                line_total += 1
                if abs(pred_line - gold_line) <= 3:
                    line_within_3 += 1

    return {
        "json_parse_rate": layer_pass[1] / n,
        "schema_conformance_rate": layer_pass[2] / n,
        "severity_validity_rate": layer_pass[3] / n,
        "code_validity_rate": layer_pass[4] / n,
        "field_shape_validity_rate": layer_pass[5] / n,
        "consistency_rate": layer_pass[6] / n,
        "all_layers_pass_rate": all_layers_pass / n,
        "severity_accuracy": severity_correct / severity_total if severity_total else 0.0,
        "code_accuracy": code_correct / code_total if code_total else 0.0,
        "valid_flag_accuracy": valid_flag_correct / n,
        "line_within_3_accuracy": line_within_3 / line_total if line_total else 0.0,
    }
