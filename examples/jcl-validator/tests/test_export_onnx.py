"""serving.json contract for the ONNX serving bundle (no torch required)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

EXAMPLE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXAMPLE_ROOT))

from jcl_plugin.export_onnx import build_serving_config  # noqa: E402

from maatml.config import load_model_def  # noqa: E402
from maatml.training.multi_head import parse_heads  # noqa: E402


def _head_specs(md) -> list[dict]:
    return [h.to_dict() for h in parse_heads(dict(md.training or {}))]


def test_serving_config_matches_contract() -> None:
    md = load_model_def(EXAMPLE_ROOT, load_plugins=False)
    serving = build_serving_config(md, _head_specs(md))
    assert serving == {
        "contract": "maatml.serving/1",
        "kind": "multi_head_classifier",
        "max_input_tokens": 2048,
        "text_transform": "jcl_columns",
        "line_pointer_marker": "<COL1>",
        "special_tokens": {
            "pad": "<PAD>",
            "unk": "<UNK>",
            "cls": "<CLS>",
            "sep": "<SEP>",
        },
        "heads": [
            {"name": "validity", "kind": "classification", "labels": ["invalid", "valid"]},
            {
                "name": "error_code",
                "kind": "classification",
                "labels": [
                    "missing_dd",
                    "invalid_job_card",
                    "unresolved_symbolic_parameter",
                    "continuation_error",
                    "invalid_exec_statement",
                    "invalid_dataset_reference_structure",
                    "other",
                    "none",
                ],
            },
            {
                "name": "severity",
                "kind": "classification",
                "labels": ["error", "warning", "info", "none"],
            },
            {"name": "line", "kind": "line_pointer"},
        ],
        "onnx": {
            "file": "model.onnx",
            "inputs": ["input_ids", "attention_mask"],
            "outputs": ["validity", "error_code", "severity", "line"],
        },
    }


def test_serving_config_line_head_has_no_labels() -> None:
    md = load_model_def(EXAMPLE_ROOT, load_plugins=False)
    serving = build_serving_config(md, _head_specs(md))
    line = [h for h in serving["heads"] if h["kind"] == "line_pointer"]
    assert line and "labels" not in line[0]


def test_serving_config_requires_heads() -> None:
    md = load_model_def(EXAMPLE_ROOT, load_plugins=False)
    with pytest.raises(ValueError, match="head specs"):
        build_serving_config(md, [])
