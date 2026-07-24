"""Scaffold defaults for the plugin-owned ``vlm_sft`` architecture.

``maatml scaffold DIR --architecture vlm_sft --plugin <this folder>`` produces
a folder that ``maatml validate`` accepts; the corpus itself comes from
``maatml datagen`` because the rows reference rendered images.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from maatml.registry import register_scaffold_hook

# A scaffolded folder must carry the same schema and prompt spec the validator
# and generator were written against, otherwise `maatml datagen` produces rows
# its own validator rejects. Both are read from the example corpus when it is
# present, with an inline copy for a standalone plugin install.
_ASSETS = Path(__file__).resolve().parents[1] / "datasets"

_FALLBACK_SCHEMA: dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "title": "VisionVlmDescription",
    "type": "object",
    "additionalProperties": False,
    "required": ["description"],
    "properties": {
        "description": {"type": "string", "minLength": 8, "maxLength": 280},
    },
}

_FALLBACK_PROMPT_SPEC: dict[str, Any] = {
    "system": (
        "You describe synthetic scene images used for vision training. Reply "
        "with a single JSON object only.\n\nSchema:\n"
        '{\n  "description": "<one short factual sentence>"\n}\n'
    ),
    "user_template": (
        "Describe this synthetic scene in one short factual sentence covering "
        "the background style, any colored shapes, and the stick figure's pose."
    ),
    "response_format": "json",
}


def _asset(filename: str, fallback: dict[str, Any]) -> str:
    path = _ASSETS / filename
    if path.is_file():
        return path.read_text(encoding="utf-8")
    return json.dumps(fallback, indent=2) + "\n"


@register_scaffold_hook("vlm_sft")
def scaffold_vlm_sft(target_dir: Path, *, architecture: str, name: str) -> dict[str, Any]:
    """Contribute the VLM sections, schema, prompt spec, and an empty corpus."""
    del target_dir, architecture, name
    return {
        "model_yml": {
            "base_model": "HuggingFaceTB/SmolVLM-256M-Instruct",
            "dataset": {
                "format": "jsonl_seed",
                "request_field": "image",
                "target_field": "expected_output",
                "group_by": "family",
                "schema": "datasets/schema.json",
                "prompt_spec": "datasets/prompt_spec.json",
                "seed_samples": "datasets/samples/seed_samples.jsonl",
                "split_ratios": [0.7, 0.15, 0.15],
                "generator": "described_scenes",
                "seed": 42,
            },
            "training": {
                "model_id": "HuggingFaceTB/SmolVLM-256M-Instruct",
                "image_size": 320,
                "image_longest_edge": 384,
                "max_input_tokens": 512,
                "batch_size": 1,
                "grad_accum": 4,
                "learning_rate": 1.0e-4,
                "weight_decay": 0.0,
                "epochs": 2,
                "seed": 42,
                "precision": "bf16",
                "logging_steps": 5,
                "max_steps": -1,
                "generation": {"max_new_tokens": 64},
                "lora": {
                    "enabled": True,
                    "r": 8,
                    "alpha": 16,
                    "dropout": 0.05,
                    "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
                },
            },
            "smoke": {
                "batch_size": 1,
                "grad_accum": 1,
                "epochs": 1,
                "max_steps": 4,
                "logging_steps": 1,
                "image_longest_edge": 256,
                "max_input_tokens": 256,
            },
            "evaluation": {
                "predictor": "vision_vlm",
                "validator": "vision_vlm",
                "metrics": "vision_vlm",
                "gates": {
                    "scene_mention_rate": 0.5,
                    "shape_mention_f1": 0.3,
                    "brevity_rate": 0.8,
                },
            },
            "packaging": {
                "max_input_tokens": 512,
                "expected_latency_ms": 2000,
                "weights_dtype": "f16",
            },
        },
        # Rows reference rendered images: `maatml datagen <dir>` writes them.
        "seed_rows": [],
        "files": {
            "datasets/schema.json": _asset("schema.json", _FALLBACK_SCHEMA),
            "datasets/prompt_spec.json": _asset(
                "prompt_spec.json", _FALLBACK_PROMPT_SPEC
            ),
            "GENERATE.md": (
                "# Next steps\n\n"
                "Seed rows reference rendered images, so this corpus starts\n"
                "empty. Fill it with\n\n"
                "```bash\n"
                "maatml datagen . --target 200\n"
                "maatml prepare .\n"
                "maatml train . --smoke\n"
                "```\n"
            ),
        },
    }
