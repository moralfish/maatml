"""LoRA fine-tune for FlowGraphGenerator: natural-language workflow request
→ Flow Graph JSON proposal.

Pure SFT on 3-message conversations (system, user, assistant). The
assistant turn is a serialised JSON object matching `FlowGraphDto` plus
a `warnings` array. Loss masked over system+user; unmasked over assistant.

This module is a thin task-specific wrapper over `flow_ml.training.sft_base`.
What's different about FlowGraph: the gold output lives at
`sample["expected_graph"]` and the prompt template uses
`<<USER_REQUEST>>` for the user-text placeholder.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from ..config import ModelDefinition
from .sft_base import (
    LoraSettings,
    SFTDataCollator,
    SFTTrainConfig,
    SFTTrainResult,
    _ListDataset,
    _maybe_attach_lora,
    _resolve_device,
    build_chat_example,
    render_assistant_target,
    render_inference_prompt as _base_render_inference_prompt,
    train_sft,
)


# Re-exported for legacy imports (tests, evaluation runner, etc.)
__all__ = [
    "LoraSettings",
    "SFTDataCollator",
    "SFTTrainConfig",
    "SFTTrainResult",
    "_ListDataset",
    "_maybe_attach_lora",
    "_resolve_device",
    "build_chat_example",
    "render_assistant_target",
    "render_inference_prompt",
    "train_flow_graph",
    "FlowGraphTrainConfig",
    "FlowGraphTrainResult",
]

# Aliases preserved for callers that imported the old class names.
FlowGraphTrainConfig = SFTTrainConfig
FlowGraphTrainResult = SFTTrainResult

DEFAULT_PROMPT_SPEC = (
    Path(__file__).resolve().parents[3]
    / "models"
    / "flow-graph-generator"
    / "datasets"
    / "prompt_spec.json"
)

TARGET_FIELD = "expected_graph"
USER_PLACEHOLDER = "<<USER_REQUEST>>"


def render_inference_prompt(request: str, prompt_spec: dict, tokenizer) -> list[int]:
    """Single-shot inference prompt for FlowGraph. Wraps the shared base
    with the FlowGraph-specific user_placeholder."""
    return _base_render_inference_prompt(
        request, prompt_spec, tokenizer, user_placeholder=USER_PLACEHOLDER
    )


def train_flow_graph(
    model_def: ModelDefinition,
    *,
    smoke: bool = False,
    dataset_dir: Optional[Path] = None,
    out_dir: Optional[Path] = None,
    prompt_spec_path: Optional[str | Path] = None,
    limit: Optional[int] = None,
    device: str = "auto",
    seed: Optional[int] = None,
) -> SFTTrainResult:
    """Fine-tune FlowGraphGenerator (LoRA) from a `ModelDefinition`."""
    return train_sft(
        model_def,
        target_field=TARGET_FIELD,
        request_field="request",
        user_placeholder=USER_PLACEHOLDER,
        default_prompt_spec=DEFAULT_PROMPT_SPEC,
        smoke=smoke,
        dataset_dir=dataset_dir,
        out_dir=out_dir,
        prompt_spec_path=prompt_spec_path,
        limit=limit,
        device=device,
        seed=seed,
        log_label="FlowGraph",
    )


def train() -> None:
    from rich.console import Console
    Console().print(
        "Use flow_ml.training.flow_graph_generator.train_flow_graph(model_def, ...)"
    )
