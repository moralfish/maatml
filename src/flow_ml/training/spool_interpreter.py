"""LoRA fine-tune for Spool Interpreter: sanitized z/OS spool → SpoolInterpretation JSON.

Pure SFT on 3-message conversations (system, user, assistant). Replaces
the legacy SmolLM2-360M generative trainer with the Qwen3-1.7B base
shared across all three flow-ml SFT models.

Thin task-specific wrapper over `flow_ml.training.sft_base`.
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


__all__ = [
    "LoraSettings",
    "SFTDataCollator",
    "SpoolTrainConfig",
    "SpoolTrainResult",
    "_ListDataset",
    "_maybe_attach_lora",
    "_resolve_device",
    "build_chat_example",
    "render_assistant_target",
    "render_inference_prompt",
    "train_spool",
]

# Aliases preserved for callers that imported the old class names.
SpoolTrainConfig = SFTTrainConfig
SpoolTrainResult = SFTTrainResult

DEFAULT_PROMPT_SPEC = (
    Path(__file__).resolve().parents[3]
    / "models"
    / "spool-interpreter"
    / "datasets"
    / "prompt_spec.json"
)

TARGET_FIELD = "expected_interpretation"
USER_PLACEHOLDER = "<<USER_REQUEST>>"


def render_inference_prompt(request: str, prompt_spec: dict, tokenizer) -> list[int]:
    return _base_render_inference_prompt(
        request, prompt_spec, tokenizer, user_placeholder=USER_PLACEHOLDER
    )


def train_spool(
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
    """Fine-tune Spool Interpreter (LoRA) from a `ModelDefinition`."""
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
        log_label="Spool",
    )


def train() -> None:
    from rich.console import Console
    Console().print(
        "Use flow_ml.training.spool_interpreter.train_spool(model_def, ...)"
    )
