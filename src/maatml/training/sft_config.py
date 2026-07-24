"""CPU-safe SFT config models (no torch / transformers import)."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator

# `training.precision` values the device profiles understand. Anything else
# used to fall through resolve_load_dtype as "fp32", so `precision: bfloat16`
# or a typo trained at a precision nobody asked for.
VALID_PRECISIONS = ("bf16", "fp16", "fp32")


def validate_precision(value: Any) -> str:
    """Return ``value`` if it is a supported precision, else raise."""
    text = str(value)
    if text not in VALID_PRECISIONS:
        raise ValueError(
            f"training.precision must be one of {', '.join(VALID_PRECISIONS)}; "
            f"got {value!r}"
        )
    return text


class LoraSettings(BaseModel):
    enabled: bool = True
    r: int = 16
    alpha: int = 32
    dropout: float = 0.05
    target_modules: list[str] = Field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj"]
    )
    # merged (default) | adapter | both, see train_sft artifact save path.
    save_mode: str = "merged"


class QuantizationSettings(BaseModel):
    """Optional bitsandbytes / QLoRA settings (CUDA-only)."""

    model_config = ConfigDict(extra="forbid")

    load_in_4bit: bool = False
    load_in_8bit: bool = False
    bnb_4bit_compute_dtype: str = "bf16"
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_use_double_quant: bool = True

    def enabled(self) -> bool:
        return bool(self.load_in_4bit or self.load_in_8bit)


class SFTTrainConfig(BaseModel):
    """Generic SFT training config, same shape for all three tasks."""

    model_config = ConfigDict(extra="forbid")

    model_id: str = "Qwen/Qwen3-1.7B"
    max_input_tokens: int = Field(default=4096, gt=0)
    batch_size: int = Field(default=2, gt=0)
    grad_accum: int = Field(default=8, gt=0)
    learning_rate: float = 1e-4
    epochs: float = 4.0
    weight_decay: float = 0.01
    warmup_ratio: float = 0.05
    seed: int = 7331
    precision: str = "bf16"
    grad_checkpointing: bool = False
    eval_steps: int = 9999
    save_steps: int = 200
    logging_steps: int = 20
    max_steps: int = -1
    lora: LoraSettings = Field(default_factory=LoraSettings)
    report_to: Any = None
    group_by_length: bool = False
    quantization: Optional[QuantizationSettings] = None
    attn_implementation: Optional[str] = None
    dataloader_workers: Optional[int] = None
    model_revision: Optional[str] = None

    @field_validator("precision")
    @classmethod
    def _check_precision(cls, v: str) -> str:
        return validate_precision(v)


@dataclass
class SFTTrainResult:
    out_dir: Path
    metrics: dict[str, float]
    train_runtime: float
