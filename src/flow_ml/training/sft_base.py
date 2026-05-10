"""Shared SFT skeleton for flow-ml's three generative models.

All three trainers (`flow_graph_generator`, `jcl_validator`, `spool_interpreter`)
follow the same pattern:

  - Qwen3-1.7B (or 0.6B for smoke) base + LoRA on attention projections
  - 3-message conversations: system + user + assistant
  - Assistant content is a serialised JSON object (per-task schema)
  - Loss masked over system+user; unmasked over assistant + closing `<|im_end|>`
  - bf16 autocast on MPS/CUDA (weights stay fp32 — autocast does the work)
  - Merged safetensors + tokenizer + prompt_spec saved to `output/checkpoints/`

What varies per task: the sample-shape adapter — i.e. which field on the
sample dict carries the gold JSON. Each task module passes
`target_field` and `request_field` to `train_sft` and gets the same
TrainingArguments outer loop.

Public surface:

  - `LoraSettings`, `SFTTrainConfig`, `SFTTrainResult`
  - `_resolve_device`, `_maybe_attach_lora`
  - `render_assistant_target(sample, target_field) -> str`
  - `render_inference_prompt(request, prompt_spec, tokenizer, *, user_placeholder) -> list[int]`
  - `build_chat_example(sample, prompt_spec, tokenizer, *, max_length, target_field, request_field, user_placeholder)`
  - `SFTDataCollator(tokenizer, prompt_spec, *, max_length, target_field, request_field, user_placeholder)`
  - `train_sft(model_def, *, config_cls, target_field, request_field, user_placeholder, default_prompt_spec, ...)`
"""
from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Type

import torch
from peft import LoraConfig, TaskType, get_peft_model
from pydantic import BaseModel, ConfigDict, Field
from rich.console import Console
from torch.utils.data import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedTokenizerBase,
    Trainer,
    TrainingArguments,
    set_seed,
)

from ..config import ModelDefinition
from ..utils.io import iter_jsonl, read_json

console = Console()


# ---------------------------------------------------------------------------
# Config + result dataclasses
# ---------------------------------------------------------------------------


class LoraSettings(BaseModel):
    enabled: bool = True
    r: int = 16
    alpha: int = 32
    dropout: float = 0.05
    target_modules: list[str] = Field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj"]
    )


class SFTTrainConfig(BaseModel):
    """Generic SFT training config — same shape for all three tasks."""

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


@dataclass
class SFTTrainResult:
    out_dir: Path
    metrics: dict[str, float]
    train_runtime: float


# ---------------------------------------------------------------------------
# Tokenization helpers (robust against transformers 5.x apply_chat_template
# return-shape quirks — always go through render-to-text → tokenize)
# ---------------------------------------------------------------------------


class _ListDataset(Dataset):
    def __init__(self, rows: list[dict]) -> None:
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict:
        return self.rows[idx]


def _flatten_token_list(ids) -> list[int]:
    if hasattr(ids, "keys") and "input_ids" in ids:
        ids = ids["input_ids"]
    if hasattr(ids, "tolist"):
        ids = ids.tolist()
    if isinstance(ids, str):
        raise TypeError("expected token ids, got string")
    if ids and isinstance(ids[0], list):
        ids = ids[0]
    return list(ids)


def _render_then_tokenize(
    rendered: list[dict],
    tokenizer: PreTrainedTokenizerBase,
    *,
    add_generation_prompt: bool,
) -> list[int]:
    """Render the chat template to text, then tokenize. Robust against
    `tokenize=True` returning unexpected shapes in different transformers
    versions: we always go through string → tokens.

    `enable_thinking=False` is forwarded to Qwen3 chat templates so the
    template embeds an empty `<think></think>` block in the prompt prefix
    instead of asking the model to emit one. With thinking pre-completed
    the model produces JSON directly. Tokenizers whose templates ignore
    the kwarg silently drop it (no behaviour change for non-Qwen3 bases).
    """
    if not rendered:
        return []
    try:
        text = tokenizer.apply_chat_template(
            rendered,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
            enable_thinking=False,
        )
    except TypeError:
        # Older transformers / non-Qwen3 templates that don't accept the kwarg.
        text = tokenizer.apply_chat_template(
            rendered, tokenize=False, add_generation_prompt=add_generation_prompt
        )
    if not text:
        return []
    encoded = tokenizer(text, add_special_tokens=False, return_tensors=None)
    return _flatten_token_list(encoded)


def render_assistant_target(sample: dict, target_field: str) -> str:
    """Render the gold assistant content: the per-task expected JSON,
    serialised compactly. Compact (no indent) keeps token count down; the
    runtime parses either pretty or compact equally well.
    """
    payload = sample[target_field]
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def build_chat_example(
    sample: dict,
    prompt_spec: dict,
    tokenizer: PreTrainedTokenizerBase,
    *,
    max_length: int,
    target_field: str,
    request_field: str = "request",
    user_placeholder: str = "<<USER_REQUEST>>",
) -> dict[str, list[int]]:
    """Tokenize a single sample into input_ids + labels.

    Loss is masked over system + user turns; unmasked over the assistant
    turn only (content + closing `<|im_end|>`). Spans are located via
    prefix-stable rendering: tokenize `[system, user]` with
    `add_generation_prompt=True` to get the prefix length up to and
    including `<|im_start|>assistant\\n`, then tokenize the full
    `[system, user, assistant]` to get the final length.
    """
    user_text = prompt_spec["user_template"].replace(user_placeholder, sample[request_field])
    target_text = render_assistant_target(sample, target_field)

    rendered = [
        {"role": "system", "content": prompt_spec["system"]},
        {"role": "user", "content": user_text},
        {"role": "assistant", "content": target_text},
    ]

    full_ids = _render_then_tokenize(rendered, tokenizer, add_generation_prompt=False)
    prompt_ids = _render_then_tokenize(rendered[:2], tokenizer, add_generation_prompt=True)

    labels: list[int] = [-100] * len(full_ids)
    start = len(prompt_ids)
    end = len(full_ids)
    for i in range(start, end):
        labels[i] = full_ids[i]

    if len(full_ids) > max_length:
        full_ids = full_ids[-max_length:]
        labels = labels[-max_length:]
    return {"input_ids": full_ids, "labels": labels}


def render_inference_prompt(
    request: str,
    prompt_spec: dict,
    tokenizer: PreTrainedTokenizerBase,
    *,
    user_placeholder: str = "<<USER_REQUEST>>",
) -> list[int]:
    """Inference prompt: `[system, user]` + generation prompt. Mirrors
    `build_chat_example`'s prefix exactly so the model sees the same tail
    distribution at eval as at training."""
    user_text = prompt_spec["user_template"].replace(user_placeholder, request)
    rendered = [
        {"role": "system", "content": prompt_spec["system"]},
        {"role": "user", "content": user_text},
    ]
    return _render_then_tokenize(rendered, tokenizer, add_generation_prompt=True)


class SFTDataCollator:
    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        prompt_spec: dict,
        *,
        max_length: int,
        target_field: str,
        request_field: str = "request",
        user_placeholder: str = "<<USER_REQUEST>>",
    ) -> None:
        self.tokenizer = tokenizer
        self.prompt_spec = prompt_spec
        self.max_length = max_length
        self.target_field = target_field
        self.request_field = request_field
        self.user_placeholder = user_placeholder
        self.pad_id = (
            tokenizer.pad_token_id
            if tokenizer.pad_token_id is not None
            else tokenizer.eos_token_id
        )

    def __call__(self, batch: list[dict]) -> dict[str, torch.Tensor]:
        examples = [
            build_chat_example(
                row,
                self.prompt_spec,
                self.tokenizer,
                max_length=self.max_length,
                target_field=self.target_field,
                request_field=self.request_field,
                user_placeholder=self.user_placeholder,
            )
            for row in batch
        ]
        max_len = max(len(ex["input_ids"]) for ex in examples)
        input_ids = torch.full((len(examples), max_len), self.pad_id, dtype=torch.long)
        attention_mask = torch.zeros((len(examples), max_len), dtype=torch.long)
        labels = torch.full((len(examples), max_len), -100, dtype=torch.long)
        for i, ex in enumerate(examples):
            n = len(ex["input_ids"])
            input_ids[i, :n] = torch.tensor(ex["input_ids"], dtype=torch.long)
            attention_mask[i, :n] = 1
            labels[i, :n] = torch.tensor(ex["labels"], dtype=torch.long)
        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


# ---------------------------------------------------------------------------
# Device + LoRA
# ---------------------------------------------------------------------------


def _resolve_device(device: str) -> torch.device:
    if device == "cpu":
        return torch.device("cpu")
    if device == "mps":
        return torch.device("mps")
    if device == "auto":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    return torch.device(device)


def _maybe_attach_lora(model, lora: LoraSettings):
    if not lora.enabled:
        return model
    config = LoraConfig(
        r=lora.r,
        lora_alpha=lora.alpha,
        lora_dropout=lora.dropout,
        target_modules=lora.target_modules,
        task_type=TaskType.CAUSAL_LM,
    )
    return get_peft_model(model, config)


# ---------------------------------------------------------------------------
# Outer training loop
# ---------------------------------------------------------------------------


def train_sft(
    model_def: ModelDefinition,
    *,
    target_field: str,
    config_cls: Type[SFTTrainConfig] = SFTTrainConfig,
    request_field: str = "request",
    user_placeholder: str = "<<USER_REQUEST>>",
    default_prompt_spec: Optional[Path] = None,
    smoke: bool = False,
    dataset_dir: Optional[Path] = None,
    out_dir: Optional[Path] = None,
    prompt_spec_path: Optional[str | Path] = None,
    limit: Optional[int] = None,
    device: str = "auto",
    seed: Optional[int] = None,
    log_label: str = "SFT",
) -> SFTTrainResult:
    """The shared SFT training driver. Each task module is a thin wrapper
    that pins `target_field`, `request_field`, and `default_prompt_spec`.

    bf16 autocast is enabled on MPS/CUDA when `precision: bf16`. Weights
    are loaded at fp32 to give a stable master copy — loading at bf16 +
    autocast bf16 has produced NaN gradients on MPS in past runs.
    """
    training_dict = model_def.merged_smoke() if smoke else dict(model_def.training)
    cfg = config_cls(**training_dict)
    if seed is not None:
        cfg.seed = seed
    set_seed(cfg.seed)

    if prompt_spec_path is not None:
        spec_path = Path(prompt_spec_path)
    elif "prompt_spec" in model_def.data:
        spec_path = model_def.resolve(model_def.data["prompt_spec"])
    elif default_prompt_spec is not None:
        spec_path = default_prompt_spec
    else:
        raise ValueError("no prompt_spec_path provided and model_def has no `data.prompt_spec`")
    prompt_spec = read_json(spec_path)

    dataset_dir = Path(dataset_dir) if dataset_dir else model_def.prepared_dir
    if out_dir is None:
        run_name = "smoke" if smoke else model_def.model_id.replace(":", "-")
        out_dir = model_def.checkpoints_dir / run_name
    else:
        out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_rows = list(iter_jsonl(dataset_dir / "train.jsonl"))
    val_rows = list(iter_jsonl(dataset_dir / "val.jsonl"))
    if limit is not None:
        train_rows = train_rows[:limit]
        val_rows = val_rows[: max(2, limit // 4)]
    if not train_rows:
        raise ValueError(f"No training rows in {dataset_dir / 'train.jsonl'}")

    console.print(
        f"[cyan]{log_label} train[/]: model={cfg.model_id} train={len(train_rows)} "
        f"val={len(val_rows)} lora={cfg.lora.enabled} precision={cfg.precision}"
    )

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_id)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    target_device = _resolve_device(device)
    use_bf16 = cfg.precision == "bf16" and target_device.type in ("cuda", "mps")
    use_fp16 = cfg.precision == "fp16" and target_device.type in ("cuda", "mps")
    # Weights stay at fp32; autocast handles the bf16/fp16 forward pass.
    model = AutoModelForCausalLM.from_pretrained(cfg.model_id)
    # Pre-align the model's pad/bos/eos with the tokenizer's so the
    # Trainer doesn't emit a "Updated tokens: ..." warning on every run.
    # Qwen3 tokenizers have no BOS — assign None unconditionally so the
    # model config matches.
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.bos_token_id = tokenizer.bos_token_id
    model.config.eos_token_id = tokenizer.eos_token_id
    if hasattr(model, "generation_config") and model.generation_config is not None:
        model.generation_config.pad_token_id = tokenizer.pad_token_id
        model.generation_config.bos_token_id = tokenizer.bos_token_id
        model.generation_config.eos_token_id = tokenizer.eos_token_id
    model = _maybe_attach_lora(model, cfg.lora)

    collator = SFTDataCollator(
        tokenizer,
        prompt_spec,
        max_length=cfg.max_input_tokens,
        target_field=target_field,
        request_field=request_field,
        user_placeholder=user_placeholder,
    )
    train_ds = _ListDataset(train_rows)
    val_ds = _ListDataset(val_rows) if val_rows else None

    total_steps = (
        int(len(train_rows) / cfg.batch_size / cfg.grad_accum * cfg.epochs)
        if cfg.max_steps < 0
        else cfg.max_steps
    )
    run_eval_during_training = (
        val_ds is not None
        and target_device.type != "mps"
        and cfg.eval_steps < total_steps
    )

    # transformers >=5.2 deprecates warmup_ratio in favour of warmup_steps;
    # convert eagerly so the run is forward-compatible.
    warmup_steps = max(0, int(round(total_steps * cfg.warmup_ratio)))

    args = TrainingArguments(
        output_dir=str(out_dir),
        per_device_train_batch_size=cfg.batch_size,
        per_device_eval_batch_size=cfg.batch_size,
        gradient_accumulation_steps=cfg.grad_accum,
        learning_rate=cfg.learning_rate,
        num_train_epochs=cfg.epochs,
        weight_decay=cfg.weight_decay,
        warmup_steps=warmup_steps,
        logging_steps=cfg.logging_steps,
        eval_strategy="steps" if run_eval_during_training else "no",
        eval_steps=cfg.eval_steps if run_eval_during_training else None,
        save_strategy="steps",
        save_steps=cfg.save_steps,
        save_total_limit=2,
        seed=cfg.seed,
        bf16=use_bf16,
        fp16=use_fp16,
        gradient_checkpointing=cfg.grad_checkpointing,
        dataloader_num_workers=0,
        report_to=[],
        optim="adamw_torch",
        max_steps=cfg.max_steps,
        use_cpu=target_device.type == "cpu",
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds if run_eval_during_training else None,
        data_collator=collator,
        processing_class=tokenizer,
    )

    train_output = trainer.train()

    eval_metrics: dict[str, Any] = {}
    if val_ds is not None:
        if target_device.type == "mps":
            torch.mps.empty_cache()
        eval_metrics = trainer.evaluate(eval_dataset=val_ds) or {}

    if hasattr(model, "merge_and_unload"):
        merged = model.merge_and_unload()
        merged.save_pretrained(out_dir)
    else:
        model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
    shutil.copy2(spec_path, out_dir / "prompt_spec.json")

    return SFTTrainResult(
        out_dir=out_dir,
        metrics={k: float(v) for k, v in eval_metrics.items() if isinstance(v, (int, float))},
        train_runtime=float(
            getattr(train_output, "metrics", {}).get("train_runtime", 0.0)
            if hasattr(train_output, "metrics")
            else 0.0
        ),
    )
