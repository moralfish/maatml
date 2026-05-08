"""LoRA fine-tune for the Agent Planner (request/context -> strict agent JSON)."""
from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

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
from ..utils.io import iter_jsonl, read_json, read_yaml

console = Console()

DEFAULT_PROMPT_SPEC = (
    Path(__file__).resolve().parents[3]
    / "models"
    / "agent-planner"
    / "datasets"
    / "prompt_spec.json"
)


class LoraSettings(BaseModel):
    enabled: bool = True
    r: int = 16
    alpha: int = 32
    dropout: float = 0.05
    target_modules: list[str] = Field(default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj"])


class AgentTrainConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model_id: str = "Qwen/Qwen3-4B-Instruct-2507"
    max_input_tokens: int = Field(default=2048, gt=0)
    batch_size: int = Field(default=1, gt=0)
    grad_accum: int = Field(default=8, gt=0)
    learning_rate: float = 2e-5
    epochs: float = 5.0
    weight_decay: float = 0.01
    warmup_ratio: float = 0.05
    seed: int = 7331
    precision: str = "fp32"
    grad_checkpointing: bool = True
    eval_steps: int = 9999
    save_steps: int = 100
    logging_steps: int = 10
    max_steps: int = -1
    lora: LoraSettings = Field(default_factory=LoraSettings)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "AgentTrainConfig":
        return cls(**read_yaml(path))


@dataclass
class AgentTrainResult:
    out_dir: Path
    metrics: dict[str, float]
    train_runtime: float


class _ListDataset(Dataset):
    def __init__(self, rows: list[dict]) -> None:
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict:
        return self.rows[idx]


def render_target(sample: dict) -> str:
    """Render the supervised target as canonical strict JSON."""
    return json.dumps(sample["agent_plan"], ensure_ascii=False, sort_keys=True)


def render_agent_input(sample: dict) -> str:
    context = sample.get("context") or "No existing flow context."
    return f"User request:\n{sample['request']}\n\nCurrent Flow context:\n{context}"


def build_chat_example(
    sample: dict,
    prompt_spec: dict,
    tokenizer: PreTrainedTokenizerBase,
    *,
    max_length: int,
) -> dict[str, list[int]]:
    """Tokenize one agent-planning sample, masking loss over the prompt."""
    agent_input = render_agent_input(sample)
    user_text = prompt_spec["user_template"].replace("<<AGENT_INPUT>>", agent_input)
    # Back-compat for early two-placeholder prompt specs.
    user_text = (
        user_text
        .replace("<<USER_REQUEST>>", sample["request"])
        .replace("<<FLOW_CONTEXT>>", sample.get("context") or "No existing flow context.")
    )
    target = render_target(sample)

    messages_prompt = [
        {"role": "system", "content": prompt_spec["system"]},
        {"role": "user", "content": user_text},
    ]
    prompt_ids = tokenizer.apply_chat_template(
        messages_prompt,
        add_generation_prompt=True,
        tokenize=True,
        return_tensors=None,
    )
    if isinstance(prompt_ids, dict) or hasattr(prompt_ids, "input_ids"):
        prompt_ids = list(prompt_ids["input_ids"])
    else:
        prompt_ids = list(prompt_ids)
    target_ids = list(tokenizer(target, add_special_tokens=False)["input_ids"])
    eos_id = tokenizer.eos_token_id
    if eos_id is not None and (not target_ids or target_ids[-1] != eos_id):
        target_ids = target_ids + [eos_id]

    input_ids = prompt_ids + target_ids
    labels = [-100] * len(prompt_ids) + list(target_ids)
    if len(input_ids) > max_length:
        input_ids = input_ids[-max_length:]
        labels = labels[-max_length:]
    return {"input_ids": input_ids, "labels": labels}


class AgentDataCollator:
    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        prompt_spec: dict,
        *,
        max_length: int,
    ) -> None:
        self.tokenizer = tokenizer
        self.prompt_spec = prompt_spec
        self.max_length = max_length
        self.pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

    def __call__(self, batch: list[dict]) -> dict[str, torch.Tensor]:
        examples = [
            build_chat_example(row, self.prompt_spec, self.tokenizer, max_length=self.max_length)
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


def train_agent(
    model_def: ModelDefinition,
    *,
    smoke: bool = False,
    dataset_dir: Optional[Path] = None,
    out_dir: Optional[Path] = None,
    prompt_spec_path: Optional[str | Path] = None,
    limit: Optional[int] = None,
    device: str = "auto",
    seed: Optional[int] = None,
) -> AgentTrainResult:
    """Fine-tune the Agent Planner (LoRA) from a ``ModelDefinition``."""
    training_dict = model_def.merged_smoke() if smoke else dict(model_def.training)
    cfg = AgentTrainConfig(**training_dict)
    if seed is not None:
        cfg.seed = seed
    set_seed(cfg.seed)

    if prompt_spec_path is not None:
        spec_path = Path(prompt_spec_path)
    elif "prompt_spec" in model_def.data:
        spec_path = model_def.resolve(model_def.data["prompt_spec"])
    else:
        spec_path = DEFAULT_PROMPT_SPEC
    prompt_spec = read_json(spec_path)

    dataset_dir = Path(dataset_dir) if dataset_dir else model_def.prepared_dir
    if out_dir is None:
        run_name = "smoke" if smoke else cfg.model_id.split("/")[-1].lower()
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
        f"[cyan]Agent train[/]: model={cfg.model_id} train={len(train_rows)} "
        f"val={len(val_rows)} lora={cfg.lora.enabled}"
    )

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_id)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(cfg.model_id)
    model = _maybe_attach_lora(model, cfg.lora)

    target_device = _resolve_device(device)
    use_bf16 = cfg.precision == "bf16" and target_device.type == "cuda"

    collator = AgentDataCollator(tokenizer, prompt_spec, max_length=cfg.max_input_tokens)
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

    args = TrainingArguments(
        output_dir=str(out_dir),
        per_device_train_batch_size=cfg.batch_size,
        per_device_eval_batch_size=cfg.batch_size,
        gradient_accumulation_steps=cfg.grad_accum,
        learning_rate=cfg.learning_rate,
        num_train_epochs=cfg.epochs,
        weight_decay=cfg.weight_decay,
        warmup_ratio=cfg.warmup_ratio,
        logging_steps=cfg.logging_steps,
        eval_strategy="steps" if run_eval_during_training else "no",
        eval_steps=cfg.eval_steps if run_eval_during_training else None,
        save_strategy="steps",
        save_steps=cfg.save_steps,
        save_total_limit=2,
        seed=cfg.seed,
        bf16=use_bf16,
        fp16=False,
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

    eval_metrics: dict = {}
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

    return AgentTrainResult(
        out_dir=out_dir,
        metrics={k: float(v) for k, v in eval_metrics.items() if isinstance(v, (int, float))},
        train_runtime=float(
            getattr(train_output, "metrics", {}).get("train_runtime", 0.0)
            if hasattr(train_output, "metrics")
            else 0.0
        ),
    )


def train() -> None:
    console.print("Use flow_ml.training.agent_planner.train_agent(model_def)")
