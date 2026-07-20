"""Spool Interpreter — flan-t5-base seq2seq.

Full fine-tune of `google/flan-t5-base` as an encoder-decoder. Input is
the sanitised spool transcript prefixed with the task marker; target is
the canonical `SpoolInterpretation` JSON serialised as text.

Public surface:
  - `SpoolSeq2SeqConfig` — typed config built from `model.yml::training`
  - `train_spool_seq2seq(model_def, ...)` — entry point invoked by the CLI
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from ..config import ModelDefinition, get_dataset_cfg
from ..device import get_profile, resolve_device
from ..utils.io import iter_jsonl
from .guards import ensure_tokenizer_model_contract, make_nan_guard_callback, write_run_metadata


@dataclass
class GenerationCfg:
    num_beams: int = 1
    max_new_tokens: int = 512


@dataclass
class SpoolSeq2SeqConfig:
    model_id: str = "google/flan-t5-base"
    source_max_len: int = 1024
    target_max_len: int = 512
    batch_size: int = 8
    grad_accum: int = 2
    learning_rate: float = 3.0e-5
    epochs: int = 6
    weight_decay: float = 0.01
    warmup_ratio: float = 0.06
    seed: int = 1337
    precision: str = "bf16"
    grad_checkpointing: bool = False
    eval_steps: int = 9999
    save_steps: int = 250
    logging_steps: int = 25
    max_steps: int = -1
    generation: GenerationCfg = field(default_factory=GenerationCfg)

    @classmethod
    def from_dict(cls, d: dict) -> "SpoolSeq2SeqConfig":
        gen = d.get("generation") or {}
        return cls(
            model_id=d.get("model_id", "google/flan-t5-base"),
            source_max_len=int(d.get("source_max_len", 1024)),
            target_max_len=int(d.get("target_max_len", 512)),
            batch_size=int(d.get("batch_size", 8)),
            grad_accum=int(d.get("grad_accum", 2)),
            learning_rate=float(d.get("learning_rate", 3.0e-5)),
            epochs=int(d.get("epochs", 6)),
            weight_decay=float(d.get("weight_decay", 0.01)),
            warmup_ratio=float(d.get("warmup_ratio", 0.06)),
            seed=int(d.get("seed", 1337)),
            precision=d.get("precision", "bf16"),
            grad_checkpointing=bool(d.get("grad_checkpointing", False)),
            eval_steps=int(d.get("eval_steps", 9999)),
            save_steps=int(d.get("save_steps", 250)),
            logging_steps=int(d.get("logging_steps", 25)),
            max_steps=int(d.get("max_steps", -1)),
            generation=GenerationCfg(
                num_beams=int(gen.get("num_beams", 1)),
                max_new_tokens=int(gen.get("max_new_tokens", 512)),
            ),
        )


@dataclass
class SpoolSeq2SeqResult:
    out_dir: Path
    metrics: dict
    train_runtime: float


TASK_PREFIX = "interpret spool: "

_DEFAULT_TARGET_KEYS = [
    "summary",
    "status",
    "returnCode",
    "rootCause",
    "suggestedFix",
    "explanation",
    "relatedDocs",
    "failureCategory",
    "confidence",
]


def _serialise_target(
    interpretation: dict, *, key_order: Optional[list[str]] = None
) -> str:
    """Canonical compact-JSON serialisation of the SpoolInterpretation."""
    keys = key_order or _DEFAULT_TARGET_KEYS
    ordered = {k: interpretation.get(k) for k in keys if k in interpretation}
    return json.dumps(ordered, ensure_ascii=False, separators=(",", ":"))


def _build_dataset(
    rows: list[dict],
    tokenizer,
    source_max_len: int,
    target_max_len: int,
    *,
    source_prefix: str = TASK_PREFIX,
    target_key_order: Optional[list[str]] = None,
    request_field: str = "request",
    target_field: str = "expected_interpretation",
):
    """Build a seq2seq dataset from prepared JSONL."""
    import torch
    from torch.utils.data import Dataset

    class _SpoolDataset(Dataset):
        def __init__(self, samples: list[dict]) -> None:
            self.samples = samples

        def __len__(self) -> int:
            return len(self.samples)

        def __getitem__(self, idx: int) -> dict:
            row = self.samples[idx]
            request = row.get(request_field, "")
            interp = row.get(target_field) or {}

            source = source_prefix + request
            target = _serialise_target(interp, key_order=target_key_order)

            src = tokenizer(
                source,
                max_length=source_max_len,
                padding="max_length",
                truncation=True,
                return_tensors=None,
            )
            tgt = tokenizer(
                target,
                max_length=target_max_len,
                padding="max_length",
                truncation=True,
                return_tensors=None,
            )

            pad_id = tokenizer.pad_token_id
            labels = [tid if tid != pad_id else -100 for tid in tgt["input_ids"]]

            return {
                "input_ids": torch.tensor(src["input_ids"], dtype=torch.long),
                "attention_mask": torch.tensor(src["attention_mask"], dtype=torch.long),
                "labels": torch.tensor(labels, dtype=torch.long),
            }

    return _SpoolDataset(rows)


def _train_loop(
    cfg: SpoolSeq2SeqConfig,
    train_ds,
    val_ds,
    out_dir: Path,
    tokenizer,
    *,
    profile,
    embedding_strategy: Optional[str],
    model_def: ModelDefinition,
    dataset_dir: Path,
    smoke: bool,
    device_name: str,
) -> SpoolSeq2SeqResult:
    from transformers import (
        AutoModelForSeq2SeqLM,
        Seq2SeqTrainer,
        Seq2SeqTrainingArguments,
        DataCollatorForSeq2Seq,
        set_seed,
    )

    set_seed(cfg.seed)

    model = AutoModelForSeq2SeqLM.from_pretrained(cfg.model_id)
    ensure_tokenizer_model_contract(
        model, tokenizer, embedding_strategy=embedding_strategy
    )
    model.generation_config.max_new_tokens = cfg.generation.max_new_tokens
    model.generation_config.num_beams = cfg.generation.num_beams

    total_steps = (
        int(len(train_ds) / cfg.batch_size / cfg.grad_accum * cfg.epochs)
        if cfg.max_steps < 0
        else cfg.max_steps
    )
    warmup_steps = max(0, int(round(total_steps * cfg.warmup_ratio)))

    use_bf16 = cfg.precision == "bf16"
    use_fp16 = cfg.precision == "fp16"
    use_grad_ckpt = bool(cfg.grad_checkpointing) and profile.allow_grad_checkpointing
    run_eval = val_ds is not None and profile.allow_mid_train_eval and cfg.eval_steps < total_steps

    args = Seq2SeqTrainingArguments(
        output_dir=str(out_dir),
        per_device_train_batch_size=cfg.batch_size,
        per_device_eval_batch_size=cfg.batch_size,
        gradient_accumulation_steps=cfg.grad_accum,
        learning_rate=cfg.learning_rate,
        num_train_epochs=cfg.epochs,
        weight_decay=cfg.weight_decay,
        warmup_steps=warmup_steps,
        logging_steps=cfg.logging_steps,
        eval_strategy="steps" if run_eval else "no",
        eval_steps=cfg.eval_steps if run_eval else None,
        save_strategy="steps",
        save_steps=cfg.save_steps,
        save_total_limit=2,
        seed=cfg.seed,
        bf16=use_bf16,
        fp16=use_fp16,
        gradient_checkpointing=use_grad_ckpt,
        dataloader_num_workers=profile.dataloader_workers,
        report_to=[],
        optim="adamw_torch",
        max_steps=cfg.max_steps,
        predict_with_generate=False,
        generation_max_length=cfg.generation.max_new_tokens,
        generation_num_beams=cfg.generation.num_beams,
        remove_unused_columns=False,
    )

    collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        label_pad_token_id=-100,
        padding=False,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds if run_eval else None,
        data_collator=collator,
        processing_class=tokenizer,
        callbacks=[make_nan_guard_callback()],
    )

    train_output = trainer.train()
    eval_metrics: dict = {}
    if val_ds is not None:
        profile.empty_cache()
        eval_metrics = trainer.evaluate(eval_dataset=val_ds) or {}

    model.save_pretrained(out_dir, safe_serialization=True)
    tokenizer.save_pretrained(out_dir)

    write_run_metadata(
        out_dir,
        model_def,
        {
            "train": dataset_dir / "train.jsonl",
            "val": dataset_dir / "val.jsonl",
        },
        extra={"smoke": smoke, "device": device_name, "profile": profile.name},
    )

    return SpoolSeq2SeqResult(
        out_dir=out_dir,
        metrics={k: float(v) for k, v in eval_metrics.items() if isinstance(v, (int, float))},
        train_runtime=float(
            getattr(train_output, "metrics", {}).get("train_runtime", 0.0)
            if hasattr(train_output, "metrics")
            else 0.0
        ),
    )


def train_spool_seq2seq(
    model_def: ModelDefinition,
    *,
    smoke: bool = False,
    dataset_dir: Optional[Path] = None,
    out_dir: Optional[Path] = None,
    limit: Optional[int] = None,
    device: str = "auto",
    seed: Optional[int] = None,
) -> SpoolSeq2SeqResult:
    """Train the Spool Interpreter seq2seq model from a `ModelDefinition`."""
    from transformers import AutoTokenizer

    training_dict = model_def.merged_smoke() if smoke else dict(model_def.training)
    cfg = SpoolSeq2SeqConfig.from_dict(training_dict)
    if seed is not None:
        cfg.seed = seed

    ds_cfg = get_dataset_cfg(model_def)
    source_prefix = ds_cfg.get("source_prefix", TASK_PREFIX)
    target_key_order = ds_cfg.get("target_key_order") or _DEFAULT_TARGET_KEYS
    request_field = ds_cfg.get("request_field") or ds_cfg.get("raw_field") or "request"
    target_field = ds_cfg.get("target_field") or "expected_interpretation"

    dataset_dir = Path(dataset_dir) if dataset_dir else model_def.prepared_dir
    if out_dir is None:
        run_name = "smoke" if smoke else model_def.identity
        out_dir = model_def.checkpoints_dir / run_name
    else:
        out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_id, use_fast=True)

    train_rows = list(iter_jsonl(dataset_dir / "train.jsonl"))
    val_rows = list(iter_jsonl(dataset_dir / "val.jsonl"))
    if limit is not None:
        train_rows = train_rows[:limit]
        val_rows = val_rows[: max(2, limit // 4)]
    if not train_rows:
        raise ValueError(f"No training rows in {dataset_dir / 'train.jsonl'}")

    target_device = resolve_device(device)
    profile = get_profile(target_device)

    print(
        f"Spool seq2seq: model={cfg.model_id} train={len(train_rows)} "
        f"val={len(val_rows)} src_len={cfg.source_max_len} tgt_len={cfg.target_max_len} "
        f"epochs={cfg.epochs}"
    )

    train_ds = _build_dataset(
        train_rows,
        tokenizer,
        cfg.source_max_len,
        cfg.target_max_len,
        source_prefix=source_prefix,
        target_key_order=list(target_key_order),
        request_field=request_field,
        target_field=target_field,
    )
    val_ds = (
        _build_dataset(
            val_rows,
            tokenizer,
            cfg.source_max_len,
            cfg.target_max_len,
            source_prefix=source_prefix,
            target_key_order=list(target_key_order),
            request_field=request_field,
            target_field=target_field,
        )
        if val_rows
        else None
    )

    return _train_loop(
        cfg,
        train_ds,
        val_ds,
        out_dir,
        tokenizer,
        profile=profile,
        embedding_strategy=training_dict.get("embedding_strategy"),
        model_def=model_def,
        dataset_dir=dataset_dir,
        smoke=smoke,
        device_name=str(target_device),
    )
