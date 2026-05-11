"""Spool Interpreter v2 — flan-t5-base seq2seq.

Full fine-tune of `google/flan-t5-base` as an encoder-decoder. Input is
the sanitised spool transcript prefixed with the `interpret spool:` task
marker; target is the canonical `SpoolInterpretation` JSON serialised as
text (the runtime parses it back via the 8-layer validator).

No LoRA — flan-t5-base at ~250 M params fine-tunes comfortably on a
16-32 GB Mac and the resulting weights merge straight into the package.

Replaces the v1 generative-SFT thin wrapper at `training/spool_interpreter.py`.

Public surface:
  - `SpoolSeq2SeqConfig` — typed config built from `model.yml::training`
  - `train_spool_seq2seq(model_def, ...)` — entry point invoked by the CLI
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from ..config import ModelDefinition
from ..utils.io import iter_jsonl


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


def _serialise_target(interpretation: dict) -> str:
    """Canonical compact-JSON serialisation of the SpoolInterpretation.

    Stable key order keeps the model's target distribution narrow — every
    sample has the same key sequence regardless of how the seed builder
    emits its dict. The runtime's JSON parser is order-insensitive.
    """
    keys = [
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
    ordered = {k: interpretation.get(k) for k in keys if k in interpretation}
    return json.dumps(ordered, ensure_ascii=False, separators=(",", ":"))


def _build_dataset(rows: list[dict], tokenizer, source_max_len: int, target_max_len: int):
    """Build a seq2seq dataset from the SFT JSONL.

    Input string: `interpret spool: <request>`
    Target string: canonical-serialised `expected_interpretation` JSON.
    """
    import torch
    from torch.utils.data import Dataset

    class _SpoolDataset(Dataset):
        def __init__(self, samples: list[dict]) -> None:
            self.samples = samples

        def __len__(self) -> int:
            return len(self.samples)

        def __getitem__(self, idx: int) -> dict:
            row = self.samples[idx]
            request = row.get("request", "")
            interp = row.get("expected_interpretation") or {}

            source = TASK_PREFIX + request
            target = _serialise_target(interp)

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

            # Pad tokens in `labels` must be masked to -100 so the
            # cross-entropy loss ignores them. T5's pad_token_id is 0.
            pad_id = tokenizer.pad_token_id
            labels = [
                tid if tid != pad_id else -100 for tid in tgt["input_ids"]
            ]

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
) -> SpoolSeq2SeqResult:
    import torch
    from transformers import (
        AutoModelForSeq2SeqLM,
        Seq2SeqTrainer,
        Seq2SeqTrainingArguments,
        DataCollatorForSeq2Seq,
        set_seed,
    )

    set_seed(cfg.seed)

    model = AutoModelForSeq2SeqLM.from_pretrained(cfg.model_id)
    # T5 generation cap: keep the model's generation_config in sync with
    # the manifest so the runtime greedy decode matches what we trained.
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
        predict_with_generate=False,
        generation_max_length=cfg.generation.max_new_tokens,
        generation_num_beams=cfg.generation.num_beams,
        remove_unused_columns=False,
    )

    collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        label_pad_token_id=-100,
        padding=False,  # already padded in __getitem__
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collator,
        processing_class=tokenizer,
    )

    train_output = trainer.train()
    eval_metrics: dict = {}
    if val_ds is not None:
        eval_metrics = trainer.evaluate(eval_dataset=val_ds) or {}

    # Save the full encoder-decoder so the runtime can mmap it directly.
    model.save_pretrained(out_dir, safe_serialization=True)
    tokenizer.save_pretrained(out_dir)

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
    """Train the v2 Spool Interpreter seq2seq model from a `ModelDefinition`."""
    from transformers import AutoTokenizer

    training_dict = model_def.merged_smoke() if smoke else dict(model_def.training)
    cfg = SpoolSeq2SeqConfig.from_dict(training_dict)
    if seed is not None:
        cfg.seed = seed

    dataset_dir = Path(dataset_dir) if dataset_dir else model_def.prepared_dir
    if out_dir is None:
        run_name = "smoke" if smoke else model_def.model_id.replace(":", "-")
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

    print(
        f"Spool seq2seq: model={cfg.model_id} train={len(train_rows)} "
        f"val={len(val_rows)} src_len={cfg.source_max_len} tgt_len={cfg.target_max_len} "
        f"epochs={cfg.epochs}"
    )

    train_ds = _build_dataset(train_rows, tokenizer, cfg.source_max_len, cfg.target_max_len)
    val_ds = (
        _build_dataset(val_rows, tokenizer, cfg.source_max_len, cfg.target_max_len)
        if val_rows
        else None
    )

    return _train_loop(cfg, train_ds, val_ds, out_dir, tokenizer)
