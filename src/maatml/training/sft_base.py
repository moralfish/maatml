"""Shared SFT skeleton for causal LM fine-tuning.

The causal-SFT trainers follow this pattern:

  - Qwen3-1.7B (or 0.6B for smoke) base + LoRA on attention projections
  - 3-message conversations: system + user + assistant
  - Assistant content is a serialised JSON object (per-task schema)
  - Loss masked over system+user; unmasked over assistant + closing `<|im_end|>`
  - bf16 autocast on MPS/CUDA (weights stay fp32, autocast does the work)
  - Merged safetensors + tokenizer + prompt_spec saved to `output/checkpoints/`

What varies per task: the sample-shape adapter, i.e. which field on the
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
from pathlib import Path
from typing import Any, Optional, Type

import torch
from peft import LoraConfig, TaskType, get_peft_model
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

from ..config import ModelDefinition, get_dataset_cfg
from ..device import (
    effective_dataloader_workers,
    resolve_training_placement,
)
from ..runs import begin_training_run, finish_run, normalize_report_to
from ..utils.io import iter_jsonl, read_json, sha256_file, stable_hash
from .guards import ensure_tokenizer_model_contract, make_nan_guard_callback, write_run_metadata
from .load import from_pretrained_kwargs, maybe_prepare_kbit
from .sft_config import (  # noqa: F401  re-export public config surface
    LoraSettings,
    QuantizationSettings,
    SFTTrainConfig,
    SFTTrainResult,
)

console = Console()


# ---------------------------------------------------------------------------
# Tokenization helpers (robust against transformers 5.x apply_chat_template
# return-shape quirks, always go through render-to-text → tokenize)
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


def _messages_from_sample(
    sample: dict,
    prompt_spec: dict,
    *,
    target_field: str,
    request_field: str,
    user_placeholder: str,
) -> list[dict[str, str]]:
    """Build chat messages from either multi-turn ``messages`` or prompt_spec."""
    raw = sample.get("messages")
    if isinstance(raw, list) and raw:
        out: list[dict[str, str]] = []
        for msg in raw:
            if not isinstance(msg, dict):
                continue
            role = str(msg.get("role") or msg.get("from") or "")
            content = msg.get("content", msg.get("value", ""))
            if role in ("human", "user"):
                role = "user"
            elif role in ("gpt", "assistant"):
                role = "assistant"
            elif role == "system":
                role = "system"
            else:
                continue
            out.append({"role": role, "content": str(content)})
        if out:
            return out

    user_text = prompt_spec["user_template"].replace(
        user_placeholder, sample[request_field]
    )
    target_text = render_assistant_target(sample, target_field)
    return [
        {"role": "system", "content": prompt_spec["system"]},
        {"role": "user", "content": user_text},
        {"role": "assistant", "content": target_text},
    ]


def build_chat_example(
    sample: dict,
    prompt_spec: dict,
    tokenizer: PreTrainedTokenizerBase,
    *,
    max_length: int,
    target_field: str,
    request_field: str = "request",
    user_placeholder: str = "<<USER_REQUEST>>",
) -> dict[str, Any]:
    """Tokenize a sample into input_ids + labels.

    Loss is masked over system/user turns and unmasked over each assistant
    turn (content + closing special tokens). Multi-turn ``messages`` rows
    (alpaca / sharegpt) unmask every assistant span; the classic single-turn
    prompt_spec path is unchanged.
    """
    rendered = _messages_from_sample(
        sample,
        prompt_spec,
        target_field=target_field,
        request_field=request_field,
        user_placeholder=user_placeholder,
    )

    full_ids = _render_then_tokenize(rendered, tokenizer, add_generation_prompt=False)
    labels: list[int] = [-100] * len(full_ids)

    for i, msg in enumerate(rendered):
        if msg.get("role") != "assistant":
            continue
        prefix_ids = _render_then_tokenize(
            rendered[:i], tokenizer, add_generation_prompt=True
        )
        through_ids = _render_then_tokenize(
            rendered[: i + 1], tokenizer, add_generation_prompt=False
        )
        start = len(prefix_ids)
        end = min(len(through_ids), len(full_ids))
        for j in range(start, end):
            labels[j] = full_ids[j]

    if len(full_ids) > max_length:
        full_ids = full_ids[-max_length:]
        labels = labels[-max_length:]
    return {"input_ids": full_ids, "labels": labels, "length": len(full_ids)}


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
        pretokenized: bool = False,
    ) -> None:
        self.tokenizer = tokenizer
        self.prompt_spec = prompt_spec
        self.max_length = max_length
        self.target_field = target_field
        self.request_field = request_field
        self.user_placeholder = user_placeholder
        self.pretokenized = pretokenized
        self.pad_id = (
            tokenizer.pad_token_id
            if tokenizer.pad_token_id is not None
            else tokenizer.eos_token_id
        )

    def __call__(self, batch: list[dict]) -> dict[str, torch.Tensor]:
        if self.pretokenized:
            examples = batch
        else:
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


def _tokenizer_cache_identity(tokenizer: PreTrainedTokenizerBase) -> str:
    name = getattr(tokenizer, "name_or_path", None) or ""
    try:
        size = len(tokenizer)
    except TypeError:
        size = getattr(tokenizer, "vocab_size", 0)
    return f"{name}:{size}"


def _load_or_build_tokenized_cache(
    rows: list[dict],
    cache_path: Path,
    build_fn,
) -> list[dict]:
    """Load tokenized rows from ``cache_path`` or build + torch.save them.

    The cache payload is plain lists/dicts of ints, so it loads with
    ``weights_only=True``. We never fall back to a full pickle load: the cache
    lives under a world-writable output dir, and an unpickle there would be a
    code-execution sink.
    """
    if cache_path.is_file():
        try:
            cached = torch.load(cache_path, map_location="cpu", weights_only=True)
            if isinstance(cached, list) and len(cached) == len(rows):
                return cached
        except Exception:  # noqa: BLE001
            pass
    tokenized = [build_fn(row) for row in rows]
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(tokenized, cache_path)
    return tokenized


def _save_sft_artifacts(
    model,
    tokenizer,
    out_dir: Path,
    *,
    save_mode: str,
    base_model_id: str,
    spec_path: Optional[Path] = None,
) -> dict[str, Any]:
    """Persist merged and/or adapter weights per ``lora.save_mode``.

    ``spec_path`` is copied in as ``prompt_spec.json`` when the architecture
    has one; the preference trainers reuse this saver and have none.
    """
    mode = (save_mode or "merged").lower()
    if mode not in ("merged", "adapter", "both"):
        raise ValueError(
            f"training.lora.save_mode must be merged|adapter|both; got {save_mode!r}"
        )
    meta: dict[str, Any] = {"lora_save_mode": mode, "base_model_id": base_model_id}
    is_peft = hasattr(model, "merge_and_unload") and hasattr(model, "save_pretrained")

    if mode in ("adapter", "both") and is_peft:
        adapter_dir = out_dir / "adapter" if mode == "both" else out_dir
        adapter_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(adapter_dir)
        tokenizer.save_pretrained(adapter_dir if mode == "adapter" else out_dir)
        meta["adapter_dir"] = str(adapter_dir)

    if mode in ("merged", "both"):
        if is_peft:
            # For "both", clone path: merge_and_unload mutates; save adapter first.
            if mode == "both":
                merged = model.merge_and_unload()
            else:
                merged = model.merge_and_unload()
            merged.save_pretrained(out_dir)
        else:
            model.save_pretrained(out_dir)
        tokenizer.save_pretrained(out_dir)

    if mode == "adapter" and not is_peft:
        model.save_pretrained(out_dir)
        tokenizer.save_pretrained(out_dir)

    if spec_path is not None:
        shutil.copy2(spec_path, out_dir / "prompt_spec.json")
    return meta


# ---------------------------------------------------------------------------
# Device + LoRA
# ---------------------------------------------------------------------------


def _resolve_device(device: str) -> torch.device:
    """Backward-compatible alias for :func:`maatml.device.resolve_device`."""
    from ..device import resolve_device

    return resolve_device(device)


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
    target_field: Optional[str] = None,
    config_cls: Type[SFTTrainConfig] = SFTTrainConfig,
    request_field: Optional[str] = None,
    user_placeholder: Optional[str] = None,
    default_prompt_spec: Optional[Path] = None,
    smoke: bool = False,
    dataset_dir: Optional[Path] = None,
    out_dir: Optional[Path] = None,
    prompt_spec_path: Optional[str | Path] = None,
    limit: Optional[int] = None,
    device: str = "auto",
    seed: Optional[int] = None,
    resume: Optional[str] = None,
    trial: Optional[dict[str, Any]] = None,
    log_label: str = "SFT",
) -> SFTTrainResult:
    """The shared SFT training driver. Each task module is a thin wrapper
    that pins `target_field`, `request_field`, and `default_prompt_spec`.

    bf16 autocast is enabled on MPS/CUDA when `precision: bf16`. Weights
    follow ``DeviceProfile.weights_dtype_policy`` (fp32 master on mps/cpu;
    native bf16/fp16 on cuda when precision matches). Optional
    ``training.quantization`` enables QLoRA (CUDA + ``maatml[cuda]`` only).

    ``target_field`` / ``request_field`` / ``user_placeholder`` default from
    ``dataset:`` (or ``data:``) when not passed explicitly.
    """
    training_dict = dict(model_def.merged_smoke() if smoke else model_def.training)
    embedding_strategy = training_dict.pop("embedding_strategy", None)
    # Drop keys not modeled on SFTTrainConfig (task-specific passthrough).
    for drop in ("generation", "heads", "head_loss_weights"):
        training_dict.pop(drop, None)
    cfg = config_cls(**training_dict)
    if seed is not None:
        cfg.seed = seed
    set_seed(cfg.seed)

    ds_cfg = get_dataset_cfg(model_def)
    target_field = target_field or ds_cfg.get("target_field") or "expected_output"
    request_field = request_field or ds_cfg.get("request_field") or ds_cfg.get("raw_field") or "request"
    user_placeholder = (
        user_placeholder or ds_cfg.get("user_placeholder") or "<<USER_REQUEST>>"
    )

    if prompt_spec_path is not None:
        spec_path = Path(prompt_spec_path)
    elif "prompt_spec" in ds_cfg:
        spec_path = model_def.resolve(ds_cfg["prompt_spec"])
    elif default_prompt_spec is not None:
        spec_path = default_prompt_spec
    else:
        raise ValueError(
            "no prompt_spec_path provided and model_def has no `dataset/data.prompt_spec`"
        )
    prompt_spec = read_json(spec_path)

    dataset_dir = Path(dataset_dir) if dataset_dir else model_def.prepared_dir
    target_device, profile, distributed = resolve_training_placement(device)

    run, out_dir, resume_path = begin_training_run(
        model_def,
        smoke=smoke,
        device=str(target_device),
        profile=profile.name,
        out_dir=out_dir,
        resume=resume,
        trial=trial,
    )

    # The run record exists from here on: reading the splits is fallible too,
    # so it belongs inside the handler that marks the run `aborted`.
    try:
        train_rows = list(iter_jsonl(dataset_dir / "train.jsonl"))
        val_rows = list(iter_jsonl(dataset_dir / "val.jsonl"))
        if limit is not None:
            train_rows = train_rows[:limit]
            val_rows = val_rows[: max(2, limit // 4)]
        if not train_rows:
            raise ValueError(f"No training rows in {dataset_dir / 'train.jsonl'}")

        console.print(
            f"[cyan]{log_label} train[/]: run={run.run_id} model={cfg.model_id} "
            f"train={len(train_rows)} val={len(val_rows)} lora={cfg.lora.enabled} "
            f"precision={cfg.precision}"
            + (f" quant={cfg.quantization.enabled()}" if cfg.quantization else "")
        )

        tokenizer = AutoTokenizer.from_pretrained(
            cfg.model_id, revision=cfg.model_revision
        )
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token

        use_bf16 = cfg.precision == "bf16" and (
            distributed or target_device.type in ("cuda", "mps")
        )
        use_fp16 = cfg.precision == "fp16" and (
            distributed or target_device.type in ("cuda", "mps")
        )
        quant = cfg.quantization if cfg.quantization and cfg.quantization.enabled() else None
        load_kwargs = from_pretrained_kwargs(
            profile,
            precision=cfg.precision,
            attn_implementation=cfg.attn_implementation,
            quantization=quant,
            revision=cfg.model_revision,
        )
        model = AutoModelForCausalLM.from_pretrained(cfg.model_id, **load_kwargs)
        model = maybe_prepare_kbit(model, quant)
        ensure_tokenizer_model_contract(
            model,
            tokenizer,
            embedding_strategy=embedding_strategy,
        )
        model.config.pad_token_id = tokenizer.pad_token_id
        model.config.bos_token_id = tokenizer.bos_token_id
        model.config.eos_token_id = tokenizer.eos_token_id
        if hasattr(model, "generation_config") and model.generation_config is not None:
            model.generation_config.pad_token_id = tokenizer.pad_token_id
            model.generation_config.bos_token_id = tokenizer.bos_token_id
            model.generation_config.eos_token_id = tokenizer.eos_token_id
        model = _maybe_attach_lora(model, cfg.lora)

        train_file = dataset_dir / "train.jsonl"
        val_file = dataset_dir / "val.jsonl"
        prompt_hash = sha256_file(spec_path) if spec_path.is_file() else stable_hash(prompt_spec)
        cache_key = stable_hash(
            _tokenizer_cache_identity(tokenizer),
            cfg.max_input_tokens,
            prompt_hash,
            sha256_file(train_file) if train_file.is_file() else "",
            # val.jsonl content too: re-preparing with the same train split but
            # a different val split used to reuse a stale val cache.
            sha256_file(val_file) if val_file.is_file() else "",
            "causal_sft",
            target_field,
            request_field,
            user_placeholder,
            limit,
        )
        cache_dir = model_def.output_dir / "cache"
        train_cache = cache_dir / f"sft_train_{cache_key[:16]}.pt"
        val_cache = cache_dir / f"sft_val_{cache_key[:16]}.pt"

        def _tok(row: dict) -> dict:
            return build_chat_example(
                row,
                prompt_spec,
                tokenizer,
                max_length=cfg.max_input_tokens,
                target_field=target_field,
                request_field=request_field,
                user_placeholder=user_placeholder,
            )

        train_tok = _load_or_build_tokenized_cache(train_rows, train_cache, _tok)
        val_tok = (
            _load_or_build_tokenized_cache(val_rows, val_cache, _tok) if val_rows else []
        )

        collator = SFTDataCollator(
            tokenizer,
            prompt_spec,
            max_length=cfg.max_input_tokens,
            target_field=target_field,
            request_field=request_field,
            user_placeholder=user_placeholder,
            pretokenized=True,
        )
        train_ds = _ListDataset(train_tok)
        val_ds = _ListDataset(val_tok) if val_tok else None

        total_steps = (
            int(len(train_rows) / cfg.batch_size / cfg.grad_accum * cfg.epochs)
            if cfg.max_steps < 0
            else cfg.max_steps
        )
        use_grad_ckpt = bool(cfg.grad_checkpointing) and profile.allow_grad_checkpointing
        run_eval_during_training = (
            val_ds is not None
            and profile.allow_mid_train_eval
            and cfg.eval_steps < total_steps
        )
        warmup_steps = max(0, int(round(total_steps * cfg.warmup_ratio)))
        report_to = normalize_report_to(cfg.report_to)
        num_workers = effective_dataloader_workers(profile, cfg.dataloader_workers)

        args_kwargs: dict = {
            "output_dir": str(out_dir),
            "run_name": run.run_id,
            "per_device_train_batch_size": cfg.batch_size,
            "per_device_eval_batch_size": cfg.batch_size,
            "gradient_accumulation_steps": cfg.grad_accum,
            "learning_rate": cfg.learning_rate,
            "num_train_epochs": cfg.epochs,
            "weight_decay": cfg.weight_decay,
            "warmup_steps": warmup_steps,
            "logging_steps": cfg.logging_steps,
            "eval_strategy": "steps" if run_eval_during_training else "no",
            "eval_steps": cfg.eval_steps if run_eval_during_training else None,
            "save_strategy": "steps",
            "save_steps": cfg.save_steps,
            "save_total_limit": 2,
            "seed": cfg.seed,
            "bf16": use_bf16,
            "fp16": use_fp16,
            "gradient_checkpointing": use_grad_ckpt,
            "dataloader_num_workers": num_workers,
            "report_to": report_to,
            "optim": "adamw_torch",
            "max_steps": cfg.max_steps,
            # Distributed: let HF Trainer / accelerate place the model.
            "use_cpu": (not distributed) and target_device.type == "cpu",
            "remove_unused_columns": False,
            "length_column_name": "length",
        }
        # transformers≥5 dropped group_by_length from TrainingArguments.
        import inspect

        if "group_by_length" in inspect.signature(TrainingArguments.__init__).parameters:
            args_kwargs["group_by_length"] = bool(cfg.group_by_length)
        args = TrainingArguments(**args_kwargs)  # type: ignore[call-arg]

        trainer = Trainer(
            model=model,
            args=args,
            train_dataset=train_ds,
            eval_dataset=val_ds if run_eval_during_training else None,
            data_collator=collator,
            processing_class=tokenizer,
            callbacks=[make_nan_guard_callback()],
        )

        train_output = trainer.train(
            resume_from_checkpoint=str(resume_path) if resume_path else None
        )

        eval_metrics: dict[str, Any] = {}
        if val_ds is not None:
            profile.empty_cache()
            eval_metrics = trainer.evaluate(eval_dataset=val_ds) or {}

        save_meta = _save_sft_artifacts(
            model,
            tokenizer,
            out_dir,
            save_mode=cfg.lora.save_mode if cfg.lora.enabled else "merged",
            base_model_id=cfg.model_id,
            spec_path=spec_path,
        )

        metrics_out = {
            k: float(v) for k, v in eval_metrics.items() if isinstance(v, (int, float))
        }
        write_run_metadata(
            out_dir,
            model_def,
            {
                "train": dataset_dir / "train.jsonl",
                "val": dataset_dir / "val.jsonl",
            },
            extra={
                "run_id": run.run_id,
                "smoke": smoke,
                "device": str(target_device),
                "profile": profile.name,
                "distributed": distributed,
                "model_revision": cfg.model_revision,
                **save_meta,
            },
        )
        finish_run(model_def, run.run_id, "completed", metrics=metrics_out)

        return SFTTrainResult(
            out_dir=out_dir,
            metrics=metrics_out,
            train_runtime=float(
                getattr(train_output, "metrics", {}).get("train_runtime", 0.0)
                if hasattr(train_output, "metrics")
                else 0.0
            ),
        )
    except Exception as exc:
        finish_run(model_def, run.run_id, "aborted", error=str(exc))
        raise

