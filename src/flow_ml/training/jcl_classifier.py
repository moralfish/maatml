"""JCL Validator — ModernBERT-base multi-head classifier.

Full fine-tune of `answerdotai/ModernBERT-base` with four classification
heads (validity / error_code / severity / line_localization). Loss is a
weighted sum across heads per the per-model `model.yml`. No LoRA — the
model is small enough (~150 MB params) that full fine-tune is cheap and
the resulting safetensors merge straight into the package.

The custom JCL tokenizer (column-aware pre-tokenizer + BPE, see
`flow_ml.tokenization`) is loaded at training time via the
`datasets/tokenizer.json` declared in `model.yml`. Pre-tokenization is
applied per-sample before encoding.

Public surface:
  - `JclClassifierConfig` — typed config built from `model.yml::training`
  - `train_jcl_classifier(model_def, ...)` — entry point invoked by the CLI
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from ..config import ModelDefinition, get_dataset_cfg
from ..device import get_profile, resolve_device
from ..utils.io import iter_jsonl
from .guards import ensure_tokenizer_model_contract, make_nan_guard_callback, write_run_metadata


# Default error code ↔ index. Order is the canonical training-time index
# assignment; override via ``training.heads.error_codes`` / ``severities``.
DEFAULT_ERROR_CODES = [
    "missing_dd",
    "invalid_job_card",
    "unresolved_symbolic_parameter",
    "continuation_error",
    "invalid_exec_statement",
    "invalid_dataset_reference_structure",
    "other",
    "none",
]
DEFAULT_SEVERITIES = ["error", "warning", "info", "none"]

# Module-level aliases kept for evaluator imports.
ERROR_CODES = list(DEFAULT_ERROR_CODES)
SEVERITIES = list(DEFAULT_SEVERITIES)


def _resolve_heads(training_dict: dict) -> tuple[list[str], list[str]]:
    heads = training_dict.get("heads") or {}
    codes = list(heads.get("error_codes") or DEFAULT_ERROR_CODES)
    sevs = list(heads.get("severities") or DEFAULT_SEVERITIES)
    return codes, sevs


@dataclass
class HeadLossWeights:
    validity: float = 1.0
    error_code: float = 1.0
    severity: float = 0.5
    line: float = 0.3


@dataclass
class JclClassifierConfig:
    model_id: str = "answerdotai/ModernBERT-base"
    max_input_tokens: int = 2048
    batch_size: int = 16
    grad_accum: int = 1
    learning_rate: float = 2.0e-5
    epochs: int = 4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.06
    seed: int = 1337
    precision: str = "bf16"
    grad_checkpointing: bool = False
    eval_steps: int = 9999
    save_steps: int = 250
    logging_steps: int = 25
    max_steps: int = -1
    head_loss_weights: HeadLossWeights = field(default_factory=HeadLossWeights)

    @classmethod
    def from_dict(cls, d: dict) -> "JclClassifierConfig":
        head = d.get("head_loss_weights") or {}
        return cls(
            model_id=d.get("model_id", "answerdotai/ModernBERT-base"),
            max_input_tokens=int(d.get("max_input_tokens", 2048)),
            batch_size=int(d.get("batch_size", 16)),
            grad_accum=int(d.get("grad_accum", 1)),
            learning_rate=float(d.get("learning_rate", 2.0e-5)),
            epochs=int(d.get("epochs", 4)),
            weight_decay=float(d.get("weight_decay", 0.01)),
            warmup_ratio=float(d.get("warmup_ratio", 0.06)),
            seed=int(d.get("seed", 1337)),
            precision=d.get("precision", "bf16"),
            grad_checkpointing=bool(d.get("grad_checkpointing", False)),
            eval_steps=int(d.get("eval_steps", 9999)),
            save_steps=int(d.get("save_steps", 250)),
            logging_steps=int(d.get("logging_steps", 25)),
            max_steps=int(d.get("max_steps", -1)),
            head_loss_weights=HeadLossWeights(
                validity=float(head.get("validity", 1.0)),
                error_code=float(head.get("error_code", 1.0)),
                severity=float(head.get("severity", 0.5)),
                line=float(head.get("line", 0.3)),
            ),
        )


@dataclass
class JclClassifierResult:
    out_dir: Path
    metrics: dict
    train_runtime: float


def _classifier_head_module(
    hidden_size: int, error_codes: list[str], severities: list[str]
):
    """Construct the 4-head classifier module on top of a BERT encoder."""
    import torch
    from torch import nn

    n_codes = len(error_codes)
    n_sevs = len(severities)

    class JclClassifierHeads(nn.Module):
        def __init__(self, hidden: int) -> None:
            super().__init__()
            self.validity = nn.Linear(hidden, 2)
            self.error_code = nn.Linear(hidden, n_codes)
            self.severity = nn.Linear(hidden, n_sevs)
            self.line = nn.Linear(hidden, 2)  # per-token binary

        def forward(
            self, sequence_output: "torch.Tensor", pooled: "torch.Tensor"
        ) -> dict[str, "torch.Tensor"]:
            return {
                "validity_logits": self.validity(pooled),
                "error_code_logits": self.error_code(pooled),
                "severity_logits": self.severity(pooled),
                "line_logits": self.line(sequence_output),
            }

    return JclClassifierHeads(hidden_size)


def _build_dataset(
    rows: list[dict],
    tokenizer,
    max_length: int,
    error_codes: list[str],
    severities: list[str],
):
    """Flatten prepared JSONL rows into per-head classifier targets."""
    from torch.utils.data import Dataset
    import torch

    from ..tokenization import pre_tokenize_jcl

    none_code = error_codes.index("none") if "none" in error_codes else 0
    none_sev = severities.index("none") if "none" in severities else 0

    class _JclDataset(Dataset):
        def __init__(self, samples: list[dict]) -> None:
            self.samples = samples

        def __len__(self) -> int:
            return len(self.samples)

        def __getitem__(self, idx: int) -> dict:
            row = self.samples[idx]
            jcl = row.get("request", "")
            gold = row.get("expected_validation_result") or {}
            errors = gold.get("errors") or []
            primary = errors[0] if errors else {}

            valid = 1 if gold.get("valid", True) else 0
            code = primary.get("code", "none")
            sev = primary.get("severity", "none")
            err_line = int(primary.get("line") or 0)

            code_idx = error_codes.index(code) if code in error_codes else none_code
            sev_idx = severities.index(sev) if sev in severities else none_sev

            pre = pre_tokenize_jcl(jcl)
            enc = tokenizer(
                pre,
                max_length=max_length,
                padding="max_length",
                truncation=True,
                return_offsets_mapping=True,
                return_tensors=None,
            )

            input_ids = enc["input_ids"]
            attention_mask = enc["attention_mask"]
            offsets = enc.get("offset_mapping") or [(0, 0)] * len(input_ids)

            line_labels = [-100] * len(input_ids)
            if err_line > 0:
                start_char = _line_start_offset(pre, err_line)
                end_char = _line_start_offset(pre, err_line + 1)
                for i, (s, e) in enumerate(offsets):
                    if attention_mask[i] == 0:
                        continue
                    if s == e == 0:
                        line_labels[i] = -100
                        continue
                    line_labels[i] = 1 if (s >= start_char and s < end_char) else 0
            else:
                for i, (s, e) in enumerate(offsets):
                    if attention_mask[i] == 0 or s == e == 0:
                        continue
                    line_labels[i] = 0

            return {
                "input_ids": torch.tensor(input_ids, dtype=torch.long),
                "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
                "validity_label": torch.tensor(valid, dtype=torch.long),
                "error_code_label": torch.tensor(code_idx, dtype=torch.long),
                "severity_label": torch.tensor(sev_idx, dtype=torch.long),
                "line_labels": torch.tensor(line_labels, dtype=torch.long),
            }

    return _JclDataset(rows)


def _line_start_offset(text: str, line_no: int) -> int:
    """1-based line number → character offset of that line's start."""
    if line_no <= 1:
        return 0
    count = 0
    for i, ch in enumerate(text):
        if ch == "\n":
            count += 1
            if count == line_no - 1:
                return i + 1
    return len(text)


def _train_loop(
    cfg: JclClassifierConfig,
    train_ds,
    val_ds,
    out_dir: Path,
    tokenizer,
    *,
    error_codes: list[str],
    severities: list[str],
    profile,
    embedding_strategy: Optional[str],
    model_def: ModelDefinition,
    dataset_dir: Path,
    smoke: bool,
    device_name: str,
) -> JclClassifierResult:
    """Inner training loop with multi-head loss via Trainer.compute_loss override."""
    from torch import nn
    from transformers import (
        AutoModel,
        TrainingArguments,
        Trainer,
        set_seed,
    )

    set_seed(cfg.seed)

    encoder = AutoModel.from_pretrained(cfg.model_id)
    ensure_tokenizer_model_contract(
        encoder, tokenizer, embedding_strategy=embedding_strategy
    )
    hidden = encoder.config.hidden_size
    heads = _classifier_head_module(hidden, error_codes, severities)

    class JclMultiHead(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.encoder = encoder
            self.heads = heads
            self.ce = nn.CrossEntropyLoss()
            self.ce_line = nn.CrossEntropyLoss(ignore_index=-100)

        def forward(
            self,
            input_ids,
            attention_mask,
            validity_label=None,
            error_code_label=None,
            severity_label=None,
            line_labels=None,
        ):
            out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
            seq = out.last_hidden_state
            pooled = seq[:, 0, :]
            heads_out = self.heads(seq, pooled)
            loss = None
            if validity_label is not None:
                w = cfg.head_loss_weights
                lv = self.ce(heads_out["validity_logits"], validity_label)
                lc = self.ce(heads_out["error_code_logits"], error_code_label)
                ls = self.ce(heads_out["severity_logits"], severity_label)
                ll = self.ce_line(
                    heads_out["line_logits"].view(-1, 2),
                    line_labels.view(-1),
                )
                loss = (
                    w.validity * lv
                    + w.error_code * lc
                    + w.severity * ls
                    + w.line * ll
                )
            return {"loss": loss, **heads_out}

    model = JclMultiHead()

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
        remove_unused_columns=False,
    )

    class _MultiHeadTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            outputs = model(**inputs)
            loss = outputs["loss"]
            return (loss, outputs) if return_outputs else loss

    trainer = _MultiHeadTrainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds if run_eval else None,
        processing_class=tokenizer,
        callbacks=[make_nan_guard_callback()],
    )

    train_output = trainer.train()
    eval_metrics: dict = {}
    if val_ds is not None:
        profile.empty_cache()
        eval_metrics = trainer.evaluate(eval_dataset=val_ds) or {}

    # Safetensors only: encoder via save_pretrained + heads sidecar.
    model.eval()
    encoder.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)

    from safetensors.torch import save_file

    head_state = {
        f"heads.{name}.{param_name}": tensor.detach().cpu()
        for name, head in [
            ("validity", model.heads.validity),
            ("error_code", model.heads.error_code),
            ("severity", model.heads.severity),
            ("line", model.heads.line),
        ]
        for param_name, tensor in head.state_dict().items()
    }
    save_file(head_state, out_dir / "classifier_heads.safetensors")

    write_run_metadata(
        out_dir,
        model_def,
        {
            "train": dataset_dir / "train.jsonl",
            "val": dataset_dir / "val.jsonl",
        },
        extra={
            "smoke": smoke,
            "device": device_name,
            "profile": profile.name,
            "error_codes": error_codes,
            "severities": severities,
        },
    )

    return JclClassifierResult(
        out_dir=out_dir,
        metrics={k: float(v) for k, v in eval_metrics.items() if isinstance(v, (int, float))},
        train_runtime=float(
            getattr(train_output, "metrics", {}).get("train_runtime", 0.0)
            if hasattr(train_output, "metrics")
            else 0.0
        ),
    )


def train_jcl_classifier(
    model_def: ModelDefinition,
    *,
    smoke: bool = False,
    dataset_dir: Optional[Path] = None,
    out_dir: Optional[Path] = None,
    tokenizer_path: Optional[str | Path] = None,
    limit: Optional[int] = None,
    device: str = "auto",
    seed: Optional[int] = None,
) -> JclClassifierResult:
    """Train the JCL classifier from a `ModelDefinition`."""
    from transformers import AutoTokenizer

    training_dict = model_def.merged_smoke() if smoke else dict(model_def.training)
    cfg = JclClassifierConfig.from_dict(training_dict)
    if seed is not None:
        cfg.seed = seed

    error_codes, severities = _resolve_heads(training_dict)
    # Keep module-level aliases in sync for any mid-run evaluator imports.
    global ERROR_CODES, SEVERITIES
    ERROR_CODES = list(error_codes)
    SEVERITIES = list(severities)

    dataset_dir = Path(dataset_dir) if dataset_dir else model_def.prepared_dir
    if out_dir is None:
        run_name = "smoke" if smoke else model_def.identity
        out_dir = model_def.checkpoints_dir / run_name
    else:
        out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ds_cfg = get_dataset_cfg(model_def)
    resolved_tokenizer = (
        Path(tokenizer_path)
        if tokenizer_path
        else (
            model_def.resolve(ds_cfg["tokenizer"])
            if ds_cfg.get("tokenizer")
            else None
        )
    )
    if resolved_tokenizer and resolved_tokenizer.exists():
        from transformers import PreTrainedTokenizerFast

        tokenizer: Any = PreTrainedTokenizerFast(
            tokenizer_file=str(resolved_tokenizer),
            model_max_length=cfg.max_input_tokens,
            pad_token="<PAD>",
            unk_token="<UNK>",
            cls_token="<CLS>",
            sep_token="<SEP>",
            mask_token="<MASK>",
            additional_special_tokens=["<COL1>", "<CONT>"],
        )
    else:
        tokenizer = AutoTokenizer.from_pretrained(cfg.model_id)

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
        f"JCL classifier: model={cfg.model_id} train={len(train_rows)} "
        f"val={len(val_rows)} max_len={cfg.max_input_tokens} epochs={cfg.epochs}"
    )

    train_ds = _build_dataset(
        train_rows, tokenizer, cfg.max_input_tokens, error_codes, severities
    )
    val_ds = (
        _build_dataset(val_rows, tokenizer, cfg.max_input_tokens, error_codes, severities)
        if val_rows
        else None
    )

    return _train_loop(
        cfg,
        train_ds,
        val_ds,
        out_dir,
        tokenizer,
        error_codes=error_codes,
        severities=severities,
        profile=profile,
        embedding_strategy=training_dict.get("embedding_strategy"),
        model_def=model_def,
        dataset_dir=dataset_dir,
        smoke=smoke,
        device_name=str(target_device),
    )
