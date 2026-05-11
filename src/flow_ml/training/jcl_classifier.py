"""JCL Validator v2 — ModernBERT-base multi-head classifier.

Full fine-tune of `answerdotai/ModernBERT-base` with four classification
heads (validity / error_code / severity / line_localization). Loss is a
weighted sum across heads per the per-model `model.yml`. No LoRA — the
model is small enough (~150 MB params) that full fine-tune is cheap and
the resulting safetensors merge straight into the package.

The custom JCL tokenizer (column-aware pre-tokenizer + BPE, see
`flow_ml.tokenization`) is loaded at training time via the
`datasets/tokenizer.json` declared in `model.yml`. Pre-tokenization is
applied per-sample before encoding.

Replaces the v1 generative-SFT thin wrapper at `training/jcl_validator.py`.

Public surface:
  - `JclClassifierConfig` — typed config built from `model.yml::training`
  - `train_jcl_classifier(model_def, ...)` — entry point invoked by the CLI
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


from ..config import ModelDefinition
from ..utils.io import iter_jsonl, read_json


# Error code ↔ index. Order is the canonical training-time index assignment;
# the runtime mirrors it byte-for-byte.
ERROR_CODES = [
    "missing_dd",
    "invalid_job_card",
    "unresolved_symbolic_parameter",
    "continuation_error",
    "invalid_exec_statement",
    "invalid_dataset_reference_structure",
    "other",
    "none",
]
SEVERITIES = ["error", "warning", "info", "none"]


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


def _classifier_head_module(hidden_size: int):
    """Construct the 4-head classifier module on top of a BERT encoder.
    Kept inside a function so the heavyweight torch import is deferred.
    """
    import torch
    from torch import nn

    class JclClassifierHeads(nn.Module):
        def __init__(self, hidden: int) -> None:
            super().__init__()
            self.validity = nn.Linear(hidden, 2)
            self.error_code = nn.Linear(hidden, len(ERROR_CODES))
            self.severity = nn.Linear(hidden, len(SEVERITIES))
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


def _build_dataset(rows: list[dict], tokenizer, max_length: int):
    """Flatten the v1 SFT JSONL into per-head classifier targets.

    Each row's `expected_validation_result` is decomposed into:
      - validity: int (0 = invalid, 1 = valid)
      - error_code: int (index into ERROR_CODES)
      - severity: int (index into SEVERITIES)
      - line: list[int] — token-level binary mask, 1 on tokens whose
        character-offset overlaps the gold `errors[0].line`'s line span.
    """
    from torch.utils.data import Dataset
    import torch

    from ..tokenization import pre_tokenize_jcl

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

            code_idx = ERROR_CODES.index(code) if code in ERROR_CODES else ERROR_CODES.index("none")
            sev_idx = (
                SEVERITIES.index(sev) if sev in SEVERITIES else SEVERITIES.index("none")
            )

            # Pre-tokenize JCL with the column rules before BPE encoding.
            pre = pre_tokenize_jcl(jcl)

            # Encode with offsets so we can mark the error-line tokens.
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

            # Build line-classification labels: tokens whose offset falls on
            # the error-line get label=1, else 0. Padding gets -100 (ignored).
            line_labels = [-100] * len(input_ids)
            if err_line > 0:
                # Compute the character span of `err_line` (1-based) in `pre`.
                start_char = _line_start_offset(pre, err_line)
                end_char = _line_start_offset(pre, err_line + 1)
                for i, (s, e) in enumerate(offsets):
                    if attention_mask[i] == 0:
                        continue
                    if s == e == 0:
                        # Special token (CLS/SEP/PAD); don't supervise.
                        line_labels[i] = -100
                        continue
                    line_labels[i] = 1 if (s >= start_char and s < end_char) else 0
            else:
                # Valid sample (no error line). All non-special tokens get 0.
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
    """1-based line number → character offset of that line's start.

    `line_no` past the end returns `len(text)` so callers can use it as an
    exclusive upper bound when slicing the previous line.
    """
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
) -> JclClassifierResult:
    """Inner training loop. Custom because the multi-head loss doesn't fit
    `transformers.Trainer`'s single-loss assumption — we subclass it and
    override `compute_loss`.
    """
    import torch
    from torch import nn
    from transformers import (
        AutoModel,
        TrainingArguments,
        Trainer,
        set_seed,
    )

    set_seed(cfg.seed)

    encoder = AutoModel.from_pretrained(cfg.model_id)
    hidden = encoder.config.hidden_size
    heads = _classifier_head_module(hidden)

    class JclMultiHead(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.encoder = encoder
            self.heads = heads
            # cross-entropy with -100 ignore for line labels handles padding.
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
            # Pool via CLS for classification heads. ModernBERT puts the
            # CLS token at index 0 by template-processor convention.
            pooled = seq[:, 0, :]
            heads_out = self.heads(seq, pooled)
            loss = None
            if validity_label is not None:
                w = cfg.head_loss_weights
                lv = self.ce(heads_out["validity_logits"], validity_label)
                lc = self.ce(heads_out["error_code_logits"], error_code_label)
                ls = self.ce(heads_out["severity_logits"], severity_label)
                # line_logits shape: (B, T, 2) → flatten to (B*T, 2)
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
        eval_dataset=val_ds,
        processing_class=tokenizer,
    )

    train_output = trainer.train()
    eval_metrics: dict = {}
    if val_ds is not None:
        eval_metrics = trainer.evaluate(eval_dataset=val_ds) or {}

    # Save the fused encoder + heads under one safetensors blob.
    model.eval()
    torch.save(model.state_dict(), out_dir / "pytorch_model.bin")
    encoder.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)

    # Also dump head weights as a sidecar so the Rust backend can locate
    # them without having to load the whole state_dict.
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
    """Train the v2 JCL classifier from a `ModelDefinition`.

    `tokenizer_path` defaults to `model_def.data.tokenizer` resolved
    relative to the model dir; pass an override path during smoke runs
    before the BPE has been trained (the smoke run can use the stock
    ModernBERT tokenizer).
    """
    from transformers import AutoTokenizer

    training_dict = model_def.merged_smoke() if smoke else dict(model_def.training)
    cfg = JclClassifierConfig.from_dict(training_dict)
    if seed is not None:
        cfg.seed = seed

    dataset_dir = Path(dataset_dir) if dataset_dir else model_def.prepared_dir
    if out_dir is None:
        run_name = "smoke" if smoke else model_def.model_id.replace(":", "-")
        out_dir = model_def.checkpoints_dir / run_name
    else:
        out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Tokenizer: prefer the trained JCL BPE; fall back to the stock
    # ModernBERT tokenizer (only useful for smoke). The fallback is
    # transparent — encodes work, just without the column-aware special
    # tokens, so eval gates may miss until the JCL tokenizer is trained.
    resolved_tokenizer = (
        Path(tokenizer_path)
        if tokenizer_path
        else (
            model_def.resolve(model_def.data["tokenizer"])
            if model_def.data.get("tokenizer")
            else None
        )
    )
    if resolved_tokenizer and resolved_tokenizer.exists():
        from transformers import PreTrainedTokenizerFast

        tokenizer = PreTrainedTokenizerFast(
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

    print(
        f"JCL classifier: model={cfg.model_id} train={len(train_rows)} "
        f"val={len(val_rows)} max_len={cfg.max_input_tokens} epochs={cfg.epochs}"
    )

    train_ds = _build_dataset(train_rows, tokenizer, cfg.max_input_tokens)
    val_ds = _build_dataset(val_rows, tokenizer, cfg.max_input_tokens) if val_rows else None

    return _train_loop(cfg, train_ds, val_ds, out_dir, tokenizer)
