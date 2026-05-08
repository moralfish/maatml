from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from pydantic import BaseModel, ConfigDict, Field
from rich.console import Console
from safetensors.torch import load_file as safetensors_load
from safetensors.torch import save_file as safetensors_save
from torch.utils.data import Dataset
from transformers import (
    AutoModel,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    Trainer,
    TrainingArguments,
    set_seed,
)

from ..data.schemas import ErrorCategory
from ..config import ModelDefinition
from ..utils.io import iter_jsonl, read_yaml

console = Console()

CATEGORY_INDEX: dict[str, int] = {c.value: i for i, c in enumerate(ErrorCategory)}
CATEGORY_LABELS: list[str] = [c.value for c in ErrorCategory]
NUM_CATEGORIES = len(CATEGORY_LABELS)
NONE_INDEX = CATEGORY_INDEX[ErrorCategory.none.value]


class HeadWeights(BaseModel):
    seq: float = 1.0
    cat: float = 1.0
    line: float = 0.5


class JclTrainConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model_id: str = "answerdotai/ModernBERT-base"
    max_input_tokens: int = Field(default=2048, gt=0)
    batch_size: int = Field(default=4, gt=0)
    grad_accum: int = Field(default=2, gt=0)
    learning_rate: float = 2e-5
    epochs: float = 3.0
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    seed: int = 1337
    precision: str = "fp32"
    grad_checkpointing: bool = True
    eval_steps: int = 200
    save_steps: int = 200
    logging_steps: int = 25
    head_weights: HeadWeights = Field(default_factory=HeadWeights)
    max_steps: int = -1

    @classmethod
    def from_yaml(cls, path: str | Path) -> "JclTrainConfig":
        return cls(**read_yaml(path))


@dataclass
class JclTrainResult:
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


class JclMultiHeadModel(nn.Module):
    """Shared encoder + 3 heads (sequence, category, per-token line)."""

    def __init__(
        self,
        encoder: PreTrainedModel,
        *,
        num_categories: int = NUM_CATEGORIES,
        head_weights: Optional[HeadWeights] = None,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        hidden = encoder.config.hidden_size
        self.num_categories = num_categories
        self.head_weights = head_weights or HeadWeights()
        self.seq_head = nn.Linear(hidden, 2)
        self.cat_head = nn.Linear(hidden, num_categories)
        self.line_head = nn.Linear(hidden, 2)

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def gradient_checkpointing_enable(self, **kwargs: Any) -> None:
        if hasattr(self.encoder, "gradient_checkpointing_enable"):
            self.encoder.gradient_checkpointing_enable(**kwargs)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        seq_label: Optional[torch.Tensor] = None,
        cat_label: Optional[torch.Tensor] = None,
        line_labels: Optional[torch.Tensor] = None,
        **_: Any,
    ) -> dict[str, torch.Tensor]:
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        hidden = out.last_hidden_state  # [B, T, H]
        mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)

        seq_logits = self.seq_head(pooled)
        cat_logits = self.cat_head(pooled)
        line_logits = self.line_head(hidden)

        result: dict[str, torch.Tensor] = {
            "seq_logits": seq_logits,
            "cat_logits": cat_logits,
            "line_logits": line_logits,
        }

        if seq_label is not None and cat_label is not None and line_labels is not None:
            loss_seq = F.cross_entropy(seq_logits, seq_label)
            loss_cat = F.cross_entropy(cat_logits, cat_label)
            loss_line = F.cross_entropy(
                line_logits.reshape(-1, 2),
                line_labels.reshape(-1),
                ignore_index=-100,
            )
            w = self.head_weights
            loss = w.seq * loss_seq + w.cat * loss_cat + w.line * loss_line
            result["loss"] = loss
        return result

    def save(self, out_dir: str | Path, *, base_model_id: Optional[str] = None) -> Path:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        self.encoder.save_pretrained(out)
        head_state: dict[str, torch.Tensor] = {}
        for prefix, head in (("seq_head", self.seq_head), ("cat_head", self.cat_head), ("line_head", self.line_head)):
            for k, v in head.state_dict().items():
                head_state[f"{prefix}.{k}"] = v.detach().cpu().contiguous()
        safetensors_save(head_state, str(out / "flow_heads.safetensors"))
        meta = {
            "flow_heads": {
                "seq": {"out": 2, "type": "sequence_classification"},
                "cat": {"out": self.num_categories, "labels": CATEGORY_LABELS, "type": "category_classification"},
                "line": {"out": 2, "type": "token_classification", "aggregation": "mean_per_line"},
            },
            "flow_head_weights": self.head_weights.model_dump(),
        }
        if base_model_id:
            meta["flow_base_model_id"] = base_model_id
        flow_meta_path = out / "flow_metadata.json"
        flow_meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True), encoding="utf-8")
        return out

    @classmethod
    def load(cls, pkg_dir: str | Path) -> "JclMultiHeadModel":
        pkg = Path(pkg_dir)
        encoder = AutoModel.from_pretrained(pkg)
        meta_path = pkg / "flow_metadata.json"
        meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
        head_meta = meta.get("flow_heads", {})
        num_cat = head_meta.get("cat", {}).get("out", NUM_CATEGORIES)
        weights = HeadWeights(**meta.get("flow_head_weights", {}))
        model = cls(encoder, num_categories=num_cat, head_weights=weights)
        head_path = pkg / "flow_heads.safetensors"
        if head_path.exists():
            state = safetensors_load(str(head_path))
            seq_state = {k.split(".", 1)[1]: v for k, v in state.items() if k.startswith("seq_head.")}
            cat_state = {k.split(".", 1)[1]: v for k, v in state.items() if k.startswith("cat_head.")}
            line_state = {k.split(".", 1)[1]: v for k, v in state.items() if k.startswith("line_head.")}
            model.seq_head.load_state_dict(seq_state)
            model.cat_head.load_state_dict(cat_state)
            model.line_head.load_state_dict(line_state)
        return model


class JclCollator:
    """Tokenize a batch of JclSample-shaped dicts and build the three label tensors."""

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        *,
        max_length: int,
        category_index: dict[str, int] = CATEGORY_INDEX,
    ) -> None:
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.category_index = category_index

    def _line_labels_for(self, text: str, error_line: Optional[int], offsets: list[tuple[int, int]]) -> list[int]:
        if error_line is None:
            return [-100 if (s == 0 and e == 0) else 0 for s, e in offsets]
        newline_starts = [i + 1 for i, c in enumerate(text) if c == "\n"]
        labels: list[int] = []
        for start, end in offsets:
            if start == 0 and end == 0:
                labels.append(-100)
                continue
            line_id = sum(1 for ns in newline_starts if ns <= start) + 1
            labels.append(1 if line_id == error_line else 0)
        return labels

    def __call__(self, batch: list[dict]) -> dict[str, torch.Tensor]:
        texts = [row["sanitized_jcl"] for row in batch]
        enc = self.tokenizer(
            texts,
            max_length=self.max_length,
            truncation=True,
            padding=True,
            return_offsets_mapping=True,
            return_tensors="pt",
        )
        seq_label = torch.tensor([0 if row["is_valid"] else 1 for row in batch], dtype=torch.long)
        cat_label = torch.tensor(
            [self.category_index[row.get("error_category") or ErrorCategory.none.value] for row in batch],
            dtype=torch.long,
        )
        line_labels = torch.full(enc["input_ids"].shape, -100, dtype=torch.long)
        offsets_batch = enc.pop("offset_mapping")
        for i, row in enumerate(batch):
            offsets_i = offsets_batch[i].tolist()
            labels = self._line_labels_for(row["sanitized_jcl"], row.get("error_line"), offsets_i)
            attn = enc["attention_mask"][i]
            for t, lab in enumerate(labels):
                if attn[t] == 0:
                    line_labels[i, t] = -100
                else:
                    line_labels[i, t] = lab
        return {
            "input_ids": enc["input_ids"],
            "attention_mask": enc["attention_mask"],
            "seq_label": seq_label,
            "cat_label": cat_label,
            "line_labels": line_labels,
        }


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


def _build_encoder(model_id: str) -> PreTrainedModel:
    try:
        return AutoModel.from_pretrained(model_id, attn_implementation="eager")
    except (TypeError, ValueError):
        return AutoModel.from_pretrained(model_id)


def train_jcl(
    model_def: ModelDefinition,
    *,
    smoke: bool = False,
    dataset_dir: Optional[Path] = None,
    out_dir: Optional[Path] = None,
    limit: Optional[int] = None,
    device: str = "auto",
    seed: Optional[int] = None,
) -> JclTrainResult:
    """Fine-tune the JCL Validator multi-head BERT.

    Reads training hyperparameters from ``model_def.training`` (or the smoke
    overlay when ``smoke=True``).  Defaults dataset/output paths to the
    canonical ``output/prepared/`` and ``output/checkpoints/<run-name>/``
    locations under the model folder; both are still overridable.
    """
    training_dict = model_def.merged_smoke() if smoke else dict(model_def.training)
    cfg = JclTrainConfig(**training_dict)
    if seed is not None:
        cfg.seed = seed
    set_seed(cfg.seed)

    dataset_dir = Path(dataset_dir) if dataset_dir else model_def.prepared_dir
    if out_dir is None:
        run_name = ("smoke" if smoke else cfg.model_id.split("/")[-1].lower())
        out_dir = model_def.checkpoints_dir / run_name
    else:
        out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_rows = list(iter_jsonl(dataset_dir / "train.jsonl"))
    val_rows = list(iter_jsonl(dataset_dir / "val.jsonl"))
    if limit is not None:
        train_rows = train_rows[:limit]
        val_rows = val_rows[: max(2, limit // 4)]

    console.print(f"[cyan]JCL train[/]: model={cfg.model_id} train={len(train_rows)} val={len(val_rows)}")

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_id)
    encoder = _build_encoder(cfg.model_id)
    model = JclMultiHeadModel(encoder, num_categories=NUM_CATEGORIES, head_weights=cfg.head_weights)

    target_device = _resolve_device(device)

    collator = JclCollator(tokenizer, max_length=cfg.max_input_tokens)
    train_ds = _ListDataset(train_rows)
    val_ds = _ListDataset(val_rows)

    use_bf16 = cfg.precision == "bf16" and target_device.type == "cuda"

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
        eval_strategy="steps",
        eval_steps=cfg.eval_steps,
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
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        max_steps=cfg.max_steps,
        use_cpu=target_device.type == "cpu",
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collator,
        processing_class=tokenizer,
    )

    train_output = trainer.train()
    eval_metrics = trainer.evaluate()

    saved = model.save(out_dir, base_model_id=cfg.model_id)
    tokenizer.save_pretrained(saved)
    (saved / "labels.json").write_text(
        json.dumps(
            {
                "sequence": ["valid", "invalid"],
                "category": CATEGORY_LABELS,
                "line": ["no_error", "error"],
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    return JclTrainResult(
        out_dir=saved,
        metrics={k: float(v) for k, v in eval_metrics.items() if isinstance(v, (int, float))},
        train_runtime=float(getattr(train_output, "metrics", {}).get("train_runtime", 0.0) if hasattr(train_output, "metrics") else 0.0),
    )


def train() -> None:
    """Legacy placeholder retained for the original scripts shim signature."""
    console.print("Use flow_ml.training.jcl_validator.train_jcl(config, dataset_dir, out_dir)")
