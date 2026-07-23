"""``vlm_sft`` trainer, SmolVLM / Idefics3 LoRA SFT with frozen vision tower."""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from maatml.config import ModelDefinition, get_dataset_cfg
from maatml.device import get_profile, resolve_device
from maatml.runs import begin_training_run, finish_run
from maatml.training.guards import write_run_metadata
from maatml.utils.io import iter_jsonl


@dataclass
class VlmTrainResult:
    out_dir: Path
    metrics: dict[str, float] = field(default_factory=dict)
    train_runtime: float = 0.0


_DEFAULT_USER_PROMPT = (
    "Describe this synthetic scene in one short factual sentence covering "
    "the background style, any colored shapes, and the stick figure's pose."
)


def _user_prompt(model_def: ModelDefinition) -> str:
    cfg = get_dataset_cfg(model_def)
    if isinstance(cfg.get("prompt_spec"), str):
        try:
            from maatml.utils.io import read_json

            spec = read_json(model_def.resolve(cfg["prompt_spec"]))
            if isinstance(spec.get("user_template"), str) and spec["user_template"].strip():
                # If template is just a placeholder, fall back to system-guided default.
                ut = spec["user_template"].strip()
                if "<<USER_REQUEST>>" not in ut and "{image}" not in ut:
                    return ut
            if isinstance(spec.get("system"), str) and "Describe" in spec["system"]:
                # Prefer a short fixed user turn; system goes into chat template separately.
                pass
        except Exception:  # noqa: BLE001
            pass
    return str(
        (model_def.training or {}).get("user_prompt") or _DEFAULT_USER_PROMPT
    )


def _build_messages(user_text: str, assistant_text: str | None = None) -> list[dict]:
    user_content: list[dict[str, Any]] = [
        {"type": "image"},
        {"type": "text", "text": user_text},
    ]
    messages: list[dict[str, Any]] = [{"role": "user", "content": user_content}]
    if assistant_text is not None:
        messages.append(
            {"role": "assistant", "content": [{"type": "text", "text": assistant_text}]}
        )
    return messages


def _resolve_image(path: Path):
    from PIL import Image

    return Image.open(path).convert("RGB")


def train_vlm_sft(
    model_def: ModelDefinition,
    *,
    smoke: bool = False,
    dataset_dir: Optional[Path] = None,
    out_dir: Optional[Path] = None,
    limit: Optional[int] = None,
    device: str = "auto",
    seed: Optional[int] = None,
    resume: Optional[str] = None,
    trial: Optional[dict[str, Any]] = None,
) -> VlmTrainResult:
    import torch
    from peft import LoraConfig, get_peft_model
    from torch.utils.data import DataLoader, Dataset
    from transformers import AutoModelForImageTextToText, AutoProcessor

    cfg_train = model_def.merged_smoke() if smoke else dict(model_def.training or {})
    torch_device = resolve_device(device)
    profile = get_profile(torch_device)
    profile.apply_env()

    run, run_out, resume_path = begin_training_run(
        model_def,
        smoke=smoke,
        device=str(torch_device),
        profile=profile.name,
        out_dir=out_dir,
        resume=resume,
        trial=trial,
    )
    out = Path(run_out)
    out.mkdir(parents=True, exist_ok=True)

    prepared = Path(dataset_dir) if dataset_dir else model_def.prepared_dir
    train_path = prepared / "train.jsonl"
    if not train_path.is_file():
        finish_run(model_def, run.run_id, "aborted", error=f"missing {train_path}")
        raise FileNotFoundError(
            f"Prepared train split not found: {train_path}. Run `maatml prepare` first."
        )
    rows = list(iter_jsonl(train_path))
    if limit is not None and limit > 0:
        rows = rows[:limit]
    if not rows:
        finish_run(model_def, run.run_id, "aborted", error="empty train split")
        raise ValueError("Empty train split")

    if seed is not None:
        torch.manual_seed(int(seed))
    elif cfg_train.get("seed") is not None:
        torch.manual_seed(int(cfg_train["seed"]))

    model_id = str(
        cfg_train.get("model_id")
        or model_def.base_model
        or "HuggingFaceTB/SmolVLM-256M-Instruct"
    )
    longest_edge = int(cfg_train.get("image_longest_edge") or 384)
    # Do not truncate VLM batches: image placeholder tokens must survive intact.
    # max_input_tokens is recorded for packaging / serve, not applied here.
    user_text = _user_prompt(model_def)
    request_field = get_dataset_cfg(model_def).get("request_field") or "image"
    target_field = get_dataset_cfg(model_def).get("target_field") or "expected_output"

    processor = AutoProcessor.from_pretrained(model_id)
    # Cap image size for memory / speed on CPU smoke runs.
    try:
        if hasattr(processor, "image_processor") and processor.image_processor is not None:
            ip = processor.image_processor
            if hasattr(ip, "size") and isinstance(ip.size, dict):
                ip.size = {**ip.size, "longest_edge": longest_edge}
    except Exception:  # noqa: BLE001
        pass

    dtype = torch.float32
    if str(torch_device).startswith("cuda") or str(torch_device) == "mps":
        dtype = torch.bfloat16 if cfg_train.get("precision") == "bf16" else torch.float16

    load_kwargs: dict[str, Any] = {"dtype": dtype}
    # Prefer local checkpoint when resuming a merged save.
    load_from = model_id
    if resume_path is not None and (Path(resume_path) / "config.json").is_file():
        load_from = str(resume_path)

    model = AutoModelForImageTextToText.from_pretrained(load_from, **load_kwargs)
    model.to(torch_device)

    # Freeze vision tower when present.
    for attr in ("model", "vision_model", "vision_tower"):
        root = getattr(model, attr, None)
        if root is None:
            continue
        vision = getattr(root, "vision_model", None) or getattr(root, "vision_tower", None)
        if vision is not None:
            for p in vision.parameters():
                p.requires_grad = False

    lora_cfg = cfg_train.get("lora") or {}
    if lora_cfg.get("enabled", True):
        target_modules = lora_cfg.get("target_modules") or [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]
        peft_config = LoraConfig(
            r=int(lora_cfg.get("r") or 8),
            lora_alpha=int(lora_cfg.get("alpha") or 16),
            lora_dropout=float(lora_cfg.get("dropout") or 0.05),
            bias="none",
            target_modules=list(target_modules),
            # Restrict to language model modules so vision stays frozen.
            modules_to_save=None,
        )
        model = get_peft_model(model, peft_config)

    pad_id = processor.tokenizer.pad_token_id
    if pad_id is None:
        pad_id = processor.tokenizer.eos_token_id
    image_token_id = getattr(model.config, "image_token_id", None)
    if image_token_id is None:
        image_token_id = getattr(
            getattr(model.config, "text_config", None), "image_token_id", None
        )

    class _VlmDataset(Dataset):
        def __init__(self, data: list[dict]) -> None:
            self.data = data

        def __len__(self) -> int:
            return len(self.data)

        def __getitem__(self, idx: int) -> dict[str, Any]:
            row = self.data[idx]
            rel = row.get(request_field)
            path = model_def.model_dir / rel if not Path(str(rel)).is_absolute() else Path(str(rel))
            image = _resolve_image(path)
            target = row.get(target_field) or {}
            if isinstance(target, dict):
                assistant = target.get("description") or json.dumps(target)
            else:
                assistant = str(target)
            messages = _build_messages(user_text, assistant)
            return {"image": image, "messages": messages, "sample_id": row.get("sample_id")}

    def collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
        images = [b["image"] for b in batch]
        texts = [
            processor.apply_chat_template(
                b["messages"], add_generation_prompt=False, tokenize=False
            )
            for b in batch
        ]
        enc = processor(
            text=texts,
            images=images,
            return_tensors="pt",
            padding=True,
        )
        labels = enc["input_ids"].clone()
        if pad_id is not None:
            labels[labels == pad_id] = -100
        if image_token_id is not None:
            labels[labels == image_token_id] = -100
        enc["labels"] = labels
        return enc

    ds = _VlmDataset(rows)
    batch_size = int(cfg_train.get("batch_size") or 1)
    grad_accum = int(cfg_train.get("grad_accum") or 1)
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=collate,
    )

    lr = float(cfg_train.get("learning_rate") or 1e-4)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr,
        weight_decay=float(cfg_train.get("weight_decay") or 0.0),
    )

    epochs = int(cfg_train.get("epochs") or 1)
    max_steps = int(cfg_train.get("max_steps") or -1)
    log_every = int(cfg_train.get("logging_steps") or 10)

    write_run_metadata(
        out,
        model_def,
        {"train": train_path},
        extra={
            "smoke": smoke,
            "base_model_id": model_id,
            "image_longest_edge": longest_edge,
        },
    )

    model.train()
    t0 = time.perf_counter()
    global_step = 0
    last_loss = 0.0
    metrics: dict[str, float] = {}
    optimizer.zero_grad(set_to_none=True)

    try:
        for epoch in range(epochs):
            for batch in loader:
                batch = {
                    k: v.to(torch_device) if hasattr(v, "to") else v
                    for k, v in batch.items()
                }
                outputs = model(**batch)
                loss = outputs.loss / max(1, grad_accum)
                if not torch.isfinite(loss):
                    raise RuntimeError(
                        f"Non-finite loss={float(loss)} at step {global_step}; aborting"
                    )
                loss.backward()
                if (global_step + 1) % grad_accum == 0:
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                last_loss = float(loss.detach().cpu()) * grad_accum
                global_step += 1
                if log_every > 0 and global_step % log_every == 0:
                    metrics = {
                        "loss": last_loss,
                        "epoch": float(epoch),
                        "step": float(global_step),
                    }
                if max_steps > 0 and global_step >= max_steps:
                    break
            if max_steps > 0 and global_step >= max_steps:
                break
        metrics.setdefault("loss", last_loss)
        metrics["steps"] = float(global_step)

        # Merge LoRA if present, then save HF-format dir (vLLM-loadable).
        to_save = model
        if hasattr(model, "merge_and_unload"):
            to_save = model.merge_and_unload()
        to_save.save_pretrained(out)
        processor.save_pretrained(out)
        # Copy prompt_spec if declared.
        cfg = get_dataset_cfg(model_def)
        if isinstance(cfg.get("prompt_spec"), str):
            src = model_def.resolve(cfg["prompt_spec"])
            if src.is_file():
                import shutil

                shutil.copy2(src, out / "prompt_spec.json")
    except Exception as exc:  # noqa: BLE001
        finish_run(model_def, run.run_id, "aborted", metrics=metrics, error=str(exc))
        raise
    finally:
        runtime = time.perf_counter() - t0

    finish_run(model_def, run.run_id, "completed", metrics=metrics)
    return VlmTrainResult(out_dir=out, metrics=metrics, train_runtime=runtime)
