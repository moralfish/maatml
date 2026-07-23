"""``vision_multitask`` trainer, plain torch loop with maatml run registry."""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from maatml.config import ModelDefinition
from maatml.device import get_profile, resolve_device
from maatml.runs import begin_training_run, finish_run
from maatml.training.guards import write_run_metadata
from maatml.utils.io import iter_jsonl

from .dataset import VisionSceneDataset, collate_vision
from .model import MultitaskConfig, MultitaskNet, load_checkpoint, save_checkpoint


@dataclass
class VisionTrainResult:
    out_dir: Path
    metrics: dict[str, float] = field(default_factory=dict)
    train_runtime: float = 0.0


def train_vision_multitask(
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
) -> VisionTrainResult:
    import torch
    from torch.utils.data import DataLoader

    cfg_train = model_def.merged_smoke() if smoke else dict(model_def.training or {})
    # Allow trial / --set overrides already applied on model_def.training.
    mt_cfg = MultitaskConfig.from_model_def(model_def)
    if smoke:
        # Prefer tiny random backbone for fast smoke when declared.
        if cfg_train.get("pretrained") is False or cfg_train.get("max_steps"):
            mt_cfg.pretrained = bool(cfg_train.get("pretrained", False))

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

    ds = VisionSceneDataset.build(
        rows, model_dir=model_def.model_dir, cfg=mt_cfg, limit=None
    )
    batch_size = int(cfg_train.get("batch_size") or 4)
    workers = 0 if profile.dataloader_workers == 0 else int(
        cfg_train.get("dataloader_workers") or profile.dataloader_workers
    )
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
        collate_fn=collate_vision,
    )

    if resume_path is not None and (Path(resume_path) / "model.safetensors").is_file():
        model, mt_cfg = load_checkpoint(
            Path(resume_path), device=str(torch_device), pretrained_backbone=False
        )
    else:
        model = MultitaskNet.build(mt_cfg)
        model.to(torch_device)

    lr = float(cfg_train.get("learning_rate") or 1e-3)
    weight_decay = float(cfg_train.get("weight_decay") or 0.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    epochs = int(cfg_train.get("epochs") or 1)
    max_steps = int(cfg_train.get("max_steps") or -1)
    log_every = int(cfg_train.get("logging_steps") or 10)

    write_run_metadata(
        out,
        model_def,
        {"train": train_path},
        extra={"smoke": smoke, "image_size": mt_cfg.image_size},
    )

    model.train()
    t0 = time.perf_counter()
    global_step = 0
    last_loss = 0.0
    metrics: dict[str, float] = {}
    status = "completed"
    error: Optional[str] = None

    try:
        for epoch in range(epochs):
            for batch in loader:
                images = batch["image"].to(torch_device)
                targets = {
                    "scene_idx": batch["scene_idx"].to(torch_device),
                    "heatmaps": batch["heatmaps"].to(torch_device),
                    "sizes": batch["sizes"].to(torch_device),
                    "offsets": batch["offsets"].to(torch_device),
                    "center_mask": batch["center_mask"].to(torch_device),
                    "pose_coords": batch["pose_coords"].to(torch_device),
                }
                optimizer.zero_grad(set_to_none=True)
                outputs = model(images)
                losses = model.compute_loss(outputs, targets)
                loss = losses["loss"]
                if not torch.isfinite(loss):
                    raise RuntimeError(
                        f"Non-finite loss={float(loss)} at step {global_step}; aborting"
                    )
                loss.backward()
                optimizer.step()
                last_loss = float(loss.detach().cpu())
                global_step += 1
                if log_every > 0 and global_step % log_every == 0:
                    metrics = {
                        "loss": last_loss,
                        "scene_loss": float(losses["scene_loss"].cpu()),
                        "detect_loss": float(losses["detect_loss"].cpu()),
                        "pose_loss": float(losses["pose_loss"].cpu()),
                        "epoch": float(epoch),
                        "step": float(global_step),
                    }
                if max_steps > 0 and global_step >= max_steps:
                    break
            if max_steps > 0 and global_step >= max_steps:
                break
        metrics.setdefault("loss", last_loss)
        metrics["steps"] = float(global_step)
        save_checkpoint(model, mt_cfg, out)
    except Exception as exc:  # noqa: BLE001
        status = "aborted"
        error = str(exc)
        try:
            save_checkpoint(model, mt_cfg, out)
        except Exception:  # noqa: BLE001
            pass
        finish_run(model_def, run.run_id, status, metrics=metrics, error=error)
        raise
    finally:
        runtime = time.perf_counter() - t0

    finish_run(model_def, run.run_id, "completed", metrics=metrics)
    return VisionTrainResult(out_dir=out, metrics=metrics, train_runtime=runtime)
