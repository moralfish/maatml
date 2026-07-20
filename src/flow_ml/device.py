"""Device resolution and per-backend training profiles.

Apple Silicon / MPS has historically needed conservative settings (no mid-train
eval, zero dataloader workers, no grad checkpointing, fp32 master weights).
CUDA can run more aggressively; CPU sits in between.

Torch is imported lazily so CPU-only installs (``pip install flow-ml`` without
the ``[ml]`` extra) can still import this module for CLI plugins / validate.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Union


@dataclass(frozen=True)
class DeviceProfile:
    name: str
    allow_mid_train_eval: bool
    dataloader_workers: int
    allow_grad_checkpointing: bool
    weights_dtype_policy: str  # "fp32_master" | "native"

    def empty_cache(self) -> None:
        """Release allocator caches when the backend supports it."""
        torch = _torch()
        if self.name == "mps" and hasattr(torch, "mps"):
            try:
                torch.mps.empty_cache()
            except Exception:  # noqa: BLE001
                pass
        elif self.name == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()

    def apply_env(self) -> None:
        """Set backend-specific environment knobs (idempotent)."""
        if self.name == "mps":
            os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")


_PROFILES: dict[str, DeviceProfile] = {
    "mps": DeviceProfile(
        name="mps",
        allow_mid_train_eval=False,
        dataloader_workers=0,
        allow_grad_checkpointing=False,
        weights_dtype_policy="fp32_master",
    ),
    "cuda": DeviceProfile(
        name="cuda",
        allow_mid_train_eval=True,
        dataloader_workers=2,
        allow_grad_checkpointing=True,
        weights_dtype_policy="native",
    ),
    "cpu": DeviceProfile(
        name="cpu",
        allow_mid_train_eval=True,
        dataloader_workers=0,
        allow_grad_checkpointing=False,
        weights_dtype_policy="fp32_master",
    ),
}


def _torch() -> Any:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - exercised by wheel smoke
        raise ImportError(
            "torch is required for device resolution; install flow-ml[ml]"
        ) from exc
    return torch


def resolve_device(device: str) -> Any:
    """Map ``auto|mps|cpu|cuda|<torch device str>`` to a ``torch.device``."""
    torch = _torch()
    if device == "cpu":
        return torch.device("cpu")
    if device == "mps":
        return torch.device("mps")
    if device == "cuda":
        return torch.device("cuda")
    if device == "auto":
        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    return torch.device(device)


def get_profile(device: Union[Any, str]) -> DeviceProfile:
    """Return the training profile for a device (or device type / ``auto``)."""
    if isinstance(device, str):
        if device in _PROFILES:
            name = device
        elif device == "auto":
            name = resolve_device("auto").type
        else:
            name = resolve_device(device).type
    else:
        name = device.type

    profile = _PROFILES.get(name, _PROFILES["cpu"])
    profile.apply_env()
    return profile
