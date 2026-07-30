"""Step-count and precision helpers shared by every trainer.

Both computations used to be copy-pasted into each of the four trainer modules,
and both had already drifted: the mixed-precision device guard existed in two
copies and not the other two, and every copy sized the LR schedule as though the
run were single-process.
"""
from __future__ import annotations

import os
from typing import Any, Optional


def world_size() -> int:
    """Number of training processes, from the launcher's environment.

    ``torchrun`` / ``accelerate`` set ``WORLD_SIZE``. Falls back to the
    initialised process group, then to 1.
    """
    raw = os.environ.get("WORLD_SIZE")
    if raw is not None:
        try:
            parsed = int(raw)
        except ValueError:
            parsed = 0
        if parsed > 0:
            return parsed
    try:
        import torch.distributed as dist

        if dist.is_available() and dist.is_initialized():
            return int(dist.get_world_size())
    except ImportError:
        pass
    return 1


def total_training_steps(
    n_rows: int,
    *,
    batch_size: int,
    grad_accum: int,
    epochs: float,
    max_steps: int = -1,
    processes: Optional[int] = None,
) -> int:
    """Optimiser steps a run will take.

    Each of ``processes`` ranks sees ``n_rows / processes`` rows, so a run that
    ignores world size overstates the step count by that factor. The LR schedule
    is derived from this number, so on 8 GPUs a 6% warmup ratio silently became
    roughly 48% of the real schedule.
    """
    if max_steps is not None and max_steps >= 0:
        return int(max_steps)
    ranks = processes if processes is not None else world_size()
    ranks = max(1, int(ranks))
    per_rank_rows = n_rows / ranks
    steps = per_rank_rows / max(1, batch_size) / max(1, grad_accum) * epochs
    return max(0, int(steps))


def warmup_steps(total_steps: int, warmup_ratio: float) -> int:
    """Warmup steps for a schedule of ``total_steps``."""
    return max(0, int(round(total_steps * warmup_ratio)))


def precision_flags(
    precision: str,
    *,
    device: Any = None,
    distributed: bool = False,
) -> tuple[bool, bool]:
    """Return ``(use_bf16, use_fp16)`` for ``TrainingArguments``.

    Mixed precision is only requested where the device supports it. transformers
    rejects ``bf16=True`` on an unsupported CPU, so an unguarded flag turns a
    config that trains fine for one architecture into a hard failure for
    another on the same host.
    """
    device_type = _device_type(device)
    supported = bool(distributed) or device_type in ("cuda", "mps")
    return (precision == "bf16" and supported, precision == "fp16" and supported)


def _device_type(device: Any) -> Optional[str]:
    """Device family for ``device``, which may be a torch device or a string.

    Trainers hold the device in both forms ("cuda:0" in one scope, a
    torch.device in another), so accept either rather than making each caller
    convert.
    """
    if device is None:
        return None
    device_type = getattr(device, "type", None)
    if isinstance(device_type, str) and device_type:
        return device_type
    text = str(device).strip().lower()
    if not text:
        return None
    return text.split(":", 1)[0]
