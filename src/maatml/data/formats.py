"""Dataset format adapters (alpaca / sharegpt → canonical messages rows).

Registered via ``@register_format`` and discovered by ``discover_plugins``.
Normalized rows carry ``messages: [{role, content}, ...]`` plus optional
``sample_id`` / ``family`` / ``source`` for group-aware splits.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from ..config import ModelDefinition, get_dataset_cfg
from ..registry import register_format
from ..utils.io import iter_jsonl
from .pipeline import prepare_rows


_ROLE_MAP = {
    "human": "user",
    "user": "user",
    "gpt": "assistant",
    "assistant": "assistant",
    "system": "system",
    "bing": "assistant",
    "chatgpt": "assistant",
}


def _preserve_meta(row: dict) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key in ("sample_id", "family", "source", "category"):
        if key in row and row[key] is not None:
            out[key] = row[key]
    return out


def normalize_alpaca(row: dict) -> dict:
    """Map Alpaca ``instruction``/``input``/``output`` → messages."""
    messages: list[dict[str, str]] = []
    system = row.get("system")
    if isinstance(system, str) and system.strip():
        messages.append({"role": "system", "content": system})

    instruction = str(row.get("instruction") or "")
    inp = row.get("input")
    if isinstance(inp, str) and inp.strip():
        user = f"{instruction}\n\n{inp}" if instruction else inp
    else:
        user = instruction
    messages.append({"role": "user", "content": user})
    messages.append({"role": "assistant", "content": str(row.get("output") or "")})

    out = _preserve_meta(row)
    out["messages"] = messages
    if "sample_id" not in out:
        out["sample_id"] = row.get("id") or row.get("sample_id")
    return out


def normalize_sharegpt(row: dict) -> dict:
    """Map ShareGPT ``conversations`` (from/value) → messages."""
    conversations = row.get("conversations") or row.get("conversation") or []
    messages: list[dict[str, str]] = []
    if isinstance(conversations, list):
        for turn in conversations:
            if not isinstance(turn, dict):
                continue
            raw_role = str(turn.get("from") or turn.get("role") or "").lower()
            role = _ROLE_MAP.get(raw_role)
            if role is None:
                continue
            content = turn.get("value", turn.get("content", ""))
            messages.append({"role": role, "content": str(content)})

    out = _preserve_meta(row)
    out["messages"] = messages
    if "sample_id" not in out:
        out["sample_id"] = row.get("id") or row.get("sample_id")
    return out


def _prepare_normalized(
    model_def: ModelDefinition,
    normalize_fn,
    *,
    out_dir: Optional[Path] = None,
) -> dict:
    cfg = get_dataset_cfg(model_def)
    if "seed_samples" not in cfg:
        raise ValueError("model.yml `dataset:`/`data:` must declare `seed_samples`")
    if cfg.get("sanitize"):
        raise ValueError(
            "dataset.sanitize is set but the alpaca/sharegpt format does not "
            "sanitize (rows carry structured messages, not a flat request field). "
            "Remove dataset.sanitize or use the jsonl_seed format."
        )
    seed_path = model_def.resolve(cfg["seed_samples"])
    rows = [normalize_fn(raw) for raw in iter_jsonl(seed_path)]
    bench_rows: list[dict] = []
    benchmark_path = cfg.get("benchmark_samples")
    if benchmark_path:
        for raw in iter_jsonl(model_def.resolve(benchmark_path)):
            bench_rows.append(normalize_fn(raw))
    return prepare_rows(
        model_def,
        rows,
        out_dir=out_dir,
        seed_label=str(seed_path),
        benchmark_rows=bench_rows or None,
        benchmark_label=(
            str(model_def.resolve(benchmark_path)) if benchmark_path else None
        ),
    )


@register_format("alpaca")
def prepare_alpaca(
    model_def: ModelDefinition, out_dir: Optional[Path] = None
) -> dict:
    """Prepare Alpaca-format JSONL into train/val/test splits."""
    return _prepare_normalized(model_def, normalize_alpaca, out_dir=out_dir)


@register_format("sharegpt")
def prepare_sharegpt(
    model_def: ModelDefinition, out_dir: Optional[Path] = None
) -> dict:
    """Prepare ShareGPT-format JSONL into train/val/test splits."""
    return _prepare_normalized(model_def, normalize_sharegpt, out_dir=out_dir)
