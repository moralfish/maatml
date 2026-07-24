"""Preference-pair helpers and ``preference_jsonl`` dataset format.

Rows are ``{prompt, chosen, rejected}`` (+ optional ``sample_id`` / ``family``).
``mint_preference_pairs`` builds pairs from validator outcomes without a model.
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Any, Callable, Optional, Sequence, Union

from ..config import ModelDefinition, get_dataset_cfg
from ..registry import register_format
from ..utils.io import iter_jsonl
from .pipeline import prepare_rows

ValidatorFn = Callable[[str, str], bool]
CandidatesFn = Callable[[str], Sequence[str]]


def as_completion_text(value: Any) -> str:
    """Serialise a preference field to the text the trainer will see.

    Strings pass through; structured values become compact JSON. ``str()`` on
    a dict yields its Python repr (``{'a': 1}``, single quotes, ``None``),
    which is not the JSON the model is supposed to learn.
    """
    if isinstance(value, str):
        return value
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    return str(value)


def normalize_preference(row: dict) -> dict[str, Any]:
    """Normalize a preference JSONL row to ``prompt`` / ``chosen`` / ``rejected``."""
    prompt = row.get("prompt")
    if prompt is None:
        prompt = row.get("instruction") or row.get("query") or row.get("request")
    chosen = row.get("chosen")
    rejected = row.get("rejected")
    if prompt is None or chosen is None or rejected is None:
        raise ValueError(
            "preference row requires prompt, chosen, and rejected "
            f"(keys present: {sorted(row)})"
        )
    out: dict[str, Any] = {
        "prompt": as_completion_text(prompt),
        "chosen": as_completion_text(chosen),
        "rejected": as_completion_text(rejected),
    }
    if out["chosen"] == out["rejected"]:
        warnings.warn(
            "preference row has identical chosen and rejected completions "
            f"(sample_id={row.get('sample_id') or row.get('id') or '?'}); "
            "it carries no preference signal",
            stacklevel=2,
        )
    for key in ("sample_id", "family", "source", "category"):
        if key in row and row[key] is not None:
            out[key] = row[key]
    if "sample_id" not in out and row.get("id") is not None:
        out["sample_id"] = row["id"]
    return out


def mint_preference_pairs(
    prompts: Sequence[str],
    candidates: Union[Sequence[Sequence[str]], CandidatesFn],
    validator: ValidatorFn,
    *,
    sample_id_prefix: str = "pref",
) -> list[dict[str, Any]]:
    """Build preference pairs where one completion passes and one fails.

    For each prompt, evaluate candidates with ``validator(prompt, completion)``.
    When at least one pass and one fail exist, emit
    ``{prompt, chosen, rejected, sample_id}`` using the first of each.
    Prompts with only passes or only failures are skipped.
    """
    pairs: list[dict[str, Any]] = []
    for i, prompt in enumerate(prompts):
        if callable(candidates) and not isinstance(candidates, (list, tuple)):
            cands = list(candidates(prompt))
        else:
            cands = list(candidates[i])  # type: ignore[index]
        passed: list[str] = []
        failed: list[str] = []
        for completion in cands:
            if validator(prompt, completion):
                passed.append(completion)
            else:
                failed.append(completion)
        if not passed or not failed:
            continue
        pairs.append(
            {
                "prompt": prompt,
                "chosen": passed[0],
                "rejected": failed[0],
                "sample_id": f"{sample_id_prefix}-{i:04d}",
            }
        )
    return pairs


@register_format("preference_jsonl")
def prepare_preference_jsonl(
    model_def: ModelDefinition, out_dir: Optional[Path] = None
) -> dict:
    """Prepare preference JSONL into train/val/test splits."""
    cfg = get_dataset_cfg(model_def)
    if "seed_samples" not in cfg:
        raise ValueError("model.yml `dataset:`/`data:` must declare `seed_samples`")
    if cfg.get("sanitize"):
        raise ValueError(
            "dataset.sanitize is set but the preference format does not sanitize "
            "(rows carry prompt/chosen/rejected, not a flat request field). "
            "Remove dataset.sanitize or use the jsonl_seed format."
        )
    seed_path = model_def.resolve(cfg["seed_samples"])
    rows = [normalize_preference(raw) for raw in iter_jsonl(seed_path)]
    bench_rows: list[dict] = []
    benchmark_path = cfg.get("benchmark_samples")
    if benchmark_path:
        for raw in iter_jsonl(model_def.resolve(benchmark_path)):
            bench_rows.append(normalize_preference(raw))
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
