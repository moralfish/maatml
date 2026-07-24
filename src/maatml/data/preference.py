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
from ..registry import VALIDATORS, register_format
from ..utils.io import iter_jsonl, write_jsonl_atomic
from .pipeline import prepare_rows


class MintConfigError(ValueError):
    """maatml mint was asked to run without a validator or usable candidates."""

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


def _validator_scorer(model_def: ModelDefinition) -> ValidatorFn:
    """Wrap the model's registered validator as ``(prompt, completion) -> bool``.

    The same gate eval and serve use decides which candidate wins, so a minted
    preference pair means "this completion passes the contract and that one does
    not", not a hand-labelled guess.
    """
    validator_name = (model_def.evaluation or {}).get("validator")
    if not validator_name:
        raise MintConfigError(
            "maatml mint needs evaluation.validator to decide which candidate "
            "passes. Add it to model.yml."
        )
    try:
        validate = VALIDATORS.require(str(validator_name))
    except KeyError as exc:
        raise MintConfigError(
            f"evaluation.validator={validator_name!r} is not registered "
            f"(known: {', '.join(VALIDATORS.names()) or '(none)'})."
        ) from exc
    cfg = get_dataset_cfg(model_def)
    schema_path = (
        model_def.resolve(cfg["schema"]) if isinstance(cfg.get("schema"), str) else None
    )
    contracts_path = (
        model_def.resolve(cfg["contracts"])
        if isinstance(cfg.get("contracts"), str)
        else None
    )

    def _score(prompt: str, completion: str) -> bool:
        try:
            result = validate(
                completion,
                schema_path=schema_path,
                contracts_path=contracts_path,
                user_prompt=prompt,
            )
        except Exception:  # noqa: BLE001  a validator that errors did not pass
            return False
        return bool(getattr(result, "ok", False))

    return _score


def run_mint(
    model_def: ModelDefinition,
    input_path: str | Path,
    *,
    out_path: Optional[str | Path] = None,
    append: bool = True,
) -> dict[str, Any]:
    """Mint preference pairs from candidate completions, gated by the validator.

    Input is JSONL rows of ``{prompt|request, candidates: [completion, ...]}``.
    For each prompt the registered validator splits the candidates into pass /
    fail; a prompt with at least one of each yields one ``{prompt, chosen,
    rejected}`` pair. Pairs append to the preference seed corpus. This is an
    explicit source op (like datagen / ingest), never a default ``run`` step.
    """
    input_path = Path(input_path)
    if not input_path.is_file():
        raise MintConfigError(f"candidate file not found: {input_path}")
    score = _validator_scorer(model_def)
    cfg = get_dataset_cfg(model_def)
    request_field = str(cfg.get("request_field") or cfg.get("raw_field") or "request")

    prompts: list[str] = []
    candidate_lists: list[list[str]] = []
    malformed = 0
    for row in iter_jsonl(input_path):
        prompt = row.get("prompt") or row.get(request_field) or row.get("request")
        cands = row.get("candidates")
        if not isinstance(prompt, str) or not isinstance(cands, list) or len(cands) < 2:
            malformed += 1
            continue
        prompts.append(prompt)
        candidate_lists.append([c if isinstance(c, str) else json.dumps(c) for c in cands])

    if not prompts:
        raise MintConfigError(
            f"no usable rows in {input_path}: each needs a prompt and a "
            "candidates list of at least two completions."
        )

    pairs = mint_preference_pairs(prompts, candidate_lists, score)
    for pair in pairs:
        pair["source"] = "mint"
        pair["family"] = pair["sample_id"]

    seed_rel = out_path or cfg.get("seed_samples") or "datasets/samples/seed_samples.jsonl"
    dest = Path(out_path) if out_path else model_def.resolve(str(seed_rel))
    duplicates = 0
    if pairs:
        existing = list(iter_jsonl(dest)) if (append and dest.is_file()) else []
        seen = {_pair_id(r) for r in existing}
        fresh = [p for p in pairs if _pair_id(p) not in seen]
        duplicates = len(pairs) - len(fresh)
        if fresh:
            dest.parent.mkdir(parents=True, exist_ok=True)
            write_jsonl_atomic(dest, existing + fresh)
        pairs = fresh

    return {
        "prompts": len(prompts),
        "pairs": len(pairs),
        "duplicates": duplicates,
        "malformed": malformed,
        "out_path": str(dest),
    }


def _pair_id(row: dict[str, Any]) -> str:
    if row.get("sample_id"):
        return str(row["sample_id"])
    from ..utils.io import stable_hash

    return stable_hash(row.get("prompt"), row.get("chosen"), row.get("rejected"))[:16]


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
