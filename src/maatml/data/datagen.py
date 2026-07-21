"""Datagen orchestration: registered generators + optional teacher."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Optional

from ..config import ModelDefinition, get_dataset_cfg
from ..registry import GENERATORS, VALIDATORS
from ..utils.io import iter_jsonl, write_jsonl
from .gated import build_gated_corpus
from .teacher import TeacherClient


def _default_validate_fn(
    model_def: ModelDefinition,
) -> Callable[[dict[str, Any]], bool]:
    cfg = get_dataset_cfg(model_def)
    validator_name = (model_def.evaluation or {}).get("validator")
    if not validator_name:
        return lambda _row: True
    validate = VALIDATORS.require(str(validator_name))
    schema_path = (
        model_def.resolve(cfg["schema"]) if isinstance(cfg.get("schema"), str) else None
    )
    contracts_path = (
        model_def.resolve(cfg["contracts"])
        if isinstance(cfg.get("contracts"), str)
        else None
    )
    target_field = str(cfg.get("target_field") or "target")
    request_field = str(cfg.get("request_field") or cfg.get("raw_field") or "request")

    def _fn(row: dict[str, Any]) -> bool:
        gold = row.get(target_field)
        if gold is None:
            return False
        raw = gold if isinstance(gold, str) else json.dumps(gold)
        result = validate(
            raw,
            schema_path=schema_path,
            contracts_path=contracts_path,
            user_prompt=row.get(request_field),
        )
        return bool(result.ok)

    return _fn


def _teacher_generate_fn(
    model_def: ModelDefinition,
    teacher: TeacherClient,
    *,
    seed: int,
) -> Callable[[], Optional[dict[str, Any]]]:
    cfg = get_dataset_cfg(model_def)
    target_field = str(cfg.get("target_field") or "target")
    request_field = str(cfg.get("request_field") or cfg.get("raw_field") or "request")
    counter = {"n": 0}

    system = (
        f"You generate training samples for the MaatML model {model_def.identity}. "
        f"Return a single JSON object with keys '{request_field}' and '{target_field}'. "
        "No markdown fences."
    )

    def _fn() -> Optional[dict[str, Any]]:
        counter["n"] += 1
        user = (
            f"Propose sample #{counter['n']} (seed hint={seed}). "
            f"Task={model_def.task or model_def.architecture}."
        )
        try:
            row = teacher.propose_json_row(system, user)
        except Exception:  # noqa: BLE001
            return None
        if not isinstance(row, dict):
            return None
        row.setdefault("sample_id", f"teacher-{seed}-{counter['n']}")
        row.setdefault("source", "teacher")
        return row

    return _fn


def run_datagen(
    model_def: ModelDefinition,
    *,
    target: int = 100,
    seed: int = 0,
    out_path: Optional[str | Path] = None,
    use_teacher: bool = False,
    max_attempts: Optional[int] = None,
    append: bool = True,
) -> dict[str, Any]:
    """Generate validator-gated seed rows via a registered generator or teacher."""
    cfg = get_dataset_cfg(model_def)
    seed_rel = cfg.get("seed_samples") or "datasets/samples/seed_samples.jsonl"
    dest = Path(out_path) if out_path else model_def.resolve(str(seed_rel))
    dest.parent.mkdir(parents=True, exist_ok=True)

    validate_fn = _default_validate_fn(model_def)

    if use_teacher:
        teacher = TeacherClient()
        generate_fn = _teacher_generate_fn(model_def, teacher, seed=seed)
        gen_name = "teacher"
    else:
        gen_name = cfg.get("generator")
        if not gen_name:
            raise KeyError(
                "No dataset.generator in model.yml and --teacher not set. "
                "Set dataset.generator to a registered name (e.g. jcl, spool) "
                "or pass --teacher. Register custom generators with "
                "@register_generator."
            )
        factory = GENERATORS.require(str(gen_name))
        generate_fn = factory(model_def, seed=seed)

    accepted, rejected = build_gated_corpus(
        generate_fn=generate_fn,
        validate_fn=validate_fn,
        target_n=target,
        max_attempts=max_attempts,
    )

    existing: list[dict] = []
    if append and dest.is_file():
        existing = list(iter_jsonl(dest))
    write_jsonl(dest, existing + accepted)

    reject_path = dest.with_name(dest.stem + ".datagen_rejected.jsonl")
    if rejected:
        write_jsonl(reject_path, rejected)

    return {
        "generator": gen_name,
        "accepted": len(accepted),
        "rejected": len(rejected),
        "out_path": str(dest),
        "reject_path": str(reject_path) if rejected else None,
    }
