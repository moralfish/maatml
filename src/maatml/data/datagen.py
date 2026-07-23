"""Datagen orchestration: registered generators + optional teacher."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Optional

from ..config import ModelDefinition, get_dataset_cfg
from ..registry import GENERATORS, VALIDATORS
from ..utils.io import iter_jsonl, write_jsonl_atomic
from .gated import build_gated_corpus
from .teacher import TeacherClient


class DatagenConfigError(ValueError):
    """Raised when datagen is asked to run without a usable validator.

    Mirrors GateConfigError: datagen must not silently accept every row. A
    missing evaluation.validator is a configuration error unless the caller
    explicitly opts into an ungated run with allow_ungated.
    """


def _default_validate_fn(
    model_def: ModelDefinition,
    *,
    allow_ungated: bool = False,
) -> Callable[[dict[str, Any]], bool]:
    cfg = get_dataset_cfg(model_def)
    validator_name = (model_def.evaluation or {}).get("validator")
    if not validator_name:
        if allow_ungated:
            return lambda _row: True
        raise DatagenConfigError(
            "datagen requires evaluation.validator so every seed row is gated by "
            "the same validator used in eval and serve. Add evaluation.validator "
            "to model.yml, or pass --allow-ungated to accept every generated row "
            "(the run and its dataset card are then marked UNGATED)."
        )
    try:
        validate = VALIDATORS.require(str(validator_name))
    except KeyError as exc:
        raise DatagenConfigError(
            f"evaluation.validator={validator_name!r} is not a registered "
            f"validator (known: {', '.join(VALIDATORS.names()) or '(none)'}). "
            "Register it via @register_validator or a model.yml plugins: entry. "
            "--allow-ungated does not bypass a misconfigured validator."
        ) from exc
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
    allow_ungated: bool = False,
) -> dict[str, Any]:
    """Generate validator-gated seed rows via a registered generator or teacher."""
    cfg = get_dataset_cfg(model_def)
    seed_rel = cfg.get("seed_samples") or "datasets/samples/seed_samples.jsonl"
    dest = Path(out_path) if out_path else model_def.resolve(str(seed_rel))
    dest.parent.mkdir(parents=True, exist_ok=True)

    validator_name = (model_def.evaluation or {}).get("validator")
    gated = bool(validator_name)
    validate_fn = _default_validate_fn(model_def, allow_ungated=allow_ungated)

    if use_teacher:
        teacher = TeacherClient()
        generate_fn = _teacher_generate_fn(model_def, teacher, seed=seed)
        gen_name = "teacher"
    else:
        raw_gen = cfg.get("generator")
        if not raw_gen:
            raise KeyError(
                "No dataset.generator in model.yml and --teacher not set. "
                "Set dataset.generator to a registered name (e.g. jcl, spool) "
                "or pass --teacher. Register custom generators with "
                "@register_generator."
            )
        gen_name = str(raw_gen)
        factory = GENERATORS.require(gen_name)
        generate_fn = factory(model_def, seed=seed)

    accepted, rejected = build_gated_corpus(
        generate_fn=generate_fn,
        validate_fn=validate_fn,
        target_n=target,
        max_attempts=max_attempts,
    )

    reject_path = dest.with_name(dest.stem + ".datagen_rejected.jsonl")
    if rejected:
        write_jsonl_atomic(reject_path, rejected)

    # D2: never truncate or rewrite a non-empty seed file when nothing was
    # accepted. A validator that rejected everything (or an empty run) must not
    # destroy a hand-curated corpus.
    seed_existing_nonempty = dest.is_file() and dest.stat().st_size > 0
    if not accepted:
        seed_written = False
        protected_existing = seed_existing_nonempty
    else:
        existing = list(iter_jsonl(dest)) if (append and dest.is_file()) else []
        write_jsonl_atomic(dest, existing + accepted)
        seed_written = True
        protected_existing = False

    card_path = _write_datagen_card(
        dest,
        generator=gen_name,
        validator=str(validator_name) if validator_name else None,
        gated=gated,
        accepted=len(accepted),
        rejected=len(rejected),
        seed_written=seed_written,
        protected_existing=protected_existing,
        append=append,
    )

    return {
        "generator": gen_name,
        "validator": str(validator_name) if validator_name else None,
        "gated": gated,
        "accepted": len(accepted),
        "rejected": len(rejected),
        "seed_written": seed_written,
        "protected_existing": protected_existing,
        "out_path": str(dest),
        "reject_path": str(reject_path) if rejected else None,
        "card_path": str(card_path),
    }


def _write_datagen_card(
    dest: Path,
    *,
    generator: str,
    validator: Optional[str],
    gated: bool,
    accepted: int,
    rejected: int,
    seed_written: bool,
    protected_existing: bool,
    append: bool,
) -> Path:
    """Write a provenance card next to the seed file recording gate status."""
    status = "GATED" if gated else "UNGATED"
    lines = [
        f"# datagen card: {dest.name}",
        "",
        f"- status: {status}",
        f"- generator: {generator}",
        f"- validator: {validator or 'none (UNGATED)'}",
        f"- accepted: {accepted}",
        f"- rejected: {rejected}",
        f"- append: {append}",
        f"- seed_written: {seed_written}",
        f"- protected_existing_seed: {protected_existing}",
    ]
    if not gated:
        lines += [
            "",
            "This corpus was generated WITHOUT a validator (--allow-ungated). "
            "Rows are not validator-gated; review before training.",
        ]
    card = dest.with_name(dest.stem + ".datagen_card.md")
    card.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return card
