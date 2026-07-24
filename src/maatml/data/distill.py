"""``maatml distill``: validator-gated teacher labels over a prompt pool.

Where ``datagen`` invents whole rows, ``distill`` starts from prompts you
already have and asks the teacher only for the label. Every response is gated
by the same validator eval and serve use, so a rejected label never reaches the
seed corpus. Accepted rows carry their provenance (teacher model / revision,
prompt hash, source, family), the rejections are kept, and teacher responses
are recorded so a run replays offline and reproduces the same corpus.

The stage config is typed (:class:`DistillConfig`); there is no new untyped
``dict[str, Any]`` surface.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field

from ..config import ModelDefinition, get_dataset_cfg
from ..registry import VALIDATORS
from ..utils.io import iter_jsonl, sha256_bytes, write_jsonl_atomic
from .ingest import _row_id
from .teacher import TeacherClient


class DistillConfigError(ValueError):
    """distill was asked to run without something it needs (validator, prompts)."""


class DistillConfig(BaseModel):
    """The ``distill:`` section of ``model.yml`` (and ``maatml distill`` flags).

    Prompts come from ``prompt_source`` (a JSONL of rows with the request field,
    or a plain-text file, one prompt per line). ``teacher_model`` /
    ``teacher_revision`` identify the teacher for provenance and for the replay
    cache key, so a different teacher does not silently reuse another's labels.
    """

    model_config = ConfigDict(extra="forbid")

    prompt_source: str
    teacher_model: str = "gpt-4o-mini"
    teacher_revision: str = "unpinned"
    system_prompt: Optional[str] = None
    max_prompts: Optional[int] = Field(default=None, gt=0)
    family: str = "distill"
    cache: str = "output/distill/cache.jsonl"
    output: Optional[str] = None

    @classmethod
    def from_model(cls, model_def: ModelDefinition) -> "DistillConfig":
        section = getattr(model_def, "distill", None) or {}
        if not isinstance(section, dict) or not section:
            raise DistillConfigError(
                "no distill: section in model.yml. Add one (prompt_source is "
                "required) or pass --prompts on the command line."
            )
        try:
            return cls(**section)
        except Exception as exc:  # noqa: BLE001  pydantic error → config error
            raise DistillConfigError(f"invalid distill: section: {exc}") from exc


class TeacherCache:
    """Record/replay store for teacher responses, keyed by prompt + teacher.

    The key binds the prompt hash to the teacher model and revision, so a run
    against a different teacher does not reuse cached labels, and a replay is a
    faithful reproduction of what that teacher said. Stored as JSONL so it is
    diffable and can ship with an example for offline CI.
    """

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self._store: dict[str, str] = {}
        if self.path.is_file():
            for row in iter_jsonl(self.path):
                key = row.get("key")
                response = row.get("response")
                if isinstance(key, str) and isinstance(response, str):
                    self._store[key] = response
        self._dirty = False

    @staticmethod
    def key(prompt_hash: str, model: str, revision: str) -> str:
        return f"{model}@{revision}:{prompt_hash}"

    def get(self, key: str) -> Optional[str]:
        return self._store.get(key)

    def put(self, key: str, response: str) -> None:
        if self._store.get(key) != response:
            self._store[key] = response
            self._dirty = True

    def flush(self) -> None:
        if not self._dirty:
            return
        rows = [{"key": k, "response": v} for k, v in sorted(self._store.items())]
        write_jsonl_atomic(self.path, rows)
        self._dirty = False


def _prompt_hash(prompt: str) -> str:
    return sha256_bytes(prompt.encode("utf-8"))[:16]


def load_prompts(path: Path, request_field: str) -> list[str]:
    """Read a prompt pool from JSONL (request field) or plain text lines."""
    path = Path(path)
    if not path.is_file():
        raise DistillConfigError(f"prompt_source not found: {path}")
    prompts: list[str] = []
    if path.suffix.lower() == ".jsonl":
        for row in iter_jsonl(path):
            value = row.get(request_field) or row.get("prompt") or row.get("request")
            if isinstance(value, str) and value.strip():
                prompts.append(value)
    else:
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                prompts.append(line.rstrip("\n"))
    if not prompts:
        raise DistillConfigError(f"no usable prompts in {path}")
    return prompts


def _build_validate_fn(model_def: ModelDefinition):
    """Same validator gate datagen uses, returning (ok, result) per row."""
    cfg = get_dataset_cfg(model_def)
    validator_name = (model_def.evaluation or {}).get("validator")
    if not validator_name:
        raise DistillConfigError(
            "distill requires evaluation.validator so every teacher label is "
            "gated by the same validator used in eval and serve. Add "
            "evaluation.validator to model.yml."
        )
    try:
        validate = VALIDATORS.require(str(validator_name))
    except KeyError as exc:
        raise DistillConfigError(
            f"evaluation.validator={validator_name!r} is not registered "
            f"(known: {', '.join(VALIDATORS.names()) or '(none)'})."
        ) from exc
    schema_path = (
        model_def.resolve(cfg["schema"]) if isinstance(cfg.get("schema"), str) else None
    )
    contracts_path = (
        model_def.resolve(cfg["contracts"])
        if isinstance(cfg.get("contracts"), str)
        else None
    )

    def _fn(prompt: str, target: Any) -> bool:
        raw = target if isinstance(target, str) else json.dumps(target)
        result = validate(
            raw,
            schema_path=schema_path,
            contracts_path=contracts_path,
            user_prompt=prompt,
        )
        return bool(result.ok)

    return _fn


def run_distill(
    model_def: ModelDefinition,
    *,
    config: Optional[DistillConfig] = None,
    prompt_source: Optional[str] = None,
    replay: bool = False,
    offline: bool = False,
    limit: Optional[int] = None,
    append: bool = True,
    out_path: Optional[str] = None,
) -> dict[str, Any]:
    """Distill validator-gated teacher labels over the prompt pool.

    ``replay`` uses only cached teacher responses (a cache miss is skipped, not
    fetched); ``offline`` forbids any network call and is implied by ``replay``.
    Returns a summary dict; accepted rows are appended to the seed corpus.
    """
    if config is not None:
        cfg = config
    elif getattr(model_def, "distill", None):
        cfg = DistillConfig.from_model(model_def)
    elif prompt_source:
        # A prompt pool on the command line is enough; the rest defaults.
        cfg = DistillConfig(prompt_source=prompt_source)
    else:
        raise DistillConfigError(
            "no distill: section in model.yml and no --prompts given. "
            "Point distill at a prompt pool one way or the other."
        )
    if prompt_source:
        cfg = cfg.model_copy(update={"prompt_source": prompt_source})
    ds_cfg = get_dataset_cfg(model_def)
    request_field = str(ds_cfg.get("request_field") or ds_cfg.get("raw_field") or "request")
    target_field = str(ds_cfg.get("target_field") or "target")

    validate_fn = _build_validate_fn(model_def)
    prompts = load_prompts(model_def.resolve(cfg.prompt_source), request_field)
    cap = limit if limit is not None else cfg.max_prompts
    if cap is not None:
        prompts = prompts[:cap]

    cache = TeacherCache(model_def.resolve(cfg.cache))
    offline = offline or replay
    teacher: Optional[TeacherClient] = None
    system = cfg.system_prompt or (
        f"You label training samples for the MaatML model {model_def.identity}. "
        f"Given a user request, return only a JSON object for the "
        f"'{target_field}' field. No markdown fences, no commentary."
    )

    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    stats = {"cache_hits": 0, "teacher_calls": 0, "teacher_failures": 0, "cache_misses": 0}

    for index, prompt in enumerate(prompts):
        phash = _prompt_hash(prompt)
        key = TeacherCache.key(phash, cfg.teacher_model, cfg.teacher_revision)
        response = cache.get(key)
        if response is not None:
            stats["cache_hits"] += 1
        elif offline:
            # Replay / offline never reaches the network: an uncached prompt is
            # a skip, so a replay reproduces exactly what was recorded.
            stats["cache_misses"] += 1
            rejected.append({"prompt_sha256": phash, "_reject_reason": "cache_miss"})
            continue
        else:
            if teacher is None:
                teacher = TeacherClient(model=cfg.teacher_model)
            try:
                response = teacher.chat_completions(
                    [
                        {"role": "system", "content": system},
                        {"role": "user", "content": prompt},
                    ]
                )
            except Exception as exc:  # noqa: BLE001  a teacher failure is a rejected row
                stats["teacher_failures"] += 1
                rejected.append(
                    {"prompt_sha256": phash, "_reject_reason": f"teacher_error: {exc}"}
                )
                continue
            stats["teacher_calls"] += 1
            cache.put(key, response)

        target = _parse_target(response)
        if target is None or not validate_fn(prompt, target):
            rejected.append(
                {
                    "prompt_sha256": phash,
                    request_field: prompt[:500],
                    "teacher_response": response[:1500],
                    "_reject_reason": "invalid_target" if target is not None else "unparseable",
                }
            )
            continue

        row = {
            "sample_id": f"distill-{cfg.teacher_model}-{phash}",
            "source": "distill",
            "family": f"{cfg.family}:{phash}",
            request_field: prompt,
            target_field: target,
            "provenance": {
                "teacher_model": cfg.teacher_model,
                "teacher_revision": cfg.teacher_revision,
                "prompt_sha256": phash,
            },
        }
        accepted.append(row)

    if not (replay or offline):
        cache.flush()

    seed_target = out_path or cfg.output
    dest = (
        model_def.resolve(seed_target)
        if seed_target
        else model_def.resolve(str(ds_cfg.get("seed_samples") or "datasets/samples/seed_samples.jsonl"))
    )
    duplicates = 0
    seed_written = False
    if accepted:
        existing = list(iter_jsonl(dest)) if (append and dest.is_file()) else []
        seen = {_row_id(r) for r in existing}
        fresh = [r for r in accepted if _row_id(r) not in seen]
        duplicates = len(accepted) - len(fresh)
        if fresh:
            dest.parent.mkdir(parents=True, exist_ok=True)
            write_jsonl_atomic(dest, existing + fresh)
            seed_written = True
        accepted = fresh

    reject_path = dest.with_name(dest.stem + ".distill_rejected.jsonl")
    if rejected:
        write_jsonl_atomic(reject_path, rejected)
    else:
        reject_path.unlink(missing_ok=True)

    summary = {
        "prompts": len(prompts),
        "accepted": len(accepted),
        "rejected": len(rejected),
        "duplicates": duplicates,
        "seed_written": seed_written,
        "replay": bool(replay or offline),
        "teacher_model": cfg.teacher_model,
        "teacher_revision": cfg.teacher_revision,
        "out_path": str(dest),
        "cache_path": str(cache.path),
        **stats,
    }
    _write_distill_card(dest, summary)
    return summary


def _parse_target(response: str) -> Optional[Any]:
    text = response.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines)
    try:
        return json.loads(text)
    except (json.JSONDecodeError, ValueError):
        return None


def _write_distill_card(dest: Path, summary: dict[str, Any]) -> Path:
    lines = [
        f"# distill card: {dest.name}",
        "",
        f"- teacher: {summary['teacher_model']}@{summary['teacher_revision']}",
        f"- prompts: {summary['prompts']}",
        f"- accepted: {summary['accepted']}",
        f"- rejected: {summary['rejected']}",
        f"- duplicates_skipped: {summary['duplicates']}",
        f"- replay: {summary['replay']}",
        f"- cache: hits={summary['cache_hits']} calls={summary['teacher_calls']} "
        f"misses={summary['cache_misses']} failures={summary['teacher_failures']}",
        "",
        "Every accepted row was gated by evaluation.validator before entering "
        "the corpus, and carries teacher provenance. Teacher responses are "
        "recorded in the cache above; re-run with --replay to reproduce this "
        "corpus offline.",
    ]
    card = dest.with_name(dest.stem + ".distill_card.md")
    card.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return card
