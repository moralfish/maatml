"""Append-only run registry (`output/runs.jsonl`) per model folder.

Each training run gets a unique ``run_id`` and checkpoint directory under
``output/checkpoints/<run_id>/``. The registry records status, device profile,
metrics, and optional eval-gate results.
"""
from __future__ import annotations

import json
import secrets
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, ValidationError

from .config import ModelDefinition
from .device import is_main_process
from .utils.io import write_json

RunStatus = Literal["running", "completed", "aborted"]

_RUNS_FILENAME = "runs.jsonl"


class RunRecord(BaseModel):
    """One training (or evaluated) run entry."""

    model_config = ConfigDict(extra="forbid")

    run_id: str
    identity: str
    architecture: str
    status: RunStatus
    started_at: str
    finished_at: Optional[str] = None
    smoke: bool = False
    device: Optional[str] = None
    profile: Optional[str] = None
    out_dir: str
    spec_hash: Optional[str] = None
    metrics: Optional[dict[str, float]] = None
    error: Optional[str] = None
    gates: Optional[dict[str, Any]] = None
    # Optional HPO / sweep trial metadata (rank-0 writes only).
    trial: Optional[dict[str, Any]] = None


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def make_run_id(*, when: Optional[datetime] = None) -> str:
    """``YYYYMMDD-HHMMSS-<short>`` unique-enough run id."""
    ts = when or datetime.now(timezone.utc)
    short = secrets.token_hex(3)
    return f"{ts.strftime('%Y%m%d-%H%M%S')}-{short}"


def runs_path(model_def: ModelDefinition) -> Path:
    return model_def.output_dir / _RUNS_FILENAME


def _append_record(model_def: ModelDefinition, record: RunRecord) -> None:
    """Append to ``runs.jsonl`` on the main process only (multi-GPU safe)."""
    if not is_main_process():
        return
    path = runs_path(model_def)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Build the full line first and write once, so a crash or a concurrent
    # append cannot leave a torn record (body with no newline).
    line = json.dumps(record.model_dump(mode="json"), sort_keys=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def _quarantine_corrupt(path: Path, lines: list[str]) -> None:
    """Append never-before-seen unparseable lines to a sidecar corrupt file.

    Dedup-guarded because ``list_runs`` runs on every read: it must not append
    the same bad line repeatedly, and it must never rewrite ``runs.jsonl``.
    """
    corrupt_path = path.with_name(path.name + ".corrupt")
    try:
        seen: set[str] = set()
        if corrupt_path.is_file():
            with open(corrupt_path, "r", encoding="utf-8") as f:
                seen = {ln.strip() for ln in f}
        new = [ln for ln in lines if ln not in seen]
        if not new:
            return
        with open(corrupt_path, "a", encoding="utf-8") as f:
            for ln in new:
                f.write(ln + "\n")
    except OSError:
        pass


def list_runs(model_def: ModelDefinition) -> list[RunRecord]:
    """Return all run records (latest entry per ``run_id`` wins).

    A line that cannot be parsed (a torn record from a crash mid-write, or
    manual corruption) is skipped with a warning and recorded in
    ``runs.jsonl.corrupt`` rather than raising and bricking every consumer.
    """
    path = runs_path(model_def)
    if not path.is_file():
        return []
    latest: dict[str, RunRecord] = {}
    order: list[str] = []
    corrupt: list[str] = []
    with open(path, "r", encoding="utf-8") as f:
        for lineno, raw in enumerate(f, start=1):
            line = raw.strip()
            if not line:
                continue
            try:
                rec = RunRecord.model_validate_json(line)
            except (ValidationError, ValueError):
                corrupt.append(line)
                warnings.warn(
                    f"skipping unparseable run record at {path}:{lineno}",
                    RuntimeWarning,
                    stacklevel=2,
                )
                continue
            if rec.run_id not in latest:
                order.append(rec.run_id)
            latest[rec.run_id] = rec
    if corrupt:
        _quarantine_corrupt(path, corrupt)
    return [latest[rid] for rid in order]


def get_run(model_def: ModelDefinition, run_id: str) -> Optional[RunRecord]:
    for rec in list_runs(model_def):
        if rec.run_id == run_id:
            return rec
    return None


def start_run(
    model_def: ModelDefinition,
    *,
    smoke: bool = False,
    device: Optional[str] = None,
    profile: Optional[str] = None,
    spec_hash: Optional[str] = None,
    run_id: Optional[str] = None,
    out_dir: Optional[Path] = None,
    trial: Optional[dict[str, Any]] = None,
) -> RunRecord:
    """Create a new ``running`` run and append it to ``runs.jsonl``."""
    rid = run_id or make_run_id()
    ckpt = Path(out_dir) if out_dir else (model_def.checkpoints_dir / rid)
    ckpt.mkdir(parents=True, exist_ok=True)
    record = RunRecord(
        run_id=rid,
        identity=model_def.identity,
        architecture=model_def.architecture,
        status="running",
        started_at=_utc_now(),
        smoke=smoke,
        device=device,
        profile=profile,
        out_dir=str(ckpt.resolve()),
        spec_hash=spec_hash,
        trial=trial,
    )
    _append_record(model_def, record)
    return record


def finish_run(
    model_def: ModelDefinition,
    run_id: str,
    status: RunStatus,
    *,
    metrics: Optional[dict[str, float]] = None,
    error: Optional[str] = None,
    gates: Optional[dict[str, Any]] = None,
) -> Optional[RunRecord]:
    """Mark a run completed/aborted by appending an updated record.

    On non-main ranks (multi-GPU), returns ``None`` without writing.
    """
    if status == "running":
        raise ValueError("finish_run status must be 'completed' or 'aborted'")
    if not is_main_process():
        return None
    rec = get_run(model_def, run_id)
    if rec is None:
        raise KeyError(f"Unknown run_id {run_id!r} for {model_def.identity}")
    payload = rec.model_dump()
    payload["status"] = status
    payload["finished_at"] = _utc_now()
    if metrics is not None:
        payload["metrics"] = metrics
    if error is not None:
        payload["error"] = error
    if gates is not None:
        payload["gates"] = gates
    updated = RunRecord(**payload)
    _append_record(model_def, updated)
    return updated


def update_run_gates(
    model_def: ModelDefinition,
    run_id: str,
    gates: dict[str, Any],
    *,
    metrics: Optional[dict[str, float]] = None,
) -> Optional[RunRecord]:
    """Attach eval-gate results to a known run (no-op if run_id unknown)."""
    rec = get_run(model_def, run_id)
    if rec is None:
        return None
    payload = rec.model_dump()
    payload["gates"] = gates
    if metrics is not None:
        payload["metrics"] = {**(payload.get("metrics") or {}), **metrics}
    updated = RunRecord(**payload)
    _append_record(model_def, updated)
    return updated


def latest_completed_run(model_def: ModelDefinition) -> Optional[RunRecord]:
    """Most recently finished completed run (by ``finished_at``, then order)."""
    completed = [r for r in list_runs(model_def) if r.status == "completed"]
    if not completed:
        return None

    def _key(r: RunRecord) -> str:
        return r.finished_at or r.started_at

    return max(completed, key=_key)


def _has_hf_checkpoints(path: Path) -> bool:
    if not path.is_dir():
        return False
    return any(p.is_dir() and p.name.startswith("checkpoint-") for p in path.iterdir())


def latest_incomplete_run(model_def: ModelDefinition) -> Optional[RunRecord]:
    """Latest ``running`` run, else latest run dir that still has HF checkpoints."""
    runs = list_runs(model_def)
    running = [r for r in runs if r.status == "running"]
    if running:
        return running[-1]
    # Prefer runs that look resumable (HF Trainer mid-checkpoints present).
    for rec in reversed(runs):
        if _has_hf_checkpoints(Path(rec.out_dir)):
            return rec
    return None


def _last_trainer_checkpoint(root: Path) -> Optional[Path]:
    """Newest ``checkpoint-*`` dir under a run root (HF Trainer resume target).

    transformers.Trainer.train only auto-discovers the newest checkpoint when
    ``resume_from_checkpoint`` is the bool ``True``; a string is treated as the
    exact checkpoint dir. A run root holds only ``checkpoint-*`` subdirs, so we
    must descend to the newest one ourselves.
    """
    from transformers.trainer_utils import get_last_checkpoint

    found = get_last_checkpoint(str(root))
    return Path(found) if found else None


def resolve_resume_checkpoint(
    model_def: ModelDefinition,
    resume: Optional[str],
) -> Optional[Path]:
    """Resolve ``--resume auto|PATH`` to a checkpoint path for ``trainer.train``.

    ``None`` / empty → fresh run (no resume). ``auto`` and a run_id both resolve
    to the newest ``checkpoint-*`` directory under that run's out_dir. An
    explicit path is used as-is.
    """
    if resume is None or resume == "":
        return None
    if resume == "auto":
        rec = latest_incomplete_run(model_def)
        if rec is None:
            raise FileNotFoundError(
                f"No incomplete run to resume under {model_def.output_dir}"
            )
        root = Path(rec.out_dir)
        ckpt = _last_trainer_checkpoint(root)
        if ckpt is None:
            raise FileNotFoundError(
                f"Run {rec.run_id!r} at {root} has no checkpoint-* to resume from"
            )
        return ckpt
    path = Path(resume)
    if not path.is_absolute():
        # Allow run_id or path relative to model dir / checkpoints.
        as_run = get_run(model_def, resume)
        if as_run is not None:
            root = Path(as_run.out_dir)
            ckpt = _last_trainer_checkpoint(root)
            if ckpt is None:
                raise FileNotFoundError(
                    f"Run {resume!r} at {root} has no checkpoint-* to resume from"
                )
            return ckpt
        cand = model_def.checkpoints_dir / resume
        if cand.exists():
            return cand
        cand = model_def.model_dir / resume
        if cand.exists():
            return cand
    if not path.exists():
        raise FileNotFoundError(f"Resume checkpoint not found: {resume}")
    return path.resolve()


def resolve_checkpoint(
    model_def: ModelDefinition,
    checkpoint: str | Path | None = None,
) -> Path:
    """Resolve an eval/train checkpoint: run_id, path, or latest completed.

    Falls back to the newest directory under ``output/checkpoints/`` when the
    registry is empty (legacy name@version / smoke dirs).
    """
    if checkpoint is not None and str(checkpoint).strip():
        raw = str(checkpoint).strip()
        path = Path(raw)
        if path.exists():
            return path.resolve()
        # Model-dir-relative path (e.g. output/export/<run_id>).
        cand = model_def.model_dir / raw
        if cand.exists():
            return cand.resolve()
        rec = get_run(model_def, raw)
        if rec is not None:
            return Path(rec.out_dir)
        cand = model_def.checkpoints_dir / raw
        if cand.exists():
            return cand.resolve()
        raise FileNotFoundError(
            f"Checkpoint {raw!r} not found as path or run_id under "
            f"{model_def.checkpoints_dir}"
        )

    completed = latest_completed_run(model_def)
    if completed is not None:
        out = Path(completed.out_dir)
        if out.exists():
            return out

    # Legacy fallback: newest mtime under checkpoints/
    ckpt_root = model_def.checkpoints_dir
    if not ckpt_root.exists():
        raise FileNotFoundError(
            f"No checkpoints under {ckpt_root}. Run `maatml train {model_def.model_dir}` first."
        )
    candidates = [p for p in ckpt_root.iterdir() if p.is_dir()]
    if not candidates:
        raise FileNotFoundError(f"No checkpoint directories in {ckpt_root}")
    return max(candidates, key=lambda p: p.stat().st_mtime)


def write_run_sidecar(out_dir: Path, record: RunRecord) -> Path:
    """Optional small sidecar next to weights (debug); not the registry."""
    return write_json(Path(out_dir) / "run_record.json", record.model_dump(mode="json"))


def normalize_report_to(raw: Any) -> list[str] | str:
    """Normalize ``training.report_to`` for HuggingFace TrainingArguments."""
    if raw is None or raw == "none" or raw == []:
        return []
    if isinstance(raw, str):
        return [] if raw.lower() == "none" else [raw]
    if isinstance(raw, (list, tuple)):
        return [str(x) for x in raw if str(x).lower() != "none"]
    return []


def begin_training_run(
    model_def: ModelDefinition,
    *,
    smoke: bool = False,
    device: Optional[str] = None,
    profile: Optional[str] = None,
    out_dir: Optional[Path] = None,
    resume: Optional[str] = None,
    trial: Optional[dict[str, Any]] = None,
) -> tuple[RunRecord, Path, Optional[Path]]:
    """Start (or resume) a training run.

    Returns ``(run, out_dir, resume_checkpoint_path)``.
    """
    resume_path = resolve_resume_checkpoint(model_def, resume) if resume else None

    if out_dir is not None:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        run = start_run(
            model_def,
            smoke=smoke,
            device=device,
            profile=profile,
            out_dir=out,
            trial=trial,
        )
        return run, out, resume_path

    if resume_path is not None:
        resume_root = resume_path
        if resume_root.name.startswith("checkpoint-"):
            resume_root = resume_root.parent
        existing = get_run(model_def, resume_root.name)
        if existing is None:
            for rec in list_runs(model_def):
                if Path(rec.out_dir).resolve() == resume_root.resolve():
                    existing = rec
                    break
        if existing is not None:
            out = Path(existing.out_dir)
            run = start_run(
                model_def,
                smoke=smoke,
                device=device,
                profile=profile,
                run_id=existing.run_id,
                out_dir=out,
                trial=trial,
            )
            return run, out, resume_path
        run = start_run(
            model_def,
            smoke=smoke,
            device=device,
            profile=profile,
            out_dir=resume_root,
            trial=trial,
        )
        return run, Path(run.out_dir), resume_path

    run = start_run(
        model_def,
        smoke=smoke,
        device=device,
        profile=profile,
        trial=trial,
    )
    return run, Path(run.out_dir), None
