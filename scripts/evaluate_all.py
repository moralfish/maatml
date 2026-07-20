"""Evaluate flow-ml models discovered under models/ (and optionally examples/).

Uses the evaluation harness + plugin registry — no hardcoded evaluate_* imports.

Usage:
    .venv/bin/python scripts/evaluate_all.py
    .venv/bin/python scripts/evaluate_all.py --only jcl spool
    .venv/bin/python scripts/evaluate_all.py --split val
    .venv/bin/python scripts/evaluate_all.py --limit 20
"""
from __future__ import annotations

import argparse
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from rich.console import Console  # noqa: E402

from flow_ml.config import ModelDefinition, load_model_def  # noqa: E402
from flow_ml.registry import discover_plugins  # noqa: E402
from flow_ml.scaffold import normalize_architecture  # noqa: E402

console = Console()

_NAME_ALIASES = {
    "jcl": "jcl-validator",
    "spool": "spool-interpreter",
    "tickets": "support-ticket-triage",
    "triage": "support-ticket-triage",
}


@dataclass
class Outcome:
    task: str
    ok: bool
    elapsed_s: float
    detail: str = ""


def discover_model_dirs(*, include_examples: bool = False) -> list[Path]:
    roots = [REPO / "models"]
    if include_examples:
        roots.append(REPO / "examples")
    dirs: list[Path] = []
    for root in roots:
        if not root.is_dir():
            continue
        for child in sorted(root.iterdir()):
            if child.is_dir() and (child / "model.yml").is_file():
                dirs.append(child)
    return dirs


def _select_dirs(only: list[str] | None, *, include_examples: bool) -> list[Path]:
    all_dirs = discover_model_dirs(include_examples=include_examples or bool(only))
    if not only:
        return [d for d in all_dirs if "models" in d.parts]
    selected: list[Path] = []
    for key in only:
        alias = _NAME_ALIASES.get(key, key)
        matches = [d for d in all_dirs if d.name == alias or key in d.name]
        if not matches:
            raise SystemExit(f"No model matched --only {key!r}")
        for m in matches:
            if m not in selected:
                selected.append(m)
    return selected


def _latest_checkpoint(md: ModelDefinition) -> Path:
    ckpt_root = md.checkpoints_dir
    if not ckpt_root.exists():
        raise FileNotFoundError(
            f"No checkpoints under {ckpt_root}. Run train_all.py first."
        )
    candidates = [p for p in ckpt_root.iterdir() if p.is_dir()]
    if not candidates:
        raise FileNotFoundError(f"No checkpoint directories in {ckpt_root}")
    return max(candidates, key=lambda p: p.stat().st_mtime)


def _default_eval_keys(md: ModelDefinition) -> tuple[Optional[str], Optional[str], Optional[str]]:
    from flow_ml.registry import PREDICTORS

    ev = md.evaluation or {}
    predictor = ev.get("predictor")
    validator = ev.get("validator")
    metrics = ev.get("metrics")
    if isinstance(metrics, list):
        metrics = metrics[0] if metrics else None

    arch = normalize_architecture(md.architecture)
    if predictor is None:
        if PREDICTORS.get(md.architecture):
            predictor = md.architecture
        elif PREDICTORS.get(arch):
            predictor = arch

    if validator is None:
        if md.task == "jcl_validation":
            validator = "jcl"
        elif md.task == "spool_interpretation":
            validator = "spool"

    if metrics is None:
        if validator == "jcl" or md.task == "jcl_validation":
            metrics = "jcl"
        elif validator == "spool" or md.task == "spool_interpretation":
            metrics = "spool"

    return predictor, validator, metrics


def _run_one(
    model_dir: Path,
    *,
    checkpoint: Optional[Path],
    split: str,
    device: str,
    max_input_tokens: int,
    limit: Optional[int],
) -> Outcome:
    label = model_dir.name
    started = time.monotonic()
    console.rule(f"[bold cyan]{label}[/]")
    try:
        discover_plugins()
        from flow_ml.evaluation import predictors as _predictors  # noqa: F401
        from flow_ml.evaluation.harness import run_evaluation
        from flow_ml.evaluation.runner import write_markdown_summary

        md = load_model_def(model_dir)
        ckpt = checkpoint if checkpoint else _latest_checkpoint(md)
        md.eval_dir.mkdir(parents=True, exist_ok=True)
        out_path = md.eval_dir / f"{ckpt.name}.json"
        predictor, validator, metrics = _default_eval_keys(md)
        if predictor is None:
            raise ValueError(f"No predictor for {md.architecture!r}")
        console.print(f"[cyan]evaluate[/] {label} checkpoint={ckpt.name} split={split}")
        report = run_evaluation(
            checkpoint_dir=ckpt,
            dataset_dir=md.prepared_dir,
            out_path=out_path,
            model_def=md,
            predictor=predictor,
            validator=validator,
            metrics_fn=metrics,
            device=device,
            split=split,
            max_input_tokens=max_input_tokens,
            limit=limit,
            task=md.task,
        )
        write_markdown_summary(report, out_path.with_suffix(".md"))
        elapsed = time.monotonic() - started
        headline = (
            report.metrics.get("json_parse_rate")
            or report.metrics.get("schema_conformance_rate")
            or (next(iter(report.metrics.values())) if report.metrics else 0.0)
        )
        detail = f"n={report.n} report={out_path.name} headline={headline:.3f}"
        console.print(f"[green]{label} done[/] in {elapsed:.1f}s — {detail}")
        return Outcome(task=label, ok=True, elapsed_s=elapsed, detail=detail)
    except Exception as exc:  # noqa: BLE001
        elapsed = time.monotonic() - started
        console.print(f"[red]{label} FAILED[/] in {elapsed:.1f}s — {type(exc).__name__}: {exc}")
        traceback.print_exc()
        return Outcome(
            task=label,
            ok=False,
            elapsed_s=elapsed,
            detail=f"{type(exc).__name__}: {exc}",
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate discovered flow-ml models.")
    parser.add_argument("--only", nargs="+")
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-input-tokens", type=int, default=4096)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--include-examples", action="store_true")
    args = parser.parse_args(argv)

    selected = _select_dirs(args.only, include_examples=args.include_examples)
    outcomes: list[Outcome] = []
    overall_started = time.monotonic()
    for model_dir in selected:
        outcomes.append(
            _run_one(
                model_dir,
                checkpoint=args.checkpoint,
                split=args.split,
                device=args.device,
                max_input_tokens=args.max_input_tokens,
                limit=args.limit,
            )
        )

    total_elapsed = time.monotonic() - overall_started
    console.rule("[bold]summary")
    for o in outcomes:
        marker = "[green]ok[/]" if o.ok else "[red]FAIL[/]"
        console.print(f"  {marker} {o.task} ({o.elapsed_s:.1f}s) — {o.detail}")
    console.print(f"[bold]total elapsed[/]: {total_elapsed:.1f}s")
    return 0 if all(o.ok for o in outcomes) else 1


if __name__ == "__main__":
    raise SystemExit(main())
