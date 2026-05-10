"""Evaluate all three flow-ml SFT models sequentially in one process.

For each task: locate the most recent checkpoint under
`<model-dir>/output/checkpoints/`, render the held-out test split through
the model, run the per-task 6-layer validator, write `report.{json,md}`
under `<model-dir>/output/eval/`.

A failure in one task does NOT abort the others — failures are
collected and reported at the end.

Usage:
    .venv/bin/python scripts/evaluate_all.py
    .venv/bin/python scripts/evaluate_all.py --only jcl spool
    .venv/bin/python scripts/evaluate_all.py --split val
    .venv/bin/python scripts/evaluate_all.py --limit 20  # quick smoke
"""
from __future__ import annotations

import argparse
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from rich.console import Console  # noqa: E402

from flow_ml.config import load_model_def, ModelDefinition  # noqa: E402
from flow_ml.evaluation.runner import (  # noqa: E402
    evaluate_flow_graph,
    evaluate_jcl,
    evaluate_spool,
    write_markdown_summary,
)


console = Console()


@dataclass(frozen=True)
class Task:
    name: str
    model_dir: Path
    eval_fn: Callable


TASKS: dict[str, Task] = {
    "jcl": Task("jcl-validator", REPO / "models" / "jcl-validator", evaluate_jcl),
    "spool": Task("spool-interpreter", REPO / "models" / "spool-interpreter", evaluate_spool),
    "flow_graph": Task("flow-graph-generator", REPO / "models" / "flow-graph-generator", evaluate_flow_graph),
}


@dataclass
class Outcome:
    task: str
    ok: bool
    elapsed_s: float
    detail: str = ""


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


def _run_task(
    task: Task,
    *,
    checkpoint: Optional[Path],
    split: str,
    device: str,
    max_input_tokens: int,
    limit: Optional[int],
) -> Outcome:
    label = task.name
    started = time.monotonic()
    console.rule(f"[bold cyan]{label}[/]")
    try:
        md = load_model_def(task.model_dir)
        ckpt = checkpoint if checkpoint else _latest_checkpoint(md)
        md.eval_dir.mkdir(parents=True, exist_ok=True)
        out_path = md.eval_dir / f"{ckpt.name}.json"
        console.print(f"[cyan]evaluate[/] {label} checkpoint={ckpt.name} split={split}")
        report = task.eval_fn(
            ckpt,
            md.prepared_dir,
            out_path,
            device=device,
            split=split,
            max_input_tokens=max_input_tokens,
            limit=limit,
        )
        write_markdown_summary(report, out_path.with_suffix(".md"))
        elapsed = time.monotonic() - started
        # Headline metric per task: parse rate is a useful single-number signal,
        # falling back to the first available metric if the task uses a different name.
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
    parser = argparse.ArgumentParser(
        description="Evaluate all three flow-ml SFT models in one process."
    )
    parser.add_argument(
        "--only",
        nargs="+",
        choices=list(TASKS.keys()),
        help="Evaluate only the listed tasks (default: all three)",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Override checkpoint dir (applies to all tasks; default: latest under each model's output/checkpoints/)",
    )
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--device", default="auto", help="auto|mps|cpu|cuda")
    parser.add_argument("--max-input-tokens", type=int, default=4096)
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Cap number of eval rows (smoke / debug)",
    )
    args = parser.parse_args(argv)

    selected = args.only or list(TASKS.keys())
    outcomes: list[Outcome] = []

    overall_started = time.monotonic()
    for key in selected:
        outcomes.append(
            _run_task(
                TASKS[key],
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
