"""Train all three flow-ml SFT models sequentially in one process.

Runs `prepare` (unless --skip-prepare) and then `train` for each of:
  jcl-validator, spool-interpreter, flow-graph-generator

All three share Qwen3-1.7B as the base, so the second and third loads
hit the HuggingFace disk cache and finish in seconds. Each task still
gets its own LoRA adapter and its own checkpoint dir under
`<model-dir>/output/checkpoints/`.

A failure in one task does NOT abort the others — failures are
collected and reported at the end so a long unattended run still
gets you whatever finished.

Usage:
    .venv/bin/python scripts/train_all.py                         # full run
    .venv/bin/python scripts/train_all.py --smoke                 # all three in smoke mode
    .venv/bin/python scripts/train_all.py --only jcl spool        # subset
    .venv/bin/python scripts/train_all.py --skip-prepare          # assume splits exist
"""
from __future__ import annotations

import argparse
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from rich.console import Console  # noqa: E402

from flow_ml.config import load_model_def  # noqa: E402
from flow_ml.data.pipeline import (  # noqa: E402
    prepare_flow_graph,
    prepare_jcl,
    prepare_spool,
)
from flow_ml.training.flow_graph_generator import train_flow_graph  # noqa: E402
from flow_ml.training.jcl_classifier import train_jcl_classifier  # noqa: E402
from flow_ml.training.spool_seq2seq import train_spool_seq2seq  # noqa: E402


console = Console()


@dataclass(frozen=True)
class Task:
    name: str
    model_dir: Path
    prepare_fn: Callable
    train_fn: Callable


TASKS: dict[str, Task] = {
    "jcl": Task(
        name="jcl-validator",
        model_dir=REPO / "models" / "jcl-validator",
        prepare_fn=prepare_jcl,
        train_fn=train_jcl_classifier,
    ),
    "spool": Task(
        name="spool-interpreter",
        model_dir=REPO / "models" / "spool-interpreter",
        prepare_fn=prepare_spool,
        train_fn=train_spool_seq2seq,
    ),
    "flow_graph": Task(
        name="flow-graph-generator",
        model_dir=REPO / "models" / "flow-graph-generator",
        prepare_fn=prepare_flow_graph,
        train_fn=train_flow_graph,
    ),
}


@dataclass
class Outcome:
    task: str
    ok: bool
    elapsed_s: float
    detail: str = ""


def _run_task(
    task: Task,
    *,
    smoke: bool,
    skip_prepare: bool,
    device: str,
    seed: int | None,
    limit: int | None,
) -> Outcome:
    label = task.name
    started = time.monotonic()
    console.rule(f"[bold cyan]{label}[/]")
    try:
        md = load_model_def(task.model_dir)
        if not skip_prepare:
            console.print(f"[cyan]prepare[/] {label}")
            task.prepare_fn(md)
        else:
            console.print(f"[yellow]skip prepare[/] for {label}")
        console.print(f"[cyan]train[/] {label} (smoke={smoke}, device={device})")
        result = task.train_fn(
            md, smoke=smoke, limit=limit, device=device, seed=seed
        )
        elapsed = time.monotonic() - started
        detail = f"out_dir={result.out_dir} metrics={result.metrics}"
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
        description="Train all three flow-ml SFT models in one process."
    )
    parser.add_argument(
        "--only",
        nargs="+",
        choices=list(TASKS.keys()),
        help="Train only the listed tasks (default: all three)",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Use the `smoke:` overrides in each model.yml (Qwen3-0.6B, few steps)",
    )
    parser.add_argument(
        "--skip-prepare",
        action="store_true",
        help="Assume output/prepared/{train,val,test}.jsonl already exist",
    )
    parser.add_argument("--device", default="auto", help="auto|mps|cpu|cuda")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--limit", type=int, default=None, help="Cap train rows (debug)")
    args = parser.parse_args(argv)

    selected = args.only or list(TASKS.keys())
    outcomes: list[Outcome] = []

    overall_started = time.monotonic()
    for key in selected:
        outcomes.append(
            _run_task(
                TASKS[key],
                smoke=args.smoke,
                skip_prepare=args.skip_prepare,
                device=args.device,
                seed=args.seed,
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
