"""Train flow-ml models discovered under models/ (and optionally examples/).

Uses the plugin registry — no hardcoded trainer imports.

Usage:
    .venv/bin/python scripts/train_all.py
    .venv/bin/python scripts/train_all.py --smoke
    .venv/bin/python scripts/train_all.py --only jcl spool
    .venv/bin/python scripts/train_all.py --include-examples
"""
from __future__ import annotations

import argparse
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from rich.console import Console  # noqa: E402

from flow_ml.config import get_dataset_cfg, load_model_def  # noqa: E402
from flow_ml.registry import FORMATS, TRAINERS, discover_plugins  # noqa: E402
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


def _run_one(
    model_dir: Path,
    *,
    smoke: bool,
    skip_prepare: bool,
    device: str,
    seed: int | None,
    limit: int | None,
) -> Outcome:
    label = model_dir.name
    started = time.monotonic()
    console.rule(f"[bold cyan]{label}[/]")
    try:
        discover_plugins()
        md = load_model_def(model_dir)
        if not skip_prepare:
            console.print(f"[cyan]prepare[/] {label}")
            fmt = get_dataset_cfg(md).get("format", "jsonl_seed")
            FORMATS.require(str(fmt))(md)
        else:
            console.print(f"[yellow]skip prepare[/] for {label}")
        arch = normalize_architecture(md.architecture)
        trainer = TRAINERS.get(md.architecture) or TRAINERS.require(arch)
        console.print(f"[cyan]train[/] {label} (smoke={smoke}, device={device})")
        result = trainer(md, smoke=smoke, limit=limit, device=device, seed=seed)
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
    parser = argparse.ArgumentParser(description="Train discovered flow-ml models.")
    parser.add_argument(
        "--only",
        nargs="+",
        help="Subset by folder name or alias (jcl, spool, triage, ...)",
    )
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--skip-prepare", action="store_true")
    parser.add_argument("--include-examples", action="store_true")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args(argv)

    selected = _select_dirs(args.only, include_examples=args.include_examples)
    outcomes: list[Outcome] = []
    overall_started = time.monotonic()
    for model_dir in selected:
        outcomes.append(
            _run_one(
                model_dir,
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
