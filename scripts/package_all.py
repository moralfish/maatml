"""Package all three flow-ml SFT models into deployable .fm archives.

For each task: locate the most recent checkpoint under
`<model-dir>/output/checkpoints/`, copy weights + tokenizer + prompt
spec + schema + node_contracts into `<model-dir>/output/dist/<id>-<ver>/`,
write a manifest + sha256 sidecars, then bundle the directory into a
single `.fm` archive (deflated zip — drag-and-droppable into Flow
Studio's Models drawer).

A failure in one task does NOT abort the others.

Usage:
    .venv/bin/python scripts/package_all.py
    .venv/bin/python scripts/package_all.py --only flow_graph
    .venv/bin/python scripts/package_all.py --version v0.2
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
from flow_ml.packaging.package_model import (  # noqa: E402
    package_flow_graph,
    package_jcl,
    package_spool,
)


console = Console()


@dataclass(frozen=True)
class Task:
    name: str
    model_dir: Path
    package_fn: Callable


TASKS: dict[str, Task] = {
    "jcl": Task("jcl-validator", REPO / "models" / "jcl-validator", package_jcl),
    "spool": Task("spool-interpreter", REPO / "models" / "spool-interpreter", package_spool),
    "flow_graph": Task("flow-graph-generator", REPO / "models" / "flow-graph-generator", package_flow_graph),
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
    version: Optional[str],
) -> Outcome:
    label = task.name
    started = time.monotonic()
    console.rule(f"[bold cyan]{label}[/]")
    try:
        md = load_model_def(task.model_dir)
        ckpt = checkpoint if checkpoint else _latest_checkpoint(md)
        ver = version or md.version
        safe_id = md.model_id.replace(":", "-")
        dist_name = f"{safe_id}-{ver}" if not safe_id.endswith(f"-{ver}") else safe_id
        dist_dir = md.dist_dir / dist_name

        prompt_spec = md.resolve(md.data["prompt_spec"]) if md.data.get("prompt_spec") else None
        schema = md.resolve(md.data["schema"]) if md.data.get("schema") else None
        contracts = md.resolve(md.data["contracts"]) if md.data.get("contracts") else None

        console.print(f"[cyan]package[/] {label} checkpoint={ckpt.name} version={ver}")
        result = task.package_fn(
            ckpt,
            dist_dir,
            prompt_spec_path=prompt_spec,
            schema_path=schema,
            contracts_path=contracts,
            model_id=md.model_id,
            base_checkpoint=md.base_model,
            max_input_tokens=md.packaging.max_input_tokens,
            expected_latency_ms=md.packaging.expected_latency_ms,
            version=ver,
            weights_dtype=md.packaging.weights_dtype,
        )
        elapsed = time.monotonic() - started
        fm = result.fm_path
        size_mb = (fm.stat().st_size / (1024 * 1024)) if fm and fm.exists() else 0.0
        detail = f"dir={result.pkg_dir.name} fm={fm.name if fm else 'none'} size={size_mb:.1f}MB"
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
        description="Package all three flow-ml SFT models into .fm archives."
    )
    parser.add_argument(
        "--only",
        nargs="+",
        choices=list(TASKS.keys()),
        help="Package only the listed tasks (default: all three)",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Override checkpoint dir (default: latest under each model's output/checkpoints/)",
    )
    parser.add_argument(
        "--version",
        default=None,
        help="Override version (default: model.yml `version` of each task)",
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
                version=args.version,
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
