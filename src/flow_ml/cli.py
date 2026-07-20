"""flow-ml CLI.

Each command takes a model folder (containing ``model.yml``) and dispatches via
the plugin registry (``architecture`` / ``dataset.format`` / ``evaluation``).
Outputs land under ``<model-dir>/output/`` (gitignored).

  flow_ml prepare   <model-dir>
  flow_ml train     <model-dir> [--smoke]
  flow_ml evaluate  <model-dir> [--checkpoint X] [--split test]
  flow_ml scaffold  <dir> --architecture causal_sft [--name my-model]
  flow_ml validate  <model-dir>
  flow_ml plan      <model-dir>
  flow_ml plugins
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console

from .config import get_dataset_cfg, load_model_def
from .registry import (
    FORMATS,
    PREDICTORS,
    TRAINERS,
    discover_plugins,
    list_all_plugins,
    load_model_plugins,
)
from .scaffold import normalize_architecture, scaffold_model, validate_model_dir

# MPS unsupported-op fallback to CPU; harmless on non-Mac.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")


app = typer.Typer(no_args_is_help=True, add_completion=False, help="flow-ml CLI")
console = Console()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _latest_checkpoint(model_def) -> Path:
    """Find the most recently modified checkpoint directory under output/checkpoints/."""
    ckpt_root = model_def.checkpoints_dir
    if not ckpt_root.exists():
        raise FileNotFoundError(
            f"No checkpoints under {ckpt_root}. Run `flow_ml train {model_def.model_dir}` first."
        )
    candidates = [p for p in ckpt_root.iterdir() if p.is_dir()]
    if not candidates:
        raise FileNotFoundError(f"No checkpoint directories in {ckpt_root}")
    return max(candidates, key=lambda p: p.stat().st_mtime)


def _boot_plugins(md) -> None:
    discover_plugins()
    if md.plugins:
        load_model_plugins(md.model_dir, md.plugins)


def _default_eval_keys(md) -> tuple[Optional[str], Optional[str], Optional[str]]:
    """Infer predictor/validator/metrics from evaluation: or architecture/task."""
    ev = md.evaluation or {}
    predictor = ev.get("predictor")
    validator = ev.get("validator")
    metrics = ev.get("metrics")
    if isinstance(metrics, list):
        metrics = metrics[0] if metrics else None

    arch = normalize_architecture(md.architecture)
    if predictor is None:
        if arch in PREDICTORS.names() or PREDICTORS.get(md.architecture):
            predictor = md.architecture if PREDICTORS.get(md.architecture) else arch
        elif arch == "multi_head_classifier":
            predictor = "multi_head_classifier"
        elif arch == "seq2seq":
            predictor = "seq2seq"
        elif arch == "causal_sft":
            predictor = "causal_sft"

    if validator is None:
        if md.task == "jcl_validation":
            validator = "jcl"
        elif md.task == "spool_interpretation":
            validator = "spool"

    if metrics is None:
        if md.task == "jcl_validation" or validator == "jcl":
            metrics = "jcl"
        elif md.task == "spool_interpretation" or validator == "spool":
            metrics = "spool"

    return predictor, validator, metrics


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------


@app.command("prepare")
def cmd_prepare(
    model_dir: Path = typer.Argument(
        ...,
        exists=True,
        file_okay=False,
        help="Path to a model folder containing model.yml",
    ),
) -> None:
    """Build train/val/test splits under <model-dir>/output/prepared/."""
    md = load_model_def(model_dir)
    _boot_plugins(md)
    fmt = get_dataset_cfg(md).get("format", "jsonl_seed")
    prepare_fn = FORMATS.require(str(fmt))
    prepare_fn(md)


@app.command("train")
def cmd_train(
    model_dir: Path = typer.Argument(..., exists=True, file_okay=False),
    smoke: bool = typer.Option(False, "--smoke", help="Use the `smoke:` overrides in model.yml"),
    limit: Optional[int] = typer.Option(None, "--limit", help="Cap number of train rows for ad-hoc smoke"),
    device: str = typer.Option("auto", "--device", help="auto|mps|cpu|cuda"),
    seed: Optional[int] = typer.Option(None, "--seed"),
) -> None:
    """Fine-tune the model declared by <model-dir>/model.yml."""
    md = load_model_def(model_dir)
    _boot_plugins(md)
    arch = normalize_architecture(md.architecture)
    trainer = TRAINERS.get(md.architecture) or TRAINERS.require(arch)
    result = trainer(md, smoke=smoke, limit=limit, device=device, seed=seed)
    console.print(f"[green]done[/] out_dir={result.out_dir} metrics={result.metrics}")


@app.command("evaluate")
def cmd_evaluate(
    model_dir: Path = typer.Argument(..., exists=True, file_okay=False),
    checkpoint: Optional[Path] = typer.Option(
        None,
        "--checkpoint",
        help="Override checkpoint dir; defaults to most recent under output/checkpoints/",
    ),
    split: str = typer.Option("test", "--split"),
    device: str = typer.Option("auto", "--device"),
    baseline: Optional[Path] = typer.Option(None, "--baseline"),
    max_input_tokens: int = typer.Option(2048, "--max-input-tokens"),
    limit: Optional[int] = typer.Option(None, "--limit"),
) -> None:
    """Evaluate the most recent checkpoint and write report.{json,md} under output/eval/."""
    md = load_model_def(model_dir)
    _boot_plugins(md)
    # Ensure predictor registrations are loaded.
    from .evaluation import predictors as _predictors  # noqa: F401
    from .evaluation.harness import run_evaluation
    from .evaluation.runner import write_markdown_summary

    ckpt = checkpoint if checkpoint else _latest_checkpoint(md)
    md.eval_dir.mkdir(parents=True, exist_ok=True)
    out_path = md.eval_dir / f"{ckpt.name}.json"

    predictor, validator, metrics = _default_eval_keys(md)
    if predictor is None:
        raise typer.BadParameter(
            f"No predictor for architecture={md.architecture!r}; "
            "set evaluation.predictor in model.yml"
        )

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
        baseline_path=baseline,
        limit=limit,
        task=md.task,
    )
    write_markdown_summary(report, out_path.with_suffix(".md"))
    console.print(f"[green]done[/] report={out_path}")


@app.command("scaffold")
def cmd_scaffold(
    target_dir: Path = typer.Argument(..., help="Directory to create (model folder)"),
    architecture: str = typer.Option(
        ...,
        "--architecture",
        "-a",
        help="Registered trainer architecture (e.g. causal_sft, seq2seq, classifier)",
    ),
    name: Optional[str] = typer.Option(None, "--name", help="Model name (default: folder name)"),
) -> None:
    """Create a new model folder with model.yml, datasets, and README."""
    discover_plugins()
    path = scaffold_model(target_dir, architecture=architecture, name=name)
    console.print(f"[green]scaffolded[/] {path}")


@app.command("validate")
def cmd_validate(
    model_dir: Path = typer.Argument(..., exists=True, file_okay=False),
) -> None:
    """Validate model.yml paths, architecture registration, and dataset.format."""
    errors = validate_model_dir(model_dir)
    if errors:
        console.print("[red]validate failed[/]")
        for err in errors:
            console.print(f"  - {err}")
        raise typer.Exit(code=1)
    md = load_model_def(model_dir)
    console.print(f"[green]OK[/] {md.identity} ({md.architecture})")


@app.command("plan")
def cmd_plan(
    model_dir: Path = typer.Argument(..., exists=True, file_okay=False),
) -> None:
    """Print the lifecycle command plan for a model folder."""
    md = load_model_def(model_dir)
    console.print(f"[bold]{md.identity}[/] ({md.task} / {md.architecture})")
    console.print(f"1. flow_ml prepare {md.model_dir}")
    console.print(f"2. flow_ml train {md.model_dir} --smoke")
    console.print(f"3. flow_ml train {md.model_dir}")
    console.print(f"4. flow_ml evaluate {md.model_dir}")


@app.command("plugins")
def cmd_plugins() -> None:
    """List registered trainers, validators, metrics, formats, and predictors."""
    discover_plugins()
    # Import submodule directly so listing plugins does not require torch.
    from .evaluation import predictors as _predictors  # noqa: F401

    for kind, entries in list_all_plugins().items():
        console.print(f"[bold]{kind}[/] ({len(entries)})")
        for entry in entries:
            console.print(f"  {entry.name}  [dim]{entry.source}[/]")


def main() -> None:
    app()


if __name__ == "__main__":
    main()
