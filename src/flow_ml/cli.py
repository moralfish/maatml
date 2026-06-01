"""flow-ml CLI.

Each command takes a model folder (containing ``model.yml``) and dispatches to
the right pipeline based on ``model.yml``'s ``task`` field. Outputs land under
``<model-dir>/output/`` (gitignored).

  flow_ml prepare  <model-dir>
  flow_ml train    <model-dir> [--smoke]
  flow_ml evaluate <model-dir> [--checkpoint X] [--split test]
  flow_ml plan     <model-dir>
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console

from .config import ModelDefinition, load_model_def
from .data.pipeline import (
    prepare_flow_graph,
    prepare_jcl,
    prepare_spool,
)

# MPS unsupported-op fallback to CPU; harmless on non-Mac.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")


app = typer.Typer(no_args_is_help=True, add_completion=False, help="flow-ml CLI")
console = Console()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _latest_checkpoint(model_def: ModelDefinition) -> Path:
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


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

@app.command("prepare")
def cmd_prepare(
    model_dir: Path = typer.Argument(..., exists=True, file_okay=False, help="Path to a model folder containing model.yml"),
) -> None:
    """Build train/val/test splits under <model-dir>/output/prepared/."""
    md = load_model_def(model_dir)
    if md.task == "jcl_validation":
        prepare_jcl(md)
    elif md.task == "spool_interpretation":
        prepare_spool(md)
    elif md.task == "flow_graph_generation":
        prepare_flow_graph(md)
    else:
        raise typer.BadParameter(f"Unknown task: {md.task}")


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
    # Architecture-aware dispatch: each model.yml declares its
    # `architecture` (`classifier` / `seq2seq` / falls through to the
    # generative SFT default). Lets JCL run the BERT classifier path,
    # Spool run the T5 seq2seq path, and FGG keep the LoRA SFT path.
    architecture = md.architecture
    if md.task == "jcl_validation":
        if architecture == "classifier":
            from .training.jcl_classifier import train_jcl_classifier
            result = train_jcl_classifier(
                md, smoke=smoke, limit=limit, device=device, seed=seed
            )
        else:
            raise typer.BadParameter(
                "jcl_validation now requires `architecture: classifier` in model.yml; "
                "the v1 generative-SFT path was retired."
            )
    elif md.task == "spool_interpretation":
        if architecture == "seq2seq":
            from .training.spool_seq2seq import train_spool_seq2seq
            result = train_spool_seq2seq(
                md, smoke=smoke, limit=limit, device=device, seed=seed
            )
        else:
            raise typer.BadParameter(
                "spool_interpretation now requires `architecture: seq2seq` in model.yml; "
                "the v1 generative-SFT path was retired."
            )
    elif md.task == "flow_graph_generation":
        from .training.flow_graph_generator import train_flow_graph
        result = train_flow_graph(md, smoke=smoke, limit=limit, device=device, seed=seed)
    else:
        raise typer.BadParameter(f"Unknown task: {md.task}")
    console.print(f"[green]done[/] out_dir={result.out_dir} metrics={result.metrics}")


@app.command("evaluate")
def cmd_evaluate(
    model_dir: Path = typer.Argument(..., exists=True, file_okay=False),
    checkpoint: Optional[Path] = typer.Option(None, "--checkpoint", help="Override checkpoint dir; defaults to most recent under output/checkpoints/"),
    split: str = typer.Option("test", "--split"),
    device: str = typer.Option("auto", "--device"),
    baseline: Optional[Path] = typer.Option(None, "--baseline"),
    max_input_tokens: int = typer.Option(2048, "--max-input-tokens"),
) -> None:
    """Evaluate the most recent checkpoint and write report.{json,md} under output/eval/."""
    md = load_model_def(model_dir)
    ckpt = checkpoint if checkpoint else _latest_checkpoint(md)
    md.eval_dir.mkdir(parents=True, exist_ok=True)
    out_path = md.eval_dir / f"{ckpt.name}.json"

    from .evaluation.runner import write_markdown_summary
    if md.task == "jcl_validation":
        from .evaluation.runner import evaluate_jcl
        report = evaluate_jcl(
            ckpt, md.prepared_dir, out_path,
            baseline_path=baseline, device=device, split=split,
            max_input_tokens=max_input_tokens,
        )
    elif md.task == "spool_interpretation":
        from .evaluation.runner import evaluate_spool
        report = evaluate_spool(
            ckpt, md.prepared_dir, out_path,
            baseline_path=baseline, device=device, split=split,
            max_input_tokens=max_input_tokens,
        )
    elif md.task == "flow_graph_generation":
        from .evaluation.runner import evaluate_flow_graph
        report = evaluate_flow_graph(
            ckpt, md.prepared_dir, out_path,
            baseline_path=baseline, device=device, split=split,
            max_input_tokens=max_input_tokens,
        )
    else:
        raise typer.BadParameter(f"Unknown task: {md.task}")
    write_markdown_summary(report, out_path.with_suffix(".md"))
    console.print(f"[green]done[/] report={out_path}")


@app.command("plan")
def cmd_plan(
    model_dir: Path = typer.Argument(..., exists=True, file_okay=False),
) -> None:
    """Print the lifecycle command plan for a model folder."""
    md = load_model_def(model_dir)
    console.print(f"[bold]{md.model_id}[/] ({md.task})")
    console.print(f"1. flow_ml prepare {md.model_dir}")
    console.print(f"2. flow_ml train {md.model_dir} --smoke")
    console.print(f"3. flow_ml train {md.model_dir}")
    console.print(f"4. flow_ml evaluate {md.model_dir}")


def main() -> None:
    app()


if __name__ == "__main__":
    main()
