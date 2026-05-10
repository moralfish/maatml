"""flow-ml CLI.

Each command takes a model folder (containing ``model.yml``) and dispatches to
the right pipeline based on ``model.yml``'s ``task`` field. Outputs land under
``<model-dir>/output/`` (gitignored).

  flow_ml prepare  <model-dir>
  flow_ml train    <model-dir> [--smoke]
  flow_ml evaluate <model-dir> [--checkpoint X] [--split test]
  flow_ml package  <model-dir> [--checkpoint X] [--version vN]
  flow_ml verify   <fm-or-dir>
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
    if md.task == "jcl_validation":
        from .training.jcl_validator import train_jcl
        result = train_jcl(md, smoke=smoke, limit=limit, device=device, seed=seed)
    elif md.task == "spool_interpretation":
        from .training.spool_interpreter import train_spool
        result = train_spool(md, smoke=smoke, limit=limit, device=device, seed=seed)
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


@app.command("package")
def cmd_package(
    model_dir: Path = typer.Argument(..., exists=True, file_okay=False),
    checkpoint: Optional[Path] = typer.Option(None, "--checkpoint"),
    version: Optional[str] = typer.Option(None, "--version", help="Override version (default: model.yml `version`)"),
) -> None:
    """Package a checkpoint into <model-dir>/output/dist/<model_id>-<version>{,.fm}."""
    md = load_model_def(model_dir)
    ckpt = checkpoint if checkpoint else _latest_checkpoint(md)
    ver = version or md.version
    # `model_id` looks like 'dsl-generator:v1' - sanitize for filesystem use
    safe_id = md.model_id.replace(":", "-")
    dist_name = f"{safe_id}-{ver}" if not safe_id.endswith(f"-{ver}") else safe_id
    dist_dir = md.dist_dir / dist_name
    pkg_kwargs = dict(
        model_id=md.model_id,
        base_checkpoint=md.base_model,
        max_input_tokens=md.packaging.max_input_tokens,
        expected_latency_ms=md.packaging.expected_latency_ms,
        version=ver,
    )
    prompt_spec = md.resolve(md.data["prompt_spec"]) if md.data.get("prompt_spec") else None
    schema = md.resolve(md.data["schema"]) if md.data.get("schema") else None
    contracts = md.resolve(md.data["contracts"]) if md.data.get("contracts") else None
    if md.task == "jcl_validation":
        from .packaging.package_model import package_jcl
        result = package_jcl(
            ckpt,
            dist_dir,
            prompt_spec_path=prompt_spec,
            schema_path=schema,
            contracts_path=contracts,
            weights_dtype=md.packaging.weights_dtype,
            **pkg_kwargs,
        )
    elif md.task == "spool_interpretation":
        from .packaging.package_model import package_spool
        result = package_spool(
            ckpt,
            dist_dir,
            prompt_spec_path=prompt_spec,
            schema_path=schema,
            contracts_path=contracts,
            weights_dtype=md.packaging.weights_dtype,
            **pkg_kwargs,
        )
    elif md.task == "flow_graph_generation":
        from .packaging.package_model import package_flow_graph
        result = package_flow_graph(
            ckpt,
            dist_dir,
            prompt_spec_path=prompt_spec,
            schema_path=schema,
            contracts_path=contracts,
            weights_dtype=md.packaging.weights_dtype,
            **pkg_kwargs,
        )
    else:
        raise typer.BadParameter(f"Unknown task: {md.task}")
    console.print(f"[green]packaged[/] dir={result.pkg_dir} fm={result.fm_path}")


@app.command("verify")
def cmd_verify(
    pkg: Path = typer.Argument(..., exists=True, help="Either an unpacked package directory or a .fm archive"),
) -> None:
    """Reload the package via transformers and run a one-shot forward pass."""
    from .packaging.package_model import verify_package
    result = verify_package(pkg)
    if result.ok:
        console.print(f"[green]verify ok[/] pkg={pkg}")
        return
    console.print(f"[red]verify failed[/] pkg={pkg}")
    for f, ok in result.checked_files.items():
        marker = "OK" if ok else "BAD"
        console.print(f"  [{marker}] {f}")
    for issue in result.issues:
        console.print(f"  - {issue}")
    raise typer.Exit(code=1)


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
    console.print(f"5. flow_ml package {md.model_dir} --version {md.version}")


def main() -> None:
    app()


if __name__ == "__main__":
    main()
