"""maatml CLI.

Each command takes a model folder (containing ``model.yml``) and dispatches via
the plugin registry (``architecture`` / ``dataset.format`` / ``evaluation``).
Outputs land under ``<model-dir>/output/`` (gitignored).

  maatml prepare   <model-dir>
  maatml train     <model-dir> [--smoke] [--resume auto|PATH] [--set K=V]
  maatml sweep     <model-dir> --param K=a,b [--metric NAME] [--smoke]
  maatml evaluate  <model-dir> [--checkpoint X] [--split test] [--gate]
  maatml export    <model-dir> [--checkpoint X] [--format gguf|mlx|safetensors|onnx]
  maatml verify    <export-dir-or-manifest>
  maatml serve     <model-dir> [--checkpoint X] [--host HOST] [--port N]
  maatml datagen   <model-dir> [--target N] [--teacher]
  maatml ingest    <model-dir> --input PATH [--map field=col] [--sanitize tag]
  maatml runs      <model-dir>
  maatml scaffold  <dir> --architecture causal_sft [--name my-model]
  maatml validate  <model-dir>
  maatml plan      <model-dir>
  maatml plugins
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Optional

import typer
from rich.console import Console

from .config import config_key_warnings, get_dataset_cfg, load_model_def
from .overrides import (
    apply_overrides,
    expand_param_grid,
    overrides_from_mapping,
    pick_metric,
)
from .registry import (
    FORMATS,
    PREDICTORS,
    TRAINERS,
    discover_plugins,
    list_all_plugins,
    load_model_plugins,
)
from .runs import list_runs, resolve_checkpoint
from .scaffold import normalize_architecture, scaffold_model, validate_model_dir

# MPS unsupported-op fallback to CPU; harmless on non-Mac.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")


app = typer.Typer(no_args_is_help=True, add_completion=False, help="maatml CLI")
console = Console()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _boot_plugins(md) -> None:
    discover_plugins()
    if md.plugins:
        load_model_plugins(md.model_dir, md.plugins)


def _default_eval_keys(md) -> tuple[Optional[str], Optional[str], Any]:
    """Infer predictor/validator/metrics from evaluation: or architecture/task.

    ``evaluation.metrics`` may be a single name or a list; every entry runs and
    the results are merged (the harness rejects two plugins claiming the same
    metric key). A list is no longer silently truncated to its first entry.
    """
    ev = md.evaluation or {}
    predictor = ev.get("predictor")
    validator = ev.get("validator")
    metrics = ev.get("metrics")
    if isinstance(metrics, list) and not metrics:
        metrics = None

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

    # validator / metrics come from evaluation: (or model plugins); no
    # hardcoded task-name fallbacks in core.
    return predictor, validator, metrics


def _clone_model_def(md):
    """Deep-copy nested dict sections so sweep trials don't share state."""
    clone = md.model_copy(deep=True)
    object.__setattr__(clone, "model_dir", md.model_dir)
    return clone


def _parse_field_maps(maps: Optional[list[str]]) -> dict[str, str]:
    """Parse ``dest=src`` field maps for ingest."""
    out: dict[str, str] = {}
    for item in maps or []:
        if "=" not in item:
            raise typer.BadParameter(f"Expected field=col mapping, got {item!r}")
        dest, src = item.split("=", 1)
        dest, src = dest.strip(), src.strip()
        if not dest or not src:
            raise typer.BadParameter(f"Empty field map entry: {item!r}")
        out[dest] = src
    return out


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
    resume: Optional[str] = typer.Option(
        None,
        "--resume",
        help="Resume training: 'auto' (latest incomplete run) or checkpoint path/run_id",
    ),
    set_overrides: Optional[list[str]] = typer.Option(
        None,
        "--set",
        help="Override model.yml after load (repeatable), e.g. --set training.learning_rate=1e-4",
    ),
) -> None:
    """Fine-tune the model declared by <model-dir>/model.yml."""
    md = load_model_def(model_dir)
    if set_overrides:
        try:
            apply_overrides(md, set_overrides)
        except ValueError as exc:
            raise typer.BadParameter(str(exc), param_hint="--set") from exc
    _boot_plugins(md)
    arch = normalize_architecture(md.architecture)
    trainer = TRAINERS.get(md.architecture) or TRAINERS.require(arch)
    result = trainer(
        md, smoke=smoke, limit=limit, device=device, seed=seed, resume=resume
    )
    console.print(f"[green]done[/] out_dir={result.out_dir} metrics={result.metrics}")


@app.command("sweep")
def cmd_sweep(
    model_dir: Path = typer.Argument(..., exists=True, file_okay=False),
    param: Optional[list[str]] = typer.Option(
        None,
        "--param",
        help="Grid axis KEY=v1,v2 (repeatable); cartesian product of values",
    ),
    set_overrides: Optional[list[str]] = typer.Option(
        None,
        "--set",
        help="Base overrides applied to every trial (repeatable)",
    ),
    metric: Optional[str] = typer.Option(
        None,
        "--metric",
        help="Metric key to rank trials (default: first numeric metric)",
    ),
    smoke: bool = typer.Option(False, "--smoke"),
    limit: Optional[int] = typer.Option(None, "--limit"),
    device: str = typer.Option("auto", "--device"),
    seed: Optional[int] = typer.Option(None, "--seed"),
    max_trials: Optional[int] = typer.Option(
        None, "--max-trials", help="Cap number of grid combinations"
    ),
) -> None:
    """Offline cartesian HPO over ``--param`` axes (no Optuna required)."""
    base = load_model_def(model_dir)
    _boot_plugins(base)
    grid = expand_param_grid(param, max_trials=max_trials)
    if not grid:
        console.print("[yellow]empty grid — nothing to run[/]")
        raise typer.Exit(code=0)

    arch = normalize_architecture(base.architecture)
    trainer = TRAINERS.get(base.architecture) or TRAINERS.require(arch)
    ranking: list[tuple[Optional[float], dict, Any]] = []

    for i, trial_map in enumerate(grid):
        md = _clone_model_def(base)
        try:
            if set_overrides:
                apply_overrides(md, set_overrides)
            apply_overrides(md, overrides_from_mapping(trial_map))
        except ValueError as exc:
            raise typer.BadParameter(str(exc), param_hint="--set/--param") from exc
        trial_meta = {"index": i, "params": trial_map}
        console.print(
            f"[cyan]sweep[/] trial {i + 1}/{len(grid)} params={trial_map}"
        )
        result = trainer(
            md,
            smoke=smoke,
            limit=limit,
            device=device,
            seed=seed,
            trial=trial_meta,
        )
        key, val = pick_metric(result.metrics, metric)
        ranking.append((val, trial_map, result))
        console.print(
            f"  -> run={Path(result.out_dir).name} "
            f"{key or 'metric'}={val if val is not None else 'n/a'}"
        )

    scored: list[tuple[float, bool, dict, Any, Optional[str]]] = []
    for val, trial_map, result in ranking:
        key, _ = pick_metric(result.metrics, metric)
        minimize = key is not None and "loss" in key.lower()
        if val is None:
            continue
        scored.append((val, minimize, trial_map, result, key))

    if scored:
        minimize = scored[0][1]
        scored.sort(key=lambda t: t[0], reverse=not minimize)
        console.print("[bold]sweep ranking[/]")
        for rank, (val, _, trial_map, result, key) in enumerate(scored, start=1):
            console.print(
                f"  {rank}. {key}={val:.6g}  params={trial_map}  "
                f"out={Path(result.out_dir).name}"
            )
    else:
        console.print("[yellow]sweep finished with no numeric metrics to rank[/]")


@app.command("evaluate")
def cmd_evaluate(
    model_dir: Path = typer.Argument(..., exists=True, file_okay=False),
    checkpoint: Optional[str] = typer.Option(
        None,
        "--checkpoint",
        help="run_id or checkpoint dir; defaults to latest completed run",
    ),
    split: str = typer.Option("test", "--split"),
    device: str = typer.Option("auto", "--device"),
    baseline: Optional[Path] = typer.Option(None, "--baseline"),
    max_input_tokens: Optional[int] = typer.Option(
        None,
        "--max-input-tokens",
        help="Input token budget; defaults to packaging.max_input_tokens so "
        "eval measures the same budget serve and export --parity enforce",
    ),
    limit: Optional[int] = typer.Option(None, "--limit"),
    gate: bool = typer.Option(
        False,
        "--gate",
        help="Enforce evaluation.gates minima; exit non-zero on failure",
    ),
) -> None:
    """Evaluate a checkpoint and write report.{json,md} under output/eval/."""
    md = load_model_def(model_dir)
    _boot_plugins(md)
    # Ensure predictor registrations are loaded.
    from .evaluation import predictors as _predictors  # noqa: F401
    from .evaluation.harness import (
        GateConfigError,
        _resolve_metrics,
        resolve_gate_spec,
        resolve_validator,
        run_evaluation,
    )
    from .evaluation.runner import write_markdown_summary
    from .runs import get_run, update_run_gates

    # Fast-fail before the (expensive) checkpoint load if --gate has nothing to enforce.
    if gate:
        try:
            resolve_gate_spec(md)
        except GateConfigError as exc:
            raise typer.BadParameter(str(exc), param_hint="--gate") from exc

    ckpt = resolve_checkpoint(md, checkpoint)
    md.eval_dir.mkdir(parents=True, exist_ok=True)
    out_path = md.eval_dir / f"{ckpt.name}.json"

    predictor, validator, metrics = _default_eval_keys(md)
    if predictor is None:
        raise typer.BadParameter(
            f"No predictor for architecture={md.architecture!r}; "
            "set evaluation.predictor in model.yml"
        )
    if validator is None:
        console.print("[yellow]no validator configured, scoring JSON parse only[/]")
    else:
        # Fail before the expensive checkpoint load if a validator is configured
        # but does not resolve (plugins are already loaded via _boot_plugins).
        try:
            resolve_validator(validator)
        except GateConfigError as exc:
            raise typer.BadParameter(str(exc), param_hint="evaluation.validator") from exc

    # Same fast-fail for metrics: an unregistered name is a config error, not a
    # crash after the checkpoint load.
    try:
        _resolve_metrics(metrics)
    except KeyError as exc:
        raise typer.BadParameter(str(exc), param_hint="evaluation.metrics") from exc

    token_budget = (
        max_input_tokens
        if max_input_tokens is not None
        else md.packaging.max_input_tokens
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
        max_input_tokens=token_budget,
        baseline_path=baseline,
        limit=limit,
        task=md.task,
        enforce_gates=gate,
    )
    write_markdown_summary(report, out_path.with_suffix(".md"))

    # Record gate results on the run registry when checkpoint is a known run_id.
    run_rec = get_run(md, ckpt.name)
    if run_rec is not None and report.gates is not None:
        update_run_gates(
            md,
            run_rec.run_id,
            report.gates,
            metrics=report.metrics,
        )

    console.print(f"[green]done[/] report={out_path}")
    if gate and report.passed is False:
        console.print("[red]eval gates failed[/]")
        raise typer.Exit(code=1)


@app.command("export")
def cmd_export(
    model_dir: Path = typer.Argument(..., exists=True, file_okay=False),
    checkpoint: Optional[str] = typer.Option(
        None,
        "--checkpoint",
        help="run_id or checkpoint dir; defaults to latest completed run",
    ),
    format: Optional[str] = typer.Option(
        None,
        "--format",
        help="safetensors (default) | gguf | mlx | onnx (plugin-registered)",
    ),
    out: Optional[Path] = typer.Option(
        None,
        "--out",
        help="Export directory (default: <model-dir>/output/export/<run_id>)",
    ),
    parity: bool = typer.Option(
        False,
        "--parity",
        help="Re-run evaluation.gates on dataset.benchmark_samples after export",
    ),
    device: str = typer.Option("auto", "--device"),
) -> None:
    """Export a checkpoint as a deployable bundle with manifest.json."""
    md = load_model_def(model_dir)
    _boot_plugins(md)
    from .export.bundle import export_model, run_parity_check
    from .runs import get_run

    ckpt = resolve_checkpoint(md, checkpoint)
    run_rec = get_run(md, ckpt.name)
    run_id = run_rec.run_id if run_rec else ckpt.name
    out_dir = out if out is not None else (md.output_dir / "export" / run_id)
    try:
        export_model(md, ckpt, out_dir, format=format, run_id=run_id)
    except (ImportError, RuntimeError, ValueError, KeyError, FileNotFoundError) as exc:
        console.print(f"[red]export failed[/] {exc}")
        raise typer.Exit(code=1) from exc
    console.print(f"[green]exported[/] {out_dir}")

    if parity:
        result = run_parity_check(md, out_dir, device=device)
        if result.get("skipped"):
            console.print(f"[yellow]parity skipped[/] {result.get('reason')}")
        elif not result.get("passed"):
            console.print(f"[red]parity gates failed[/] {result.get('gates')}")
            raise typer.Exit(code=1)
        else:
            console.print("[green]parity ok[/]")


@app.command("verify")
def cmd_verify(
    path: Path = typer.Argument(
        ...,
        help="Export directory or path to manifest.json",
    ),
) -> None:
    """Recompute sha256 of files listed in an export manifest; exit 1 on mismatch."""
    from .export.manifest import verify_manifest

    errors = verify_manifest(path)
    if errors:
        console.print("[red]verify failed[/]")
        for err in errors:
            console.print(f"  - {err}")
        raise typer.Exit(code=1)
    console.print(f"[green]OK[/] {path}")


@app.command("serve")
def cmd_serve(
    model_dir: Path = typer.Argument(..., exists=True, file_okay=False),
    checkpoint: Optional[str] = typer.Option(
        None,
        "--checkpoint",
        help="run_id, checkpoint dir, or export dir; defaults to latest completed run",
    ),
    host: str = typer.Option("127.0.0.1", "--host", help="Bind address"),
    port: int = typer.Option(8080, "--port", help="Bind port"),
    device: str = typer.Option("auto", "--device"),
    cors: Optional[str] = typer.Option(
        None,
        "--cors",
        help=(
            "Enable CORS for this origin ('*' or e.g. https://app.example.com). "
            "Off by default; also reads MAATML_SERVE_CORS."
        ),
    ),
    max_body_bytes: int = typer.Option(
        1_048_576,
        "--max-body-bytes",
        help="Reject POST bodies larger than this many bytes (default 1 MiB).",
    ),
    enforce: bool = typer.Option(
        False,
        "--enforce",
        help="Return HTTP 422 when the configured validator rejects a prediction "
        "(gate live inference). Off by default (annotate only).",
    ),
    debug: bool = typer.Option(
        False,
        "--debug",
        help="Include the exception message and traceback in 500 responses. "
        "Off by default so internals never leak to clients.",
    ),
) -> None:
    """Serve a checkpoint or export bundle as a JSON inference API.

    Endpoints: GET /health, GET /info, POST /predict (?validate=1 optional).
    Works for any architecture with a registered predictor (text + vision).
    """
    md = load_model_def(model_dir)
    _boot_plugins(md)
    from .serve import run_server

    cors_origin = cors if cors is not None else os.environ.get("MAATML_SERVE_CORS")
    try:
        run_server(
            md,
            checkpoint=checkpoint,
            host=host,
            port=port,
            device=device,
            cors_origin=cors_origin,
            max_body_bytes=max_body_bytes,
            enforce=enforce,
            debug=debug,
        )
    except (FileNotFoundError, KeyError, ImportError, RuntimeError, ValueError) as exc:
        console.print(f"[red]serve failed[/] {exc}")
        raise typer.Exit(code=1) from exc


@app.command("datagen")
def cmd_datagen(
    model_dir: Path = typer.Argument(..., exists=True, file_okay=False),
    target: int = typer.Option(100, "--target", help="Number of accepted samples"),
    seed: int = typer.Option(0, "--seed"),
    out: Optional[Path] = typer.Option(
        None, "--out", help="Override seed JSONL path"
    ),
    teacher: bool = typer.Option(
        False,
        "--teacher",
        help="Use OpenAI-compatible teacher (MAATML_TEACHER_* env); requires [teacher]",
    ),
    append: bool = typer.Option(True, "--append/--no-append"),
    allow_ungated: bool = typer.Option(
        False,
        "--allow-ungated",
        help="Permit datagen with no evaluation.validator; the run and dataset "
        "card are marked UNGATED (rows are NOT validator-gated).",
    ),
) -> None:
    """Generate validator-gated seed rows via a registered generator or teacher."""
    md = load_model_def(model_dir)
    _boot_plugins(md)
    from .data.datagen import DatagenConfigError, run_datagen

    try:
        result = run_datagen(
            md,
            target=target,
            seed=seed,
            out_path=out,
            use_teacher=teacher,
            append=append,
            allow_ungated=allow_ungated,
        )
    except DatagenConfigError as exc:
        console.print(f"[red]datagen refused[/] {exc}")
        raise typer.Exit(code=1) from exc
    except (KeyError, ImportError) as exc:
        console.print(f"[red]datagen failed[/] {exc}")
        raise typer.Exit(code=1) from exc
    status = "GATED" if result["gated"] else "UNGATED"
    console.print(
        f"[green]datagen[/] status={status} generator={result['generator']} "
        f"accepted={result['accepted']} rejected={result['rejected']} "
        f"out={result['out_path']}"
    )
    if result["protected_existing"]:
        console.print(
            "[yellow]note[/] nothing accepted; existing seed file left unchanged"
        )


@app.command("ingest")
def cmd_ingest(
    model_dir: Path = typer.Argument(..., exists=True, file_okay=False),
    input_path: Path = typer.Option(
        ...,
        "--input",
        exists=True,
        help="JSON or JSONL file to ingest",
    ),
    field_map: Optional[list[str]] = typer.Option(
        None,
        "--map",
        help="Map dest=src fields (repeatable), e.g. --map request=text",
    ),
    sanitize: Optional[str] = typer.Option(
        None, "--sanitize", help="Sanitizer registry tag (e.g. jcl, spool)"
    ),
    append: bool = typer.Option(True, "--append/--no-append"),
    out: Optional[Path] = typer.Option(None, "--out", help="Override seed JSONL path"),
) -> None:
    """Ingest external samples into seed_samples with optional validation."""
    md = load_model_def(model_dir)
    _boot_plugins(md)
    from .data.ingest import ingest_samples

    try:
        result = ingest_samples(
            md,
            input_path,
            field_map=_parse_field_maps(field_map),
            sanitize_tag=sanitize,
            append=append,
            out_path=out,
        )
    except (KeyError, ValueError, FileNotFoundError) as exc:
        console.print(f"[red]ingest failed[/] {exc}")
        raise typer.Exit(code=1) from exc
    console.print(
        f"[green]ingest[/] accepted={result['accepted']} "
        f"rejected={result['rejected']} "
        f"skipped_unvalidated={result['skipped_unvalidated']} "
        f"seeds={result['seeds_path']}"
    )


@app.command("runs")
def cmd_runs(
    model_dir: Path = typer.Argument(..., exists=True, file_okay=False),
) -> None:
    """List training runs recorded in <model-dir>/output/runs.jsonl."""
    md = load_model_def(model_dir)
    records = list_runs(md)
    if not records:
        console.print(f"[dim]no runs yet under {md.output_dir}[/]")
        return
    for rec in records:
        metrics = ""
        if rec.metrics:
            top = list(rec.metrics.items())[:3]
            metrics = " " + " ".join(f"{k}={v:.4g}" for k, v in top)
        console.print(
            f"{rec.run_id}  [{rec.status}]  smoke={rec.smoke}  "
            f"device={rec.device or '-'}  {rec.out_dir}{metrics}"
        )


@app.command("scaffold")
def cmd_scaffold(
    target_dir: Path = typer.Argument(..., help="Directory to create (model folder)"),
    architecture: str = typer.Option(
        ...,
        "--architecture",
        "-a",
        help="Registered trainer architecture (e.g. causal_sft, seq2seq, classifier, dpo)",
    ),
    name: Optional[str] = typer.Option(None, "--name", help="Model name (default: folder name)"),
    force: bool = typer.Option(
        False,
        "--force",
        "-f",
        help="Overwrite an existing model.yml and seed_samples.jsonl",
    ),
) -> None:
    """Create a new model folder with model.yml, datasets, and README."""
    discover_plugins()
    try:
        path = scaffold_model(
            target_dir, architecture=architecture, name=name, force=force
        )
    except FileExistsError as exc:
        console.print(f"[red]scaffold refused[/] {exc}")
        raise typer.Exit(code=1) from exc
    console.print(f"[green]scaffolded[/] {path}")


@app.command("validate")
def cmd_validate(
    model_dir: Path = typer.Argument(..., exists=True, file_okay=False),
    no_plugins: bool = typer.Option(
        False,
        "--no-plugins",
        help="Validate schema and paths without importing trainer or model "
        "plugin code (a model folder is executable code).",
    ),
) -> None:
    """Validate model.yml paths, architecture registration, and dataset.format."""
    errors = validate_model_dir(model_dir, load_plugins=not no_plugins)
    if errors:
        console.print("[red]validate failed[/]")
        for err in errors:
            console.print(f"  - {err}")
        raise typer.Exit(code=1)
    md = load_model_def(model_dir, load_plugins=not no_plugins)
    for warn in config_key_warnings(md):
        console.print(f"[yellow]warning[/] {warn}")
    if no_plugins:
        console.print(
            "[dim]architecture and dataset.format not verified (--no-plugins)[/]"
        )
    console.print(f"[green]OK[/] {md.identity} ({md.architecture})")


@app.command("plan")
def cmd_plan(
    model_dir: Path = typer.Argument(..., exists=True, file_okay=False),
) -> None:
    """Print the lifecycle command plan for a model folder."""
    md = load_model_def(model_dir)
    console.print(f"[bold]{md.identity}[/] ({md.task} / {md.architecture})")
    console.print(f"1. maatml prepare {md.model_dir}")
    console.print(f"2. maatml train {md.model_dir} --smoke")
    console.print(f"3. maatml train {md.model_dir}")
    console.print(f"4. maatml evaluate {md.model_dir}")
    console.print(f"5. maatml export {md.model_dir}")
    console.print(f"6. maatml verify {md.model_dir}/output/export/<run_id>")
    console.print(
        f"7. maatml serve {md.model_dir} --checkpoint output/export/<run_id>"
    )


@app.command("plugins")
def cmd_plugins() -> None:
    """List registered trainers, validators, metrics, formats, and predictors."""
    discover_plugins()
    # Import submodule directly so listing plugins does not require torch.
    from .evaluation import predictors as _predictors  # noqa: F401
    from .export import bundle as _bundle  # noqa: F401
    from .export import gguf as _gguf  # noqa: F401
    from .export import mlx_export as _mlx  # noqa: F401

    for kind, entries in list_all_plugins().items():
        console.print(f"[bold]{kind}[/] ({len(entries)})")
        for entry in entries:
            console.print(f"  {entry.name}  [dim]{entry.source}[/]")


def main() -> None:
    app()


if __name__ == "__main__":
    main()
