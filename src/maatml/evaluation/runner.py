"""Evaluation entry points and report helpers.

Task evaluation goes through :func:`maatml.evaluation.harness.run_evaluation`
(via the CLI). Shared report types live in ``harness`` and are re-exported
here for backward compatibility.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from rich.console import Console

from .harness import (
    LatencyStats,
    Report,
    baseline_delta,
    binary_prf,
    latency_stats,
    per_class_prf,
    percentile,
    run_evaluation,
)

# Backward-compatible private aliases used by tests.
_percentile = percentile
_latency_stats = latency_stats
_binary_prf = binary_prf
_per_class_prf = per_class_prf
_baseline_delta = baseline_delta

console = Console()

__all__ = [
    "LatencyStats",
    "Report",
    "evaluate_model",
    "run_evaluation",
    "write_markdown_summary",
    "_percentile",
    "_latency_stats",
    "_binary_prf",
    "_per_class_prf",
    "_baseline_delta",
]


def evaluate_model(
    model_def,
    *,
    checkpoint: Optional[str] = None,
    split: str = "test",
    device: str = "auto",
    baseline: Optional[Path] = None,
    max_input_tokens: Optional[int] = None,
    limit: Optional[int] = None,
    gate: bool = False,
    smoke: bool = False,
    cache_predictions: Optional[bool] = None,
    strict_population: bool = False,
) -> tuple["Report", Path]:
    """Evaluate a checkpoint of ``model_def`` and write report.{json,md}.

    The single implementation behind ``maatml evaluate`` and the lifecycle
    runner's evaluate step, so both enforce gates, resolve the token budget,
    and record results on the run identically. Configuration problems raise
    before the checkpoint is resolved or loaded. ``cache_predictions`` falls
    back to ``evaluation.cache_predictions`` when not given.
    """
    from ..runs import get_run, resolve_checkpoint, update_run_gates
    from . import predictors as _predictors  # noqa: F401  register built-ins
    from .harness import (
        _resolve_metrics,
        resolve_gate_spec,
        resolve_slices,
        resolve_validator,
        run_evaluation,
        uses_smoke_gates,
    )

    evaluation = model_def.evaluation or {}
    predictor = evaluation.get("predictor")
    validator = evaluation.get("validator")
    metrics = evaluation.get("metrics")
    if isinstance(metrics, list) and not metrics:
        metrics = None

    if predictor is None:
        from ..registry import PREDICTORS
        from ..scaffold import normalize_architecture

        arch = normalize_architecture(model_def.architecture)
        if PREDICTORS.get(model_def.architecture):
            predictor = model_def.architecture
        elif PREDICTORS.get(arch):
            predictor = arch
    if predictor is None:
        raise KeyError(
            f"No predictor for architecture={model_def.architecture!r}; "
            "set evaluation.predictor in model.yml"
        )

    gate_spec = None
    smoke_gated = False
    if gate:
        gate_spec = resolve_gate_spec(model_def, smoke=smoke)
        smoke_gated = smoke and uses_smoke_gates(model_def)
    if validator is not None:
        resolve_validator(validator)
    _resolve_metrics(metrics)
    slices = resolve_slices(model_def)
    if cache_predictions is None:
        cache_predictions = bool(evaluation.get("cache_predictions", False))

    ckpt = resolve_checkpoint(model_def, checkpoint)
    model_def.eval_dir.mkdir(parents=True, exist_ok=True)
    out_path = model_def.eval_dir / f"{ckpt.name}.json"
    budget = (
        max_input_tokens if max_input_tokens is not None else model_def.packaging.max_input_tokens
    )

    report = run_evaluation(
        checkpoint_dir=ckpt,
        dataset_dir=model_def.prepared_dir,
        out_path=out_path,
        model_def=model_def,
        predictor=predictor,
        validator=validator,
        metrics_fn=metrics,
        device=device,
        split=split,
        max_input_tokens=budget,
        baseline_path=baseline,
        limit=limit,
        task=model_def.task,
        enforce_gates=gate,
        gate_spec=gate_spec,
        smoke_gated=smoke_gated,
        slices=slices,
        cache_predictions=cache_predictions,
        strict_population=strict_population,
    )
    write_markdown_summary(report, out_path.with_suffix(".md"))

    run_rec = get_run(model_def, ckpt.name)
    if run_rec is not None and report.gates is not None:
        update_run_gates(
            model_def,
            run_rec.run_id,
            report.gates,
            metrics=report.metrics,
            smoke_gated=smoke_gated,
        )
    return report, out_path


_COUNT_KEYS = frozenset({"n", "support", "passed"})


def _format_class_stats(vals: dict[str, float]) -> str:
    """Render whatever per-class keys the report carries.

    Category buckets report ``pass_rate`` / ``passed`` / ``n``; metrics plugins
    that compute real per-class P/R/F1 report those instead. Neither shape is
    padded with invented keys.
    """
    parts = []
    for key in sorted(vals):
        value = vals[key]
        if key in _COUNT_KEYS:
            parts.append(f"{key}={int(value)}")
        else:
            parts.append(f"{key}={value:.3f}")
    return " ".join(parts)


def write_markdown_summary(report: Report, path: str | Path) -> Path:
    title = report.task or report.name or "eval"
    lines = [
        f"# {title} eval report",
        "",
        f"- model: `{report.model_id}`",
    ]
    if report.name:
        lines.append(f"- name: `{report.name}`")
    if report.version:
        lines.append(f"- version: `{report.version}`")
    lines.extend(
        [
            f"- dataset: `{report.dataset}`",
            f"- n: {report.n}",
            "",
            "## Metrics",
            "",
        ]
    )
    for k, v in sorted(report.metrics.items()):
        lines.append(f"- {k}: {v:.4f}")
    if report.gates is not None:
        lines.extend(["", "## Gates", ""])
        if report.passed is not None:
            lines.append(f"- passed: {report.passed}")
        results = report.gates.get("results") or {}
        for name, info in sorted(results.items()):
            tier = info.get("tier", "blocking")
            suffix = "" if tier == "blocking" else f" tier={tier}"
            lines.append(
                f"- {name}: actual={info.get('actual')} "
                f"minimum={info.get('minimum')} passed={info.get('passed')}{suffix}"
            )
        if report.gates.get("benchmark_sha256"):
            lines.append(f"- benchmark: `{report.gates['benchmark_sha256']}`")
        if report.gates.get("population_mismatch"):
            lines.append(
                f"- floors derived on `{report.gates.get('floors_benchmark_sha256')}` "
                "(population mismatch)"
            )
    if report.latency_ms:
        lines.extend(
            [
                "",
                "## Latency (ms)",
                f"- p50: {report.latency_ms.p50:.2f}",
                f"- p95: {report.latency_ms.p95:.2f}",
                f"- mean: {report.latency_ms.mean:.2f}",
                f"- n: {report.latency_ms.n}",
            ]
        )
    if report.per_class:
        lines.extend(["", "## Per-class", ""])
        for label, vals in sorted(report.per_class.items()):
            lines.append(f"- {label}: {_format_class_stats(vals)}")
    if report.slices:
        lines.extend(["", "## Slices", ""])
        for field_name, values in sorted(report.slices.items()):
            for value, stats in sorted(values.items()):
                n = int(stats.get("n", 0))
                if n == 0:
                    lines.append(f"- {field_name}={value}: n=0 (no rate)")
                    continue
                lines.append(
                    f"- {field_name}={value}: n={n} "
                    f"pass_rate={stats['pass_rate']:.3f} w95={stats['pass_rate_w95']:.3f}"
                )
    if report.baseline_delta:
        lines.extend(["", "## Baseline delta", ""])
        for k, v in sorted(report.baseline_delta.items()):
            sign = "+" if v >= 0 else ""
            lines.append(f"- {k}: {sign}{v:.4f}")
    if report.extras:
        lines.extend(["", "## Extras", ""])
        for k, v in sorted(report.extras.items()):
            lines.append(f"- {k}: {v}")
    if report.sample_failures:
        lines.extend(["", "## Sample failures", ""])
        # Cap the markdown list so a long eval stays readable; the JSON report
        # already keeps the full failures_to_keep budget.
        for failure in report.sample_failures[:20]:
            sample_id = failure.get("sample_id") or "(no sample_id)"
            lines.append(f"- `{sample_id}`")
            for err in failure.get("errors") or []:
                layer = err.get("layer")
                code = err.get("code")
                location = err.get("location")
                message = err.get("message") or ""
                hint = err.get("hint")
                where = f" at `{location}`" if location else ""
                lines.append(f"  - L{layer}/{code}{where}: {message}")
                if hint:
                    lines.append(f"    - fix: {hint}")
    out = Path(path)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out
