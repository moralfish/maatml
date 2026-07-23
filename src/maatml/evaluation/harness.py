"""Shared evaluation harness used by task evaluators and the CLI.

Predictors emit raw text; validators gate structure; metrics plugins score
semantics. Asset paths (schema, contracts, prompt_spec, tokenizer) resolve
from ``model_def`` / explicit kwargs / ``checkpoint_dir``: never from a
hardcoded repo-relative fallback.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional, Sequence, Union

from pydantic import BaseModel, ConfigDict, Field
from rich.console import Console

from ..config import ModelDefinition, get_dataset_cfg
from ..device import resolve_device
from ..registry import METRICS, PREDICTORS, VALIDATORS, discover_plugins
from ..utils.io import iter_jsonl, write_json
from ..validation.base import ValidationResult

console = Console()


class LatencyStats(BaseModel):
    model_config = ConfigDict(extra="forbid")
    p50: float
    p95: float
    mean: float
    n: int


class Report(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model_id: str = ""
    name: str = ""
    version: str = ""
    task: str = ""
    dataset: str = ""
    n: int = 0
    metrics: dict[str, float] = Field(default_factory=dict)
    per_class: dict[str, dict[str, float]] = Field(default_factory=dict)
    latency_ms: Optional[LatencyStats] = None
    baseline_delta: Optional[dict[str, float]] = None
    sample_failures: list[dict] = Field(default_factory=list)
    extras: dict[str, Any] = Field(default_factory=dict)
    # Eval gates (evaluation.gates + --gate)
    gates: Optional[dict[str, Any]] = None
    passed: Optional[bool] = None

    def write(self, path: str | Path) -> Path:
        return write_json(path, self.model_dump(mode="json"))

    @classmethod
    def read(cls, path: str | Path) -> "Report":
        return cls.model_validate_json(Path(path).read_text(encoding="utf-8"))


class GateConfigError(ValueError):
    """Raised when gate enforcement is requested but no gates are configured."""


def resolve_gate_spec(model_def: Any) -> dict[str, float]:
    """Return the configured gate minima, or raise if none are set.

    ``evaluate --gate`` (and ``enforce_gates=True``) must not pass vacuously: a
    model with no ``evaluation.gates`` has nothing to enforce, so requesting
    enforcement against an empty spec is a configuration error rather than a
    silent success.
    """
    evaluation = getattr(model_def, "evaluation", None)
    gate_spec = evaluation.get("gates") if isinstance(evaluation, dict) else None
    if not (isinstance(gate_spec, dict) and gate_spec):
        raise GateConfigError(
            "gate enforcement requested but no evaluation.gates are configured. "
            "Add a gates: block to model.yml (see any example) or drop --gate."
        )
    return {str(k): float(v) for k, v in gate_spec.items()}


def check_gates(
    metrics: dict[str, float],
    gates: dict[str, float],
) -> dict[str, Any]:
    """Compare metrics against minimum thresholds.

    Returns a dict with ``passed`` (bool) and ``results`` mapping each gate
    name to ``{minimum, actual, passed}``.
    """
    results: dict[str, dict[str, Any]] = {}
    all_ok = True
    for name, minimum in gates.items():
        actual = metrics.get(name)
        ok = actual is not None and float(actual) >= float(minimum)
        results[name] = {
            "minimum": float(minimum),
            "actual": None if actual is None else float(actual),
            "passed": ok,
        }
        if not ok:
            all_ok = False
    return {"passed": all_ok, "results": results}


@dataclass
class RowEval:
    """One evaluated row: gold sample, model text, validator outcome."""

    row: dict
    gen_text: str
    result: ValidationResult
    latency_ms: float = 0.0


@dataclass
class _EvalCtx:
    schema_path: Optional[Path] = None
    contracts_path: Optional[Path] = None
    prompt_spec_path: Optional[Path] = None
    extras: dict[str, Any] = field(default_factory=dict)


PredictorLike = Union[str, Any]
ValidatorLike = Union[str, Callable[..., ValidationResult]]
MetricsLike = Union[str, Callable[..., dict[str, float]]]


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    k = (len(s) - 1) * pct
    f = int(k)
    c = min(f + 1, len(s) - 1)
    if f == c:
        return s[f]
    return s[f] + (s[c] - s[f]) * (k - f)


def latency_stats(samples_ms: list[float]) -> LatencyStats:
    return LatencyStats(
        p50=percentile(samples_ms, 0.5),
        p95=percentile(samples_ms, 0.95),
        mean=sum(samples_ms) / len(samples_ms) if samples_ms else 0.0,
        n=len(samples_ms),
    )


def binary_prf(tp: int, fp: int, fn: int) -> dict[str, float]:
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f = 2 * p * r / (p + r) if (p + r) else 0.0
    return {"precision": p, "recall": r, "f1": f, "support": float(tp + fn)}


def per_class_prf(
    true: list[str], pred: list[str], labels: list[str]
) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for label in labels:
        tp = sum(1 for t, p in zip(true, pred) if t == label and p == label)
        fp = sum(1 for t, p in zip(true, pred) if t != label and p == label)
        fn = sum(1 for t, p in zip(true, pred) if t == label and p != label)
        out[label] = binary_prf(tp, fp, fn)
    return out


def baseline_delta(
    metrics: dict[str, float], baseline_path: Optional[str | Path]
) -> Optional[dict[str, float]]:
    if not baseline_path:
        return None
    base = Report.read(baseline_path)
    delta: dict[str, float] = {}
    for k, v in metrics.items():
        if k in base.metrics:
            delta[k] = v - base.metrics[k]
    return delta


def resolve_eval_asset(
    key: str,
    *,
    model_def: Optional[ModelDefinition] = None,
    checkpoint_dir: Path,
    filenames: Sequence[str] = (),
    explicit: Optional[str | Path] = None,
) -> Path:
    """Resolve schema/contracts/prompt_spec/tokenizer without repo fallbacks.

    Order: explicit path → ``model_def`` dataset/data key → file under checkpoint.
    """
    if explicit is not None:
        path = Path(explicit)
        if not path.is_file():
            raise FileNotFoundError(f"{key} not found at explicit path: {path}")
        return path.resolve()

    if model_def is not None:
        cfg = get_dataset_cfg(model_def)
        if key in cfg and isinstance(cfg[key], str):
            path = model_def.resolve(cfg[key])
            if not path.is_file():
                raise FileNotFoundError(
                    f"model.yml declares {key}={cfg[key]!r} but file missing: {path}"
                )
            return path

    checkpoint_dir = Path(checkpoint_dir)
    for name in filenames:
        path = checkpoint_dir / name
        if path.is_file():
            return path.resolve()

    hints = list(filenames) if filenames else ["(none)"]
    raise FileNotFoundError(
        f"Could not resolve {key!r}. Provide model_def with data/dataset.{key}, "
        f"pass an explicit path, or place one of {hints} under {checkpoint_dir}."
    )


def _resolve_callable(kind: str, value: Any, registry) -> Any:
    if isinstance(value, str):
        return registry.require(value)
    if value is None:
        raise KeyError(f"No {kind} provided")
    return value


def _noop_validate(
    raw_output: str,
    *,
    schema_path: str | Path | None = None,
    contracts_path: str | Path | None = None,
    user_prompt: Optional[str] = None,
    strip_fences: bool = True,
) -> ValidationResult:
    del schema_path, contracts_path, user_prompt, strip_fences
    import json

    from ..validation.base import strip_fences as _strip

    text = _strip(raw_output)
    result = ValidationResult(raw_output=raw_output, n_layers=1, required_layers={1})
    try:
        result.parsed = json.loads(text)
        result.passed_layers.add(1)
    except json.JSONDecodeError as exc:
        from ..validation.base import ValidationError

        result.errors.append(
            ValidationError(layer=1, code="invalid_json", message=str(exc))
        )
    return result


def _category_buckets(row_results: list[RowEval]) -> dict[str, dict[str, float]]:
    per_category: dict[str, dict[str, int]] = {}
    for item in row_results:
        category = str(item.row.get("category") or "unknown")
        bucket = per_category.setdefault(category, {"n": 0, "passed_all": 0})
        bucket["n"] += 1
        if item.result.ok:
            bucket["passed_all"] += 1
    return {
        cat: {
            "precision": b["passed_all"] / max(1, b["n"]),
            "recall": 1.0,
            "f1": 0.0,
            "support": float(b["n"]),
        }
        for cat, b in per_category.items()
    }


def run_evaluation(
    *,
    checkpoint_dir: Path,
    dataset_dir: Path,
    out_path: Path,
    model_def: Optional[ModelDefinition] = None,
    predictor: PredictorLike,
    validator: Optional[ValidatorLike] = None,
    metrics_fn: Optional[MetricsLike] = None,
    device: str = "auto",
    split: str = "test",
    max_input_tokens: int = 2048,
    baseline_path: Optional[Path] = None,
    failures_to_keep: int = 20,
    limit: Optional[int] = None,
    schema_path: Optional[Path] = None,
    contracts_path: Optional[Path] = None,
    prompt_spec_path: Optional[Path] = None,
    task: Optional[str] = None,
    enforce_gates: bool = False,
) -> Report:
    """Run the shared eval loop and write a :class:`Report` JSON."""
    discover_plugins()

    checkpoint_dir = Path(checkpoint_dir)
    dataset_dir = Path(dataset_dir)
    out_path = Path(out_path)

    rows = list(iter_jsonl(dataset_dir / f"{split}.jsonl"))
    if not rows:
        raise ValueError(f"No rows in {dataset_dir / f'{split}.jsonl'}")
    if limit is not None and limit > 0:
        rows = rows[:limit]

    target_device = resolve_device(device)

    pred_obj = _resolve_callable("predictor", predictor, PREDICTORS)
    if isinstance(pred_obj, type):
        pred_obj = pred_obj()

    resolved_schema = None
    resolved_contracts = None
    resolved_prompt = None
    try:
        resolved_schema = resolve_eval_asset(
            "schema",
            model_def=model_def,
            checkpoint_dir=checkpoint_dir,
            filenames=(
                "schema.json",
                "jcl_validation_schema.json",
                "spool_interpretation_schema.json",
            ),
            explicit=schema_path,
        )
    except FileNotFoundError:
        if schema_path is not None:
            raise
        # Causal-SFT / no-validator paths may omit schema.
        resolved_schema = None

    try:
        resolved_contracts = resolve_eval_asset(
            "contracts",
            model_def=model_def,
            checkpoint_dir=checkpoint_dir,
            filenames=("node_contracts.json",),
            explicit=contracts_path,
        )
    except FileNotFoundError:
        if contracts_path is not None:
            raise
        resolved_contracts = None

    try:
        resolved_prompt = resolve_eval_asset(
            "prompt_spec",
            model_def=model_def,
            checkpoint_dir=checkpoint_dir,
            filenames=("prompt_spec.json",),
            explicit=prompt_spec_path,
        )
    except FileNotFoundError:
        if prompt_spec_path is not None:
            raise
        resolved_prompt = None

    setup = getattr(pred_obj, "setup", None)
    if callable(setup):
        setup(
            checkpoint_dir,
            model_def=model_def,
            device=target_device,
            max_input_tokens=max_input_tokens,
            schema_path=resolved_schema,
            contracts_path=resolved_contracts,
            prompt_spec_path=resolved_prompt,
        )

    if validator is None:
        validate_fn: Callable[..., ValidationResult] = _noop_validate
    else:
        validate_fn = _resolve_callable("validator", validator, VALIDATORS)
        # Task validators need schema when declared; contracts are optional
        # (text models like JCL/spool declare them; vision may not).
        if resolved_schema is None:
            raise FileNotFoundError(
                "Evaluator requires a schema file. Set data/dataset.schema in "
                "model.yml, pass schema_path=, or place schema.json under the checkpoint."
            )

    metrics_callable: Optional[Callable[..., dict[str, float]]] = None
    if metrics_fn is not None:
        metrics_callable = _resolve_callable("metrics", metrics_fn, METRICS)

    row_results: list[RowEval] = []
    failures: list[dict] = []
    timings: list[float] = []

    request_field = "request"
    if model_def is not None:
        cfg = get_dataset_cfg(model_def)
        request_field = str(
            cfg.get("request_field") or cfg.get("raw_field") or "request"
        )

    predict = pred_obj.predict if hasattr(pred_obj, "predict") else pred_obj

    for row in rows:
        t0 = time.perf_counter()
        gen_text = predict(row)
        if target_device.type == "mps":
            import torch

            torch.mps.synchronize()
        elif target_device.type == "cuda":
            import torch

            torch.cuda.synchronize()
        elapsed = (time.perf_counter() - t0) * 1000.0
        timings.append(elapsed)

        user_prompt = row.get(request_field)
        if not isinstance(user_prompt, str):
            # Fall back to classic text field when request_field is an image path.
            alt = row.get("request")
            user_prompt = alt if isinstance(alt, str) else None

        if resolved_schema is None and resolved_contracts is None:
            result = validate_fn(gen_text, user_prompt=user_prompt)
        else:
            kwargs: dict[str, Any] = {"user_prompt": user_prompt}
            if resolved_schema is not None:
                kwargs["schema_path"] = resolved_schema
            if resolved_contracts is not None:
                kwargs["contracts_path"] = resolved_contracts
            result = validate_fn(gen_text, **kwargs)

        item = RowEval(row=row, gen_text=gen_text, result=result, latency_ms=elapsed)
        row_results.append(item)

        if not result.ok and len(failures) < failures_to_keep:
            input_val = row.get(request_field, row.get("request"))
            if isinstance(input_val, str):
                input_preview: Any = input_val[:500]
            else:
                input_preview = input_val
            failures.append(
                {
                    "sample_id": row.get("sample_id"),
                    "category": row.get("category"),
                    "request": input_preview,
                    request_field: input_preview,
                    "raw_output": gen_text[:1500],
                    "errors": [
                        {
                            "layer": e.layer,
                            "code": e.code,
                            "message": e.message,
                            "location": e.location,
                        }
                        for e in result.errors
                    ],
                }
            )

    n = len(row_results)
    if metrics_callable is not None:
        metrics = metrics_callable(row_results)
    else:
        # Layer-pass rates only.
        layer_pass: dict[int, int] = {}
        all_ok = 0
        for item in row_results:
            for layer in item.result.passed_layers:
                layer_pass[layer] = layer_pass.get(layer, 0) + 1
            if item.result.ok:
                all_ok += 1
        metrics = {
            f"layer_{k}_pass_rate": layer_pass.get(k, 0) / n for k in sorted(layer_pass)
        }
        metrics["all_layers_pass_rate"] = all_ok / n if n else 0.0

    per_class = _category_buckets(row_results)

    identity_name = model_def.name if model_def else checkpoint_dir.name
    identity_version = model_def.version if model_def else ""
    identity_id = (
        model_def.model_id
        if model_def
        else str(checkpoint_dir)
    )
    report_task = task or (model_def.task if model_def else "")

    gates_payload: Optional[dict[str, Any]] = None
    passed: Optional[bool] = None
    if enforce_gates:
        # Raises GateConfigError when no gates are configured, enforcement must
        # never pass vacuously.
        gate_spec = resolve_gate_spec(model_def)
        gates_payload = check_gates(metrics, gate_spec)
        passed = bool(gates_payload["passed"])

    report = Report(
        model_id=identity_id,
        name=identity_name,
        version=identity_version,
        task=report_task,
        dataset=str(dataset_dir / f"{split}.jsonl"),
        n=n,
        metrics=metrics,
        per_class=per_class,
        latency_ms=latency_stats(timings),
        baseline_delta=baseline_delta(metrics, baseline_path),
        sample_failures=failures,
        gates=gates_payload,
        passed=passed,
    )
    report.write(out_path)
    console.print(
        f"[green]eval[/] {report.task or report.name}: n={n} "
        + " ".join(f"{k}={v:.3f}" for k, v in list(metrics.items())[:4])
    )
    return report
