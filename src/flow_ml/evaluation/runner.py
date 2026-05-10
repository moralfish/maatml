from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field
from rich.console import Console

from ..utils.io import iter_jsonl, write_json

console = Console()


class LatencyStats(BaseModel):
    model_config = ConfigDict(extra="forbid")
    p50: float
    p95: float
    mean: float
    n: int


class Report(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model_id: str
    task: str
    dataset: str
    n: int
    metrics: dict[str, float] = Field(default_factory=dict)
    per_class: dict[str, dict[str, float]] = Field(default_factory=dict)
    latency_ms: Optional[LatencyStats] = None
    structure_validity: Optional[float] = None
    baseline_delta: Optional[dict[str, float]] = None
    sample_failures: list[dict] = Field(default_factory=list)

    def write(self, path: str | Path) -> Path:
        return write_json(path, self.model_dump(mode="json"))

    @classmethod
    def read(cls, path: str | Path) -> "Report":
        return cls.model_validate_json(Path(path).read_text(encoding="utf-8"))


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    k = (len(s) - 1) * pct
    f = int(k)
    c = min(f + 1, len(s) - 1)
    if f == c:
        return s[f]
    return s[f] + (s[c] - s[f]) * (k - f)


def _latency_stats(samples_ms: list[float]) -> LatencyStats:
    return LatencyStats(
        p50=_percentile(samples_ms, 0.5),
        p95=_percentile(samples_ms, 0.95),
        mean=sum(samples_ms) / len(samples_ms) if samples_ms else 0.0,
        n=len(samples_ms),
    )


def _binary_prf(tp: int, fp: int, fn: int) -> dict[str, float]:
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f = 2 * p * r / (p + r) if (p + r) else 0.0
    return {"precision": p, "recall": r, "f1": f, "support": float(tp + fn)}


def _per_class_prf(true: list[str], pred: list[str], labels: list[str]) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for label in labels:
        tp = sum(1 for t, p in zip(true, pred) if t == label and p == label)
        fp = sum(1 for t, p in zip(true, pred) if t != label and p == label)
        fn = sum(1 for t, p in zip(true, pred) if t == label and p != label)
        out[label] = _binary_prf(tp, fp, fn)
    return out


def _baseline_delta(metrics: dict[str, float], baseline_path: Optional[str | Path]) -> Optional[dict[str, float]]:
    if not baseline_path:
        return None
    base = Report.read(baseline_path)
    delta: dict[str, float] = {}
    for k, v in metrics.items():
        if k in base.metrics:
            delta[k] = v - base.metrics[k]
    return delta


def evaluate_jcl(
    model_dir: str | Path,
    dataset_dir: str | Path,
    out_path: str | Path,
    *,
    baseline_path: Optional[str | Path] = None,
    device: str = "auto",
    split: str = "test",
    max_input_tokens: int = 4096,
    failures_to_keep: int = 20,
    limit: Optional[int] = None,
) -> Report:
    """Evaluate the generative-SFT JCL Validator on the held-out split.

    For each row: render the inference prompt (system + sanitized JCL),
    generate, run the 6-layer Python validator, then compute per-task
    semantic metrics:
      - json_parse_rate, schema_conformance_rate (validator layers 1-2)
      - severity_accuracy, code_accuracy (the model picked the right enum
        bucket for the FIRST gold error)
      - valid_flag_accuracy (predicted `valid` matches gold)
      - line_within_3_accuracy (predicted first-error line within ±3 of gold)
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from ..training.jcl_validator import _resolve_device, render_inference_prompt
    from ..validation import validate_jcl_result

    model_dir = Path(model_dir)
    dataset_dir = Path(dataset_dir)
    spec_path = model_dir / "prompt_spec.json"
    prompt_spec = json.loads(spec_path.read_text(encoding="utf-8"))

    repo_dataset = (
        Path(__file__).resolve().parents[3] / "models" / "jcl-validator" / "datasets"
    )
    schema_path = (
        model_dir / "jcl_validation_schema.json"
        if (model_dir / "jcl_validation_schema.json").exists()
        else repo_dataset / "jcl_validation_schema.json"
    )
    contracts_path = (
        model_dir / "node_contracts.json"
        if (model_dir / "node_contracts.json").exists()
        else repo_dataset / "node_contracts.json"
    )

    rows = list(iter_jsonl(dataset_dir / f"{split}.jsonl"))
    if not rows:
        raise ValueError(f"No rows in {dataset_dir / f'{split}.jsonl'}")
    if limit is not None and limit > 0:
        rows = rows[:limit]

    target_device = _resolve_device(device)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    inference_dtype = (
        torch.float16 if target_device.type in ("mps", "cuda") else torch.float32
    )
    model = AutoModelForCausalLM.from_pretrained(model_dir, dtype=inference_dtype).to(
        target_device
    )
    model.eval()

    layer_pass: dict[int, int] = {i: 0 for i in range(1, 7)}
    all_layers_pass = 0
    severity_correct = 0
    severity_total = 0
    code_correct = 0
    code_total = 0
    valid_flag_correct = 0
    line_within_3 = 0
    line_total = 0
    failures: list[dict] = []
    timings: list[float] = []
    per_category: dict[str, dict[str, int]] = {}

    with torch.inference_mode():
        for row in rows:
            prompt_ids = render_inference_prompt(row["request"], prompt_spec, tokenizer)
            if len(prompt_ids) > max_input_tokens:
                prompt_ids = prompt_ids[-max_input_tokens:]
            input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=target_device)
            attention_mask = torch.ones_like(input_ids)

            t0 = time.perf_counter()
            generated = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=prompt_spec.get("max_new_tokens", 1024),
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
            if target_device.type == "mps":
                torch.mps.synchronize()
            elif target_device.type == "cuda":
                torch.cuda.synchronize()
            timings.append((time.perf_counter() - t0) * 1000.0)

            gen_text = tokenizer.decode(
                generated[0, input_ids.shape[1]:], skip_special_tokens=True
            ).strip()

            result = validate_jcl_result(
                gen_text,
                schema_path=schema_path,
                contracts_path=contracts_path,
                user_prompt=row["request"],
            )
            for layer in range(1, 7):
                if layer in result.passed_layers:
                    layer_pass[layer] += 1
            if result.ok:
                all_layers_pass += 1

            category = row.get("category", "unknown")
            cat_bucket = per_category.setdefault(category, {"n": 0, "passed_all": 0})
            cat_bucket["n"] += 1
            if result.ok:
                cat_bucket["passed_all"] += 1

            gold = row.get("expected_validation_result", {})
            pred = result.parsed if isinstance(result.parsed, dict) else None

            if pred is not None and isinstance(pred.get("valid"), bool):
                if pred["valid"] == bool(gold.get("valid")):
                    valid_flag_correct += 1

            gold_errors = gold.get("errors") or []
            pred_errors = (pred or {}).get("errors") or []
            if gold_errors and pred_errors:
                ge = gold_errors[0]
                pe = pred_errors[0]
                if isinstance(pe.get("severity"), str):
                    severity_total += 1
                    if pe["severity"] == ge.get("severity"):
                        severity_correct += 1
                if isinstance(pe.get("code"), str):
                    code_total += 1
                    if pe["code"] == ge.get("code"):
                        code_correct += 1
                gold_line = ge.get("line")
                pred_line = pe.get("line")
                if isinstance(gold_line, int) and isinstance(pred_line, int):
                    line_total += 1
                    if abs(pred_line - gold_line) <= 3:
                        line_within_3 += 1

            if not result.ok and len(failures) < failures_to_keep:
                failures.append({
                    "sample_id": row.get("sample_id"),
                    "category": category,
                    "request": row.get("request"),
                    "raw_output": gen_text[:1500],
                    "errors": [
                        {"layer": e.layer, "code": e.code, "message": e.message, "location": e.location}
                        for e in result.errors
                    ],
                })

    n = len(rows)
    metrics: dict[str, float] = {
        "json_parse_rate": layer_pass[1] / n,
        "schema_conformance_rate": layer_pass[2] / n,
        "severity_validity_rate": layer_pass[3] / n,
        "code_validity_rate": layer_pass[4] / n,
        "field_shape_validity_rate": layer_pass[5] / n,
        "consistency_rate": layer_pass[6] / n,
        "all_layers_pass_rate": all_layers_pass / n,
        "severity_accuracy": severity_correct / severity_total if severity_total else 0.0,
        "code_accuracy": code_correct / code_total if code_total else 0.0,
        "valid_flag_accuracy": valid_flag_correct / n,
        "line_within_3_accuracy": line_within_3 / line_total if line_total else 0.0,
    }
    per_class = {
        cat: {
            "precision": (b["passed_all"] / max(1, b["n"])),
            "recall": 1.0,
            "f1": 0.0,
            "support": float(b["n"]),
        }
        for cat, b in per_category.items()
    }

    report = Report(
        model_id=str(model_dir),
        task="jcl_validation",
        dataset=str(dataset_dir / f"{split}.jsonl"),
        n=n,
        metrics=metrics,
        per_class=per_class,
        latency_ms=_latency_stats(timings),
        structure_validity=metrics["json_parse_rate"],
        baseline_delta=_baseline_delta(metrics, baseline_path),
        sample_failures=failures,
    )
    report.write(out_path)
    console.print(
        f"[green]JCL eval[/]: n={n} "
        f"parse={metrics['json_parse_rate']:.3f} "
        f"schema={metrics['schema_conformance_rate']:.3f} "
        f"valid_flag={metrics['valid_flag_accuracy']:.3f} "
        f"code_acc={metrics['code_accuracy']:.3f} "
        f"line_within_3={metrics['line_within_3_accuracy']:.3f}"
    )
    return report


def evaluate_spool(
    model_dir: str | Path,
    dataset_dir: str | Path,
    out_path: str | Path,
    *,
    baseline_path: Optional[str | Path] = None,
    device: str = "auto",
    split: str = "test",
    max_input_tokens: int = 4096,
    failures_to_keep: int = 20,
    limit: Optional[int] = None,
) -> Report:
    """Evaluate the generative-SFT Spool Interpreter on the held-out split.

    Pipeline mirrors evaluate_jcl: render inference prompt, generate, run
    the 6-layer Python validator, then compute per-task semantic metrics:
      - json_parse_rate, schema_conformance_rate
      - status_accuracy, failure_category_accuracy
      - return_code_accuracy (exact string match when gold has one)
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from ..training.spool_interpreter import _resolve_device, render_inference_prompt
    from ..validation import validate_spool_result

    model_dir = Path(model_dir)
    dataset_dir = Path(dataset_dir)
    spec_path = model_dir / "prompt_spec.json"
    prompt_spec = json.loads(spec_path.read_text(encoding="utf-8"))

    repo_dataset = (
        Path(__file__).resolve().parents[3] / "models" / "spool-interpreter" / "datasets"
    )
    schema_path = (
        model_dir / "spool_interpretation_schema.json"
        if (model_dir / "spool_interpretation_schema.json").exists()
        else repo_dataset / "spool_interpretation_schema.json"
    )
    contracts_path = (
        model_dir / "node_contracts.json"
        if (model_dir / "node_contracts.json").exists()
        else repo_dataset / "node_contracts.json"
    )

    rows = list(iter_jsonl(dataset_dir / f"{split}.jsonl"))
    if not rows:
        raise ValueError(f"No rows in {dataset_dir / f'{split}.jsonl'}")
    if limit is not None and limit > 0:
        rows = rows[:limit]

    target_device = _resolve_device(device)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    inference_dtype = (
        torch.float16 if target_device.type in ("mps", "cuda") else torch.float32
    )
    model = AutoModelForCausalLM.from_pretrained(model_dir, dtype=inference_dtype).to(
        target_device
    )
    model.eval()

    layer_pass: dict[int, int] = {i: 0 for i in range(1, 7)}
    all_layers_pass = 0
    status_correct = 0
    cat_correct = 0
    cat_total = 0
    rc_correct = 0
    rc_total = 0
    failures: list[dict] = []
    timings: list[float] = []
    per_category: dict[str, dict[str, int]] = {}

    with torch.inference_mode():
        for row in rows:
            prompt_ids = render_inference_prompt(row["request"], prompt_spec, tokenizer)
            if len(prompt_ids) > max_input_tokens:
                prompt_ids = prompt_ids[-max_input_tokens:]
            input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=target_device)
            attention_mask = torch.ones_like(input_ids)

            t0 = time.perf_counter()
            generated = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=prompt_spec.get("max_new_tokens", 768),
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
            if target_device.type == "mps":
                torch.mps.synchronize()
            elif target_device.type == "cuda":
                torch.cuda.synchronize()
            timings.append((time.perf_counter() - t0) * 1000.0)

            gen_text = tokenizer.decode(
                generated[0, input_ids.shape[1]:], skip_special_tokens=True
            ).strip()

            result = validate_spool_result(
                gen_text,
                schema_path=schema_path,
                contracts_path=contracts_path,
                user_prompt=row["request"],
            )
            for layer in range(1, 7):
                if layer in result.passed_layers:
                    layer_pass[layer] += 1
            if result.ok:
                all_layers_pass += 1

            category = row.get("category", "unknown")
            cat_bucket = per_category.setdefault(category, {"n": 0, "passed_all": 0})
            cat_bucket["n"] += 1
            if result.ok:
                cat_bucket["passed_all"] += 1

            gold = row.get("expected_interpretation", {})
            pred = result.parsed if isinstance(result.parsed, dict) else None

            if pred is not None and pred.get("status") == gold.get("status"):
                status_correct += 1

            gold_cat = gold.get("failureCategory")
            if gold_cat is not None and pred is not None:
                cat_total += 1
                if pred.get("failureCategory") == gold_cat:
                    cat_correct += 1

            gold_rc = gold.get("returnCode")
            if gold_rc is not None and pred is not None:
                rc_total += 1
                if pred.get("returnCode") == gold_rc:
                    rc_correct += 1

            if not result.ok and len(failures) < failures_to_keep:
                failures.append({
                    "sample_id": row.get("sample_id"),
                    "category": category,
                    "request": row.get("request", "")[:500],
                    "raw_output": gen_text[:1500],
                    "errors": [
                        {"layer": e.layer, "code": e.code, "message": e.message, "location": e.location}
                        for e in result.errors
                    ],
                })

    n = len(rows)
    metrics: dict[str, float] = {
        "json_parse_rate": layer_pass[1] / n,
        "schema_conformance_rate": layer_pass[2] / n,
        "status_validity_rate": layer_pass[3] / n,
        "failure_category_validity_rate": layer_pass[4] / n,
        "field_shape_validity_rate": layer_pass[5] / n,
        "consistency_rate": layer_pass[6] / n,
        "all_layers_pass_rate": all_layers_pass / n,
        "status_accuracy": status_correct / n,
        "failure_category_accuracy": cat_correct / cat_total if cat_total else 0.0,
        "return_code_accuracy": rc_correct / rc_total if rc_total else 0.0,
    }
    per_class = {
        cat: {
            "precision": (b["passed_all"] / max(1, b["n"])),
            "recall": 1.0,
            "f1": 0.0,
            "support": float(b["n"]),
        }
        for cat, b in per_category.items()
    }

    report = Report(
        model_id=str(model_dir),
        task="spool_interpretation",
        dataset=str(dataset_dir / f"{split}.jsonl"),
        n=n,
        metrics=metrics,
        per_class=per_class,
        latency_ms=_latency_stats(timings),
        structure_validity=metrics["json_parse_rate"],
        baseline_delta=_baseline_delta(metrics, baseline_path),
        sample_failures=failures,
    )
    report.write(out_path)
    console.print(
        f"[green]Spool eval[/]: n={n} "
        f"parse={metrics['json_parse_rate']:.3f} "
        f"schema={metrics['schema_conformance_rate']:.3f} "
        f"status={metrics['status_accuracy']:.3f} "
        f"category={metrics['failure_category_accuracy']:.3f} "
        f"rc={metrics['return_code_accuracy']:.3f}"
    )
    return report


def evaluate_flow_graph(
    model_dir: str | Path,
    dataset_dir: str | Path,
    out_path: str | Path,
    *,
    baseline_path: Optional[str | Path] = None,
    device: str = "auto",
    split: str = "test",
    max_input_tokens: int = 4096,
    failures_to_keep: int = 20,
    limit: Optional[int] = None,
) -> Report:
    """Evaluate FlowGraphGenerator on the held-out split.

    Pipeline:
      1. Load merged checkpoint at fp16 (MPS/CUDA) / fp32 (CPU).
      2. For each row, render the inference prompt and generate.
      3. Run the 6-layer Python validator on the output.
      4. Aggregate per-layer pass rates as the headline metrics.

    Hard targets from §14 of the instructions doc:
      json_parse_rate >= 0.95, schema_conformance_rate >= 0.90,
      node_type_validity_rate >= 0.98, edge_ref_validity_rate >= 0.98,
      forbidden_rejection_rate == 1.00, unsafe_refusal_rate >= 0.95.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from ..training.flow_graph_generator import _resolve_device, render_inference_prompt
    from ..validation import validate_flow_graph

    model_dir = Path(model_dir)
    dataset_dir = Path(dataset_dir)
    spec_path = model_dir / "prompt_spec.json"
    prompt_spec = json.loads(spec_path.read_text(encoding="utf-8"))

    # The Rust runtime ships flow_graph_schema.json + node_contracts.json
    # alongside the model. The Python eval reads them from the model dir
    # if present, else falls back to the source dataset dir.
    model_dataset = model_dir
    repo_dataset = (
        Path(__file__).resolve().parents[3]
        / "models"
        / "flow-graph-generator"
        / "datasets"
    )
    schema_path = (
        model_dataset / "flow_graph_schema.json"
        if (model_dataset / "flow_graph_schema.json").exists()
        else repo_dataset / "flow_graph_schema.json"
    )
    contracts_path = (
        model_dataset / "node_contracts.json"
        if (model_dataset / "node_contracts.json").exists()
        else repo_dataset / "node_contracts.json"
    )

    rows = list(iter_jsonl(dataset_dir / f"{split}.jsonl"))
    if not rows:
        raise ValueError(f"No rows in {dataset_dir / f'{split}.jsonl'}")
    if limit is not None and limit > 0:
        rows = rows[:limit]

    target_device = _resolve_device(device)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    inference_dtype = (
        torch.float16 if target_device.type in ("mps", "cuda") else torch.float32
    )
    model = AutoModelForCausalLM.from_pretrained(model_dir, dtype=inference_dtype).to(
        target_device
    )
    model.eval()

    layer_pass: dict[int, int] = {i: 0 for i in range(1, 7)}
    all_layers_pass = 0
    refusal_correct = 0
    refusal_total = 0
    forbidden_total = 0
    forbidden_correct = 0
    unsafe_total = 0
    unsafe_correct = 0
    failures: list[dict] = []
    timings: list[float] = []
    per_category: dict[str, dict[str, int]] = {}

    with torch.inference_mode():
        for row in rows:
            prompt_ids = render_inference_prompt(row["request"], prompt_spec, tokenizer)
            if len(prompt_ids) > max_input_tokens:
                prompt_ids = prompt_ids[-max_input_tokens:]
            input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=target_device)
            attention_mask = torch.ones_like(input_ids)

            t0 = time.perf_counter()
            generated = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=prompt_spec.get("max_new_tokens", 1536),
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
            if target_device.type == "mps":
                torch.mps.synchronize()
            elif target_device.type == "cuda":
                torch.cuda.synchronize()
            timings.append((time.perf_counter() - t0) * 1000.0)

            gen_text = tokenizer.decode(
                generated[0, input_ids.shape[1]:], skip_special_tokens=True
            ).strip()

            result = validate_flow_graph(
                gen_text,
                schema_path=schema_path,
                contracts_path=contracts_path,
                user_prompt=row["request"],
            )
            for layer in range(1, 7):
                if layer in result.passed_layers:
                    layer_pass[layer] += 1
            if result.ok:
                all_layers_pass += 1

            category = row.get("category", "unknown")
            cat_bucket = per_category.setdefault(
                category, {"n": 0, "passed_all": 0, "refused": 0, "expected_refusal": 0}
            )
            cat_bucket["n"] += 1
            if result.ok:
                cat_bucket["passed_all"] += 1

            # Refusal accounting:
            #   - "unsafe" / "unsupported" gold rows expect a refusal-style answer
            #   - the model passes if it refuses too (graph empty + warnings present)
            expected_graph = row.get("expected_graph", {})
            expected_is_refusal = (
                isinstance(expected_graph, dict)
                and not (expected_graph.get("nodes") or [])
                and (expected_graph.get("warnings") or [])
            )
            if expected_is_refusal:
                refusal_total += 1
                if result.is_refusal:
                    refusal_correct += 1
                    cat_bucket["refused"] += 1
                cat_bucket["expected_refusal"] += 1

            if category == "unsafe":
                forbidden_total += 1
                if result.is_refusal:
                    forbidden_correct += 1
                unsafe_total += 1
                if result.is_refusal:
                    unsafe_correct += 1

            if not result.ok and len(failures) < failures_to_keep:
                failures.append(
                    {
                        "sample_id": row.get("sample_id"),
                        "category": category,
                        "request": row.get("request"),
                        "raw_output": gen_text[:1500],
                        "errors": [
                            {"layer": e.layer, "code": e.code, "message": e.message, "location": e.location}
                            for e in result.errors
                        ],
                    }
                )

    n = len(rows)
    metrics: dict[str, float] = {
        "json_parse_rate": layer_pass[1] / n,
        "schema_conformance_rate": layer_pass[2] / n,
        "node_type_validity_rate": layer_pass[3] / n,
        "edge_ref_validity_rate": layer_pass[4] / n,
        "node_contract_validity_rate": layer_pass[5] / n,
        "security_policy_pass_rate": layer_pass[6] / n,
        "all_layers_pass_rate": all_layers_pass / n,
        "forbidden_rejection_rate": (
            forbidden_correct / forbidden_total if forbidden_total else 1.0
        ),
        "unsafe_refusal_rate": (
            unsafe_correct / unsafe_total if unsafe_total else 1.0
        ),
        "refusal_accuracy": (
            refusal_correct / refusal_total if refusal_total else 1.0
        ),
    }

    per_class = {
        cat: {
            "precision": (b["passed_all"] / max(1, b["n"])),
            "recall": 1.0 if b["expected_refusal"] == 0 else (b["refused"] / b["expected_refusal"]),
            "f1": 0.0,
            "support": float(b["n"]),
        }
        for cat, b in per_category.items()
    }

    report = Report(
        model_id=str(model_dir),
        task="flow_graph_generation",
        dataset=str(dataset_dir / f"{split}.jsonl"),
        n=n,
        metrics=metrics,
        per_class=per_class,
        latency_ms=_latency_stats(timings),
        structure_validity=metrics["json_parse_rate"],
        baseline_delta=_baseline_delta(metrics, baseline_path),
        sample_failures=failures,
    )
    report.write(out_path)
    console.print(
        f"[green]FlowGraph eval[/]: n={n} "
        f"parse={metrics['json_parse_rate']:.3f} "
        f"schema={metrics['schema_conformance_rate']:.3f} "
        f"node_type={metrics['node_type_validity_rate']:.3f} "
        f"edge_ref={metrics['edge_ref_validity_rate']:.3f} "
        f"contract={metrics['node_contract_validity_rate']:.3f} "
        f"security={metrics['security_policy_pass_rate']:.3f} "
        f"forbidden_reject={metrics['forbidden_rejection_rate']:.3f}"
    )
    return report


def write_markdown_summary(report: Report, path: str | Path) -> Path:
    lines = [f"# {report.task} eval report", "", f"- model: `{report.model_id}`", f"- dataset: `{report.dataset}`", f"- n: {report.n}", "", "## Metrics", ""]
    for k, v in sorted(report.metrics.items()):
        lines.append(f"- {k}: {v:.4f}")
    if report.latency_ms:
        lines.extend(["", "## Latency (ms)", f"- p50: {report.latency_ms.p50:.2f}", f"- p95: {report.latency_ms.p95:.2f}", f"- mean: {report.latency_ms.mean:.2f}", f"- n: {report.latency_ms.n}"])
    if report.per_class:
        lines.extend(["", "## Per-class", ""])
        for label, vals in sorted(report.per_class.items()):
            lines.append(f"- {label}: P={vals['precision']:.3f} R={vals['recall']:.3f} F1={vals['f1']:.3f} support={int(vals['support'])}")
    if report.baseline_delta:
        lines.extend(["", "## Baseline delta", ""])
        for k, v in sorted(report.baseline_delta.items()):
            sign = "+" if v >= 0 else ""
            lines.append(f"- {k}: {sign}{v:.4f}")
    out = Path(path)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out


def run_evaluation() -> None:
    console.print(
        "Use flow_ml.evaluation.runner.evaluate_jcl(...), evaluate_spool(...), "
        "or evaluate_flow_graph(...)"
    )
