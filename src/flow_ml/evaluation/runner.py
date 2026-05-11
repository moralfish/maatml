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
    max_input_tokens: int = 1024,
    failures_to_keep: int = 20,
    limit: Optional[int] = None,
) -> Report:
    """Evaluate the v2 BERT multi-head JCL classifier on the held-out split.

    For each row: pre-tokenize JCL → BPE encode → ModernBERT forward →
    4 head argmax → build `JclValidationResult` via the `error_message_templates`
    block in `node_contracts.json` → run the 6-layer Python validator → aggregate.
    """
    import json as _json
    import torch
    from safetensors.torch import load_file
    from transformers import AutoModel, PreTrainedTokenizerFast

    from ..tokenization import pre_tokenize_jcl
    from ..training.jcl_classifier import ERROR_CODES, SEVERITIES
    from ..training.sft_base import _resolve_device
    from ..validation import validate_jcl_result

    model_dir = Path(model_dir)
    dataset_dir = Path(dataset_dir)

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

    # Tokenizer: prefer the custom JCL BPE; fall back to a sibling `tokenizer.json`
    # in the model dir if the trainer saved one.
    tokenizer_path = model_dir / "tokenizer.json"
    if not tokenizer_path.exists():
        tokenizer_path = repo_dataset / "tokenizer.json"
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_file=str(tokenizer_path),
        model_max_length=max_input_tokens,
        pad_token="<PAD>",
        unk_token="<UNK>",
        cls_token="<CLS>",
        sep_token="<SEP>",
        mask_token="<MASK>",
        additional_special_tokens=["<COL1>", "<CONT>"],
    )

    encoder = AutoModel.from_pretrained(model_dir).to(target_device)
    encoder.eval()
    hidden = encoder.config.hidden_size

    head_state = load_file(model_dir / "classifier_heads.safetensors")
    heads: dict[str, dict[str, torch.Tensor]] = {}
    for name in ("validity", "error_code", "severity", "line"):
        heads[name] = {
            "weight": head_state[f"heads.{name}.weight"].to(target_device),
            "bias": head_state[f"heads.{name}.bias"].to(target_device),
        }

    # node_contracts.json holds the per-code message templates the runtime
    # fills into ValidationResult.errors[].message / .suggestion.
    contracts = _json.loads(contracts_path.read_text(encoding="utf-8"))
    templates = contracts.get("error_message_templates", {})

    def _head_forward(name: str, x: torch.Tensor) -> torch.Tensor:
        h = heads[name]
        return torch.nn.functional.linear(x, h["weight"], h["bias"])

    def _first_error_line(line_logits: torch.Tensor, encoding, pre: str) -> Optional[int]:
        # line_logits: (1, T, 2). Argmax per token → first non-special
        # token where class==1; count `<COL1>` markers up to its char offset.
        cls_per_tok = line_logits.squeeze(0).argmax(dim=-1).tolist()
        tokens = encoding.tokens()
        offsets = encoding.offsets
        for i, label in enumerate(cls_per_tok):
            if label != 1:
                continue
            tok = tokens[i] if i < len(tokens) else ""
            if tok.startswith("<") and tok.endswith(">"):
                continue
            char_offset = offsets[i][0] if i < len(offsets) else 0
            return max(1, pre[:min(char_offset, len(pre))].count("<COL1>"))
        return None

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
            pre = pre_tokenize_jcl(row["request"])
            encoding = tokenizer(
                pre,
                max_length=max_input_tokens,
                truncation=True,
                return_offsets_mapping=True,
                return_tensors=None,
            )
            input_ids = torch.tensor([encoding["input_ids"]], dtype=torch.long, device=target_device)
            attention_mask = torch.tensor(
                [encoding["attention_mask"]], dtype=torch.long, device=target_device
            )

            # Wrap a thin object so `_first_error_line` can read .tokens()/.offsets
            class _Enc:
                def __init__(self, e):
                    self._e = e

                def tokens(self):
                    return self._e.tokens(0) if hasattr(self._e, "tokens") else []

                @property
                def offsets(self):
                    return self._e["offset_mapping"]

            t0 = time.perf_counter()
            out = encoder(input_ids=input_ids, attention_mask=attention_mask)
            seq = out.last_hidden_state
            pooled = seq[:, 0, :]
            validity_logits = _head_forward("validity", pooled)
            error_code_logits = _head_forward("error_code", pooled)
            severity_logits = _head_forward("severity", pooled)
            line_logits = _head_forward("line", seq)
            if target_device.type == "mps":
                torch.mps.synchronize()
            elif target_device.type == "cuda":
                torch.cuda.synchronize()
            timings.append((time.perf_counter() - t0) * 1000.0)

            validity_probs = torch.softmax(validity_logits.squeeze(0), dim=-1)
            valid_idx = int(validity_probs.argmax().item())
            valid_conf = float(validity_probs[valid_idx].item())
            code_idx = int(error_code_logits.squeeze(0).argmax().item())
            severity_idx = int(severity_logits.squeeze(0).argmax().item())

            is_valid = valid_idx == 1
            code = ERROR_CODES[code_idx] if code_idx < len(ERROR_CODES) else "other"
            severity_str = SEVERITIES[severity_idx] if severity_idx < len(SEVERITIES) else "error"

            line_no: Optional[int] = None
            if not is_valid:
                line_no = _first_error_line(line_logits, _Enc(encoding), pre)

            errors_out: list[dict] = []
            if not is_valid:
                tpl = templates.get(code) or templates.get("other") or {
                    "message": f"{code} (no template registered)",
                    "suggestion": "",
                }
                errors_out.append({
                    "line": int(line_no) if line_no else 1,
                    "column": 1,
                    "severity": severity_str if severity_str != "none" else "error",
                    "code": code,
                    "message": tpl.get("message", ""),
                    "suggestion": tpl.get("suggestion") or None,
                })
            pred_json = {
                "valid": bool(is_valid),
                "errors": errors_out,
                "confidence": valid_conf,
            }
            gen_text = _json.dumps(pred_json, ensure_ascii=False)

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
    max_input_tokens: int = 1024,
    failures_to_keep: int = 20,
    limit: Optional[int] = None,
) -> Report:
    """Evaluate the v2 seq2seq Spool Interpreter on the held-out split.

    Pipeline: encode the spool transcript with the `interpret spool:`
    task prefix → greedy decode via T5ForConditionalGeneration → run the
    8-layer Python validator → compute per-task semantic metrics:
      - json_parse_rate, schema_conformance_rate
      - status_accuracy, failure_category_accuracy
      - return_code_accuracy (exact string match when gold has one)
      - explanation_present_rate (new in v2)
      - related_docs_coverage_rate (new in v2)
    """
    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    from ..training.sft_base import _resolve_device
    from ..training.spool_seq2seq import TASK_PREFIX
    from ..validation import validate_spool_result

    model_dir = Path(model_dir)
    dataset_dir = Path(dataset_dir)

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
    tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=True)
    inference_dtype = (
        torch.float16 if target_device.type in ("mps", "cuda") else torch.float32
    )
    model = AutoModelForSeq2SeqLM.from_pretrained(model_dir, dtype=inference_dtype).to(
        target_device
    )
    model.eval()

    layer_pass: dict[int, int] = {i: 0 for i in range(1, 9)}
    all_layers_pass = 0
    status_correct = 0
    cat_correct = 0
    cat_total = 0
    rc_correct = 0
    rc_total = 0
    explanation_present = 0
    explanation_expected = 0
    docs_covered = 0
    docs_expected = 0
    failures: list[dict] = []
    timings: list[float] = []
    per_category: dict[str, dict[str, int]] = {}

    with torch.inference_mode():
        for row in rows:
            source = TASK_PREFIX + row["request"]
            enc = tokenizer(
                source,
                max_length=max_input_tokens,
                truncation=True,
                return_tensors="pt",
            ).to(target_device)

            t0 = time.perf_counter()
            generated = model.generate(
                input_ids=enc["input_ids"],
                attention_mask=enc["attention_mask"],
                max_new_tokens=512,
                num_beams=1,
                do_sample=False,
            )
            if target_device.type == "mps":
                torch.mps.synchronize()
            elif target_device.type == "cuda":
                torch.cuda.synchronize()
            timings.append((time.perf_counter() - t0) * 1000.0)

            gen_text = tokenizer.decode(generated[0], skip_special_tokens=True).strip()
            # T5's SentencePiece tokenizer maps `{` and `}` to <unk>, which
            # `skip_special_tokens=True` strips. The model only ever
            # learned to emit the JSON *interior*; wrap it back before
            # the validator parses.
            if gen_text and not gen_text.startswith("{"):
                gen_text = "{" + gen_text
            if gen_text and not gen_text.endswith("}"):
                gen_text = gen_text + "}"

            result = validate_spool_result(
                gen_text,
                schema_path=schema_path,
                contracts_path=contracts_path,
                user_prompt=row["request"],
            )
            for layer in range(1, 9):
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

            # New v2 metrics
            gold_status = gold.get("status")
            if gold_status and gold_status != "completed":
                explanation_expected += 1
                if pred is not None:
                    expl = pred.get("explanation")
                    if isinstance(expl, str) and expl.strip():
                        explanation_present += 1

            gold_docs = gold.get("relatedDocs") or []
            if gold_docs and pred is not None:
                docs_expected += 1
                pred_docs = set(pred.get("relatedDocs") or [])
                # Coverage = at least one gold doc key recovered.
                if any(d in pred_docs for d in gold_docs):
                    docs_covered += 1

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
        "explanation_validity_rate": layer_pass[7] / n,
        "related_docs_validity_rate": layer_pass[8] / n,
        "all_layers_pass_rate": all_layers_pass / n,
        "status_accuracy": status_correct / n,
        "failure_category_accuracy": cat_correct / cat_total if cat_total else 0.0,
        "return_code_accuracy": rc_correct / rc_total if rc_total else 0.0,
        "explanation_present_rate": (
            explanation_present / explanation_expected if explanation_expected else 0.0
        ),
        "related_docs_coverage_rate": (
            docs_covered / docs_expected if docs_expected else 0.0
        ),
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
        f"rc={metrics['return_code_accuracy']:.3f} "
        f"expl={metrics['explanation_present_rate']:.3f} "
        f"docs={metrics['related_docs_coverage_rate']:.3f}"
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
