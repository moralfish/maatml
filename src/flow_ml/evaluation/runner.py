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
    max_input_tokens: int = 2048,
    latency_n: int = 50,
    failures_to_keep: int = 20,
) -> Report:
    import torch
    from transformers import AutoTokenizer

    from ..training.jcl_validator import (
        CATEGORY_INDEX,
        CATEGORY_LABELS,
        JclCollator,
        JclMultiHeadModel,
        _resolve_device,
    )

    model_dir = Path(model_dir)
    dataset_dir = Path(dataset_dir)
    rows = list(iter_jsonl(dataset_dir / f"{split}.jsonl"))
    if not rows:
        raise ValueError(f"No rows in {dataset_dir / f'{split}.jsonl'}")

    target_device = _resolve_device(device)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = JclMultiHeadModel.load(model_dir).to(target_device)
    model.eval()

    collator = JclCollator(tokenizer, max_length=max_input_tokens)

    seq_true: list[int] = []
    seq_pred: list[int] = []
    cat_true: list[str] = []
    cat_pred: list[str] = []
    line_correct = 0
    line_top3 = 0
    line_total = 0
    failures: list[dict] = []

    with torch.inference_mode():
        for row in rows:
            batch = collator([row])
            batch = {k: v.to(target_device) for k, v in batch.items()}
            out = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
            )
            seq_logits = out["seq_logits"][0]
            cat_logits = out["cat_logits"][0]
            line_logits = out["line_logits"][0]  # [T, 2]

            seq_p = int(seq_logits.argmax(-1).item())
            cat_p = int(cat_logits.argmax(-1).item())
            seq_true.append(0 if row["is_valid"] else 1)
            seq_pred.append(seq_p)
            cat_true.append(row.get("error_category") or "none")
            cat_pred.append(CATEGORY_LABELS[cat_p])

            err_line = row.get("error_line")
            if err_line is not None:
                line_total += 1
                error_probs = line_logits.softmax(-1)[:, 1]
                offsets = collator.tokenizer(
                    row["sanitized_jcl"],
                    max_length=max_input_tokens,
                    truncation=True,
                    return_offsets_mapping=True,
                )["offset_mapping"]
                text = row["sanitized_jcl"]
                newline_starts = [i + 1 for i, c in enumerate(text) if c == "\n"]
                line_scores: dict[int, list[float]] = {}
                for t, (start, end) in enumerate(offsets):
                    if start == 0 and end == 0:
                        continue
                    line_id = sum(1 for ns in newline_starts if ns <= start) + 1
                    if t >= error_probs.shape[0]:
                        continue
                    line_scores.setdefault(line_id, []).append(float(error_probs[t].item()))
                if line_scores:
                    line_avg = {ln: sum(v) / len(v) for ln, v in line_scores.items()}
                    sorted_lines = sorted(line_avg.items(), key=lambda kv: kv[1], reverse=True)
                    pred_line = sorted_lines[0][0] if sorted_lines else None
                    top3 = {ln for ln, _ in sorted_lines[:3]}
                    if pred_line == err_line:
                        line_correct += 1
                    if err_line in top3:
                        line_top3 += 1

            if seq_p != (0 if row["is_valid"] else 1) and len(failures) < failures_to_keep:
                failures.append(
                    {
                        "sample_id": row["sample_id"],
                        "true_category": cat_true[-1],
                        "pred_category": cat_pred[-1],
                        "true_seq": seq_true[-1],
                        "pred_seq": seq_p,
                    }
                )

    # Aggregate metrics
    n = len(rows)
    seq_acc = sum(1 for t, p in zip(seq_true, seq_pred) if t == p) / n
    cat_acc = sum(1 for t, p in zip(cat_true, cat_pred) if t == p) / n
    seq_prf = _binary_prf(
        tp=sum(1 for t, p in zip(seq_true, seq_pred) if t == 1 and p == 1),
        fp=sum(1 for t, p in zip(seq_true, seq_pred) if t == 0 and p == 1),
        fn=sum(1 for t, p in zip(seq_true, seq_pred) if t == 1 and p == 0),
    )

    metrics = {
        "seq_accuracy": seq_acc,
        "seq_precision": seq_prf["precision"],
        "seq_recall": seq_prf["recall"],
        "seq_f1": seq_prf["f1"],
        "category_accuracy": cat_acc,
    }
    if line_total:
        metrics["line_accuracy"] = line_correct / line_total
        metrics["line_top3_accuracy"] = line_top3 / line_total

    per_class = _per_class_prf(cat_true, cat_pred, list(CATEGORY_INDEX.keys()))

    # Latency benchmark
    sample_for_latency = rows[: min(latency_n, n)]
    timings: list[float] = []
    with torch.inference_mode():
        for _ in range(5):
            warm = collator([sample_for_latency[0]])
            warm = {k: v.to(target_device) for k, v in warm.items()}
            model(input_ids=warm["input_ids"], attention_mask=warm["attention_mask"])
        for row in sample_for_latency:
            batch = collator([row])
            batch = {k: v.to(target_device) for k, v in batch.items()}
            t0 = time.perf_counter()
            model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
            if target_device.type == "mps":
                torch.mps.synchronize()
            elif target_device.type == "cuda":
                torch.cuda.synchronize()
            timings.append((time.perf_counter() - t0) * 1000.0)

    report = Report(
        model_id=str(model_dir),
        task="jcl_validation",
        dataset=str(dataset_dir / f"{split}.jsonl"),
        n=n,
        metrics=metrics,
        per_class=per_class,
        latency_ms=_latency_stats(timings),
        baseline_delta=_baseline_delta(metrics, baseline_path),
        sample_failures=failures,
    )
    report.write(out_path)
    console.print(f"[green]JCL eval[/]: n={n} seq_acc={seq_acc:.3f} cat_acc={cat_acc:.3f}")
    return report


def evaluate_spool(
    model_dir: str | Path,
    dataset_dir: str | Path,
    out_path: str | Path,
    *,
    baseline_path: Optional[str | Path] = None,
    device: str = "auto",
    split: str = "test",
    max_input_tokens: int = 2048,
    failures_to_keep: int = 20,
) -> Report:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from ..data.schemas import SpoolInterpretation
    from ..training.spool_interpreter import _resolve_device, build_chat_example

    model_dir = Path(model_dir)
    dataset_dir = Path(dataset_dir)
    spec_path = model_dir / "prompt_spec.json"
    prompt_spec = json.loads(spec_path.read_text(encoding="utf-8"))
    rows = list(iter_jsonl(dataset_dir / f"{split}.jsonl"))
    if not rows:
        raise ValueError(f"No rows in {dataset_dir / f'{split}.jsonl'}")

    target_device = _resolve_device(device)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_dir).to(target_device)
    model.eval()

    valid_count = 0
    cat_true: list[str] = []
    cat_pred: list[str] = []
    rc_correct = 0
    rc_total = 0
    failures: list[dict] = []
    timings: list[float] = []

    failure_categories = set(prompt_spec.get("failure_categories", []))

    with torch.inference_mode():
        for row in rows:
            ex = build_chat_example(row, prompt_spec, tokenizer, max_length=max_input_tokens)
            prompt_ids = ex["input_ids"][: ex["labels"].count(-100)]
            input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=target_device)
            attention_mask = torch.ones_like(input_ids)

            t0 = time.perf_counter()
            generated = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=prompt_spec.get("max_new_tokens", 256),
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
            if target_device.type == "mps":
                torch.mps.synchronize()
            elif target_device.type == "cuda":
                torch.cuda.synchronize()
            timings.append((time.perf_counter() - t0) * 1000.0)

            gen_text = tokenizer.decode(generated[0, input_ids.shape[1]:], skip_special_tokens=True).strip()
            parsed: Optional[dict] = None
            try:
                parsed = json.loads(gen_text)
                SpoolInterpretation.model_validate(parsed)
                valid_count += 1
            except Exception:
                parsed = None

            cat_true.append(row["failure_category"])
            pred_cat = "other"
            pred_rc = None
            if parsed is not None:
                # No failure_category in the contract; infer category by string-matching against root_cause/summary
                blob = " ".join(str(parsed.get(k, "")) for k in ("rootCause", "summary", "suggestedFix")).lower()
                for cand in failure_categories:
                    if cand.replace("_", " ") in blob:
                        pred_cat = cand
                        break
                pred_rc = parsed.get("returnCode")
            cat_pred.append(pred_cat)

            true_rc = row.get("return_code")
            if true_rc is not None:
                rc_total += 1
                if str(pred_rc) == str(true_rc):
                    rc_correct += 1

            if (parsed is None or pred_cat != row["failure_category"]) and len(failures) < failures_to_keep:
                failures.append(
                    {
                        "sample_id": row["sample_id"],
                        "true_category": row["failure_category"],
                        "pred_category": pred_cat,
                        "raw_output": gen_text[:500],
                    }
                )

    n = len(rows)
    cat_acc = sum(1 for t, p in zip(cat_true, cat_pred) if t == p) / n
    metrics: dict[str, float] = {
        "json_validity": valid_count / n,
        "category_accuracy": cat_acc,
    }
    if rc_total:
        metrics["return_code_exact_match"] = rc_correct / rc_total
    per_class = _per_class_prf(cat_true, cat_pred, sorted(failure_categories) or sorted(set(cat_true)))

    report = Report(
        model_id=str(model_dir),
        task="spool_interpretation",
        dataset=str(dataset_dir / f"{split}.jsonl"),
        n=n,
        metrics=metrics,
        per_class=per_class,
        latency_ms=_latency_stats(timings),
        structure_validity=valid_count / n,
        baseline_delta=_baseline_delta(metrics, baseline_path),
        sample_failures=failures,
    )
    report.write(out_path)
    console.print(f"[green]Spool eval[/]: n={n} json_validity={valid_count/n:.3f} cat_acc={cat_acc:.3f}")
    return report


def evaluate_dsl(
    model_dir: str | Path,
    dataset_dir: str | Path,
    out_path: str | Path,
    *,
    baseline_path: Optional[str | Path] = None,
    device: str = "auto",
    split: str = "test",
    max_input_tokens: int = 2048,
    failures_to_keep: int = 20,
) -> Report:
    """Evaluate the DSL Generator on the held-out split.

    Metrics:

    - `json_validity`: fraction of generations that parse as JSON with a
      string `dsl` field.
    - `parser_roundtrip`: fraction whose `dsl` field is accepted by the real
      flow-dsl parser (via the `flow_dsl_py` PyO3 binding). This is the
      strongest validity signal - it means the generated text is a valid
      Flow DSL document, not just plausible-looking text.
    - `dsl_header_present`: fraction whose `dsl` field starts with a
      `flow "..."` header line. Heuristic; subsumed by `parser_roundtrip`
      when the binding is installed but kept for back-compat with reports
      generated before the binding existed.
    - `node_count_jaccard`: average Jaccard overlap between predicted and gold
      node-id sets (string match on the slug before `[`).
    - `edge_count_match`: fraction where the count of `-->` lines matches gold.

    The PyO3 binding (`flow_dsl_py`) is built from
    `flow-studio/crates/flow-dsl-py` via `maturin develop`. If it isn't
    installed in the active environment, `parser_roundtrip` falls back to
    `dsl_header_present` so the evaluator never hard-fails on a missing
    optional dep; a warning is printed once.
    """
    import re
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from ..data.schemas import DslGeneration
    from ..training.dsl_generator import _resolve_device, build_chat_example
    from .graph_diff import score_pair

    # Optional dependency - import lazily so the evaluator works even when
    # the binding isn't built yet (e.g. on a fresh checkout).
    try:
        import flow_dsl_py as _dsl_py  # type: ignore
    except ImportError:
        _dsl_py = None
        console.print(
            "[yellow]flow_dsl_py not installed; parser_roundtrip + structural metrics will degrade.[/] "
            "Build it with `cd flow-studio/crates/flow-dsl-py && maturin develop --release`."
        )

    model_dir = Path(model_dir)
    dataset_dir = Path(dataset_dir)
    spec_path = model_dir / "prompt_spec.json"
    prompt_spec = json.loads(spec_path.read_text(encoding="utf-8"))
    rows = list(iter_jsonl(dataset_dir / f"{split}.jsonl"))
    if not rows:
        raise ValueError(f"No rows in {dataset_dir / f'{split}.jsonl'}")

    target_device = _resolve_device(device)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_dir).to(target_device)
    model.eval()

    valid_count = 0
    header_count = 0
    parser_ok_count = 0
    edge_count_match = 0
    node_jaccards: list[float] = []
    structural_match_count = 0
    semantic_tag_match_count = 0
    failures: list[dict] = []
    timings: list[float] = []

    node_re = re.compile(r"^\s*([a-z0-9-]+)\s*\[", re.MULTILINE)
    edge_re = re.compile(r"-->")
    header_re = re.compile(r'^\s*flow\s+"', re.MULTILINE)

    with torch.inference_mode():
        for row in rows:
            ex = build_chat_example(row, prompt_spec, tokenizer, max_length=max_input_tokens)
            prompt_ids = ex["input_ids"][: ex["labels"].count(-100)]
            input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=target_device)
            attention_mask = torch.ones_like(input_ids)

            t0 = time.perf_counter()
            generated = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=prompt_spec.get("max_new_tokens", 512),
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
                generated[0, input_ids.shape[1] :], skip_special_tokens=True
            ).strip()

            parsed: Optional[dict] = None
            dsl_pred: Optional[str] = None
            try:
                parsed = json.loads(gen_text)
                DslGeneration.model_validate(parsed)
                dsl_pred = parsed.get("dsl") if isinstance(parsed, dict) else None
                if isinstance(dsl_pred, str) and dsl_pred:
                    valid_count += 1
            except Exception:
                parsed = None

            gold_dsl = row["dsl"]
            gold_nodes = set(node_re.findall(gold_dsl))
            gold_edge_count = len(edge_re.findall(gold_dsl))

            if dsl_pred:
                if header_re.search(dsl_pred):
                    header_count += 1
                if _dsl_py is not None and _dsl_py.parses(dsl_pred):
                    parser_ok_count += 1
                    # Tier 2 + 3: parse both sides as graph dicts and run
                    # the structural / semantic-tag comparator. Wrapped in
                    # a broad try because a parser binding bug or a graph
                    # shape we did not anticipate must not zero out the
                    # whole eval.
                    try:
                        pred_graph = _dsl_py.parse(dsl_pred)
                        gold_graph = _dsl_py.parse(gold_dsl)
                        scores = score_pair(pred_graph, gold_graph)
                        if scores["structural_match"]:
                            structural_match_count += 1
                        if scores["semantic_tag_match"]:
                            semantic_tag_match_count += 1
                    except Exception:
                        # Fall through; the row contributes neither
                        # structural nor semantic credit.
                        pass
                pred_nodes = set(node_re.findall(dsl_pred))
                if gold_nodes or pred_nodes:
                    inter = gold_nodes & pred_nodes
                    union = gold_nodes | pred_nodes
                    node_jaccards.append(len(inter) / len(union) if union else 0.0)
                else:
                    node_jaccards.append(1.0)
                pred_edge_count = len(edge_re.findall(dsl_pred))
                if pred_edge_count == gold_edge_count:
                    edge_count_match += 1

            if (parsed is None or not dsl_pred) and len(failures) < failures_to_keep:
                failures.append(
                    {
                        "sample_id": row["sample_id"],
                        "description": row["description"],
                        "raw_output": gen_text[:1000],
                    }
                )

    n = len(rows)
    metrics: dict[str, float] = {
        "json_validity": valid_count / n,
        "dsl_header_present": header_count / n,
        # parser_roundtrip is the canonical validity metric. When the
        # binding isn't installed it mirrors the header heuristic so the
        # evaluator's quality gates degrade gracefully rather than zeroing.
        "parser_roundtrip": (
            parser_ok_count / n if _dsl_py is not None else header_count / n
        ),
        "node_count_jaccard": sum(node_jaccards) / max(1, len(node_jaccards)),
        "edge_count_match": edge_count_match / n,
        # Three-tier metric: structural and semantic-tag rates pair with
        # parser_roundtrip (= "tier 1: did it parse?"). Both are
        # conditional on `flow_dsl_py` being installed - without the
        # binding we cannot parse the gold/predicted DSL into graphs to
        # compare. They report 0.0 in that case rather than reverting to
        # a heuristic, because a heuristic for "is this structurally
        # equivalent" would just be a worse parser.
        "structural_match": structural_match_count / n,
        "semantic_tag_match": semantic_tag_match_count / n,
    }

    report = Report(
        model_id=str(model_dir),
        task="dsl_generation",
        dataset=str(dataset_dir / f"{split}.jsonl"),
        n=n,
        metrics=metrics,
        per_class={},
        latency_ms=_latency_stats(timings),
        structure_validity=metrics["json_validity"],
        baseline_delta=_baseline_delta(metrics, baseline_path),
        sample_failures=failures,
    )
    report.write(out_path)
    console.print(
        f"[green]DSL eval[/]: n={n} "
        f"parse={metrics['parser_roundtrip']:.3f} "
        f"struct={metrics['structural_match']:.3f} "
        f"tag={metrics['semantic_tag_match']:.3f} "
        f"node_jaccard={metrics['node_count_jaccard']:.3f}"
    )
    return report


def evaluate_agent(
    model_dir: str | Path,
    dataset_dir: str | Path,
    out_path: str | Path,
    *,
    baseline_path: Optional[str | Path] = None,
    device: str = "auto",
    split: str = "test",
    max_input_tokens: int = 2048,
    failures_to_keep: int = 20,
) -> Report:
    """Evaluate the Agent Planner on the fixed benchmark split.

    The v1 quality gates are intentionally structural: the local agent must
    produce parseable JSON, match the declared schema, choose the right broad
    intent, and emit DSL/refusal fields when the benchmark expects them.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from ..data.schemas import AgentPlan
    from ..training.agent_planner import _resolve_device, build_chat_example

    model_dir = Path(model_dir)
    dataset_dir = Path(dataset_dir)
    spec_path = model_dir / "prompt_spec.json"
    prompt_spec = json.loads(spec_path.read_text(encoding="utf-8"))
    rows = list(iter_jsonl(dataset_dir / f"{split}.jsonl"))
    if not rows:
        raise ValueError(f"No rows in {dataset_dir / f'{split}.jsonl'}")

    target_device = _resolve_device(device)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_dir).to(target_device)
    model.eval()

    json_valid = 0
    schema_valid = 0
    intent_matches = 0
    dsl_presence_matches = 0
    refusal_matches = 0
    action_jaccards: list[float] = []
    failures: list[dict] = []
    timings: list[float] = []

    with torch.inference_mode():
        for row in rows:
            ex = build_chat_example(row, prompt_spec, tokenizer, max_length=max_input_tokens)
            prompt_ids = ex["input_ids"][: ex["labels"].count(-100)]
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
                generated[0, input_ids.shape[1] :], skip_special_tokens=True
            ).strip()

            parsed: Optional[dict] = None
            plan: Optional[AgentPlan] = None
            try:
                parsed = json.loads(gen_text)
                json_valid += 1
                plan = AgentPlan.model_validate(parsed)
                schema_valid += 1
            except Exception:
                parsed = None
                plan = None

            gold = row["agent_plan"]
            if plan is not None:
                gold_intent = row["expected_intent"].lower()
                pred_intent = plan.intent_summary.lower()
                if gold_intent in pred_intent or pred_intent in gold_intent:
                    intent_matches += 1

                gold_has_dsl = bool(gold.get("dsl") or gold.get("dsl_patch"))
                pred_has_dsl = bool(plan.dsl or plan.dsl_patch)
                if gold_has_dsl == pred_has_dsl:
                    dsl_presence_matches += 1

                gold_refusal = bool(gold.get("refusal_reason"))
                pred_refusal = bool(plan.refusal_reason)
                if gold_refusal == pred_refusal:
                    refusal_matches += 1

                gold_actions = {step["action"] for step in gold.get("plan_steps", [])}
                pred_actions = {step.action for step in plan.plan_steps}
                union = gold_actions | pred_actions
                action_jaccards.append(len(gold_actions & pred_actions) / len(union) if union else 1.0)

            if (plan is None or parsed is None) and len(failures) < failures_to_keep:
                failures.append(
                    {
                        "sample_id": row["sample_id"],
                        "request": row["request"],
                        "raw_output": gen_text[:1000],
                    }
                )

    n = len(rows)
    metrics: dict[str, float] = {
        "json_validity": json_valid / n,
        "schema_validity": schema_valid / n,
        "intent_match": intent_matches / n,
        "dsl_presence_accuracy": dsl_presence_matches / n,
        "refusal_accuracy": refusal_matches / n,
        "action_jaccard": sum(action_jaccards) / max(1, len(action_jaccards)),
    }

    report = Report(
        model_id=str(model_dir),
        task="agent_planning",
        dataset=str(dataset_dir / f"{split}.jsonl"),
        n=n,
        metrics=metrics,
        per_class={},
        latency_ms=_latency_stats(timings),
        structure_validity=metrics["schema_validity"],
        baseline_delta=_baseline_delta(metrics, baseline_path),
        sample_failures=failures,
    )
    report.write(out_path)
    console.print(
        f"[green]Agent eval[/]: n={n} "
        f"json_validity={metrics['json_validity']:.3f} "
        f"schema_validity={metrics['schema_validity']:.3f} "
        f"action_jaccard={metrics['action_jaccard']:.3f}"
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
        "evaluate_dsl(...), or evaluate_agent(...)"
    )
