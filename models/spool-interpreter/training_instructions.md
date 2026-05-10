# Spool Interpreter — Training Instructions

## 1. Purpose

Local AI inference model for Flow that reads a z/OS job spool dump and emits a structured `SpoolInterpretation` JSON object. The model produces interpretation only — Flow Studio surfaces the result alongside the original spool; downstream actions (notify, archive, etc.) are scheduled by separate flow nodes.

---

## 2. Recommended Base Models

Same ladder as the FlowGraphGenerator family:

| Stage | Model | Purpose |
|---|---|---|
| First experiment | Qwen3-1.7B | Validate dataset and pipeline quickly |
| MVP local model | Qwen3-1.7B | Default local-inference target if metrics pass |
| Balanced quality | Qwen3-4B-Instruct-2507 | Use only if 1.7B misses semantic-accuracy bars |
| Code-flavoured benchmark | Qwen2.5-Coder-3B-Instruct | Comparison baseline |
| Higher-quality benchmark | Qwen2.5-Coder-7B-Instruct | Optional only after MVP is stable |

**Starting point: Qwen3-1.7B.** This is a substantial bump from the legacy SmolLM2-360M baseline; expect a meaningful quality jump on the 12-category root-cause classification.

---

## 3. Training Objective

```
Sanitized z/OS spool output  ->  SpoolInterpretation JSON
```

Learn to:

- Identify the job's terminal status (completed / failed / abended / skipped / running).
- Extract the return code (4-char string) when present.
- Localise the root cause of failure to one of 12 categories.
- Produce short, actionable `summary`, `rootCause`, and `suggestedFix` strings.
- Calibrate confidence.

Must NOT:

- Run any executor action.
- Invent failure categories.
- Return prose, markdown, or commentary outside the JSON.

---

## 4. Expected Output Format

Required shape:

```json
{
  "summary": "Build job ABENDed at STEP02 (S0C7).",
  "status": "abended",
  "returnCode": null,
  "rootCause": "Data exception (S0C7) reading numeric field with invalid packed-decimal data in INFILE record 17.",
  "suggestedFix": "Validate INFILE before submission; clean or reject record 17.",
  "failureCategory": "execution_abend",
  "confidence": 0.91
}
```

Field rules in [`training_instructions.md`](training_instructions.md) §5 / [`datasets/node_contracts.json`](datasets/node_contracts.json).

---

## 5. Bounded Vocabulary

Allowed `status` values:

```text
completed
failed
abended
skipped
running
```

Allowed `failureCategory` values (12 total, plus `null` on `status: completed`):

```text
dataset_resolution_failure
allocation_failure
permission_or_security_failure
jcl_syntax_failure
utility_parameter_failure
execution_abend
scheduler_or_environment_issue
other
smart_restart_resource_unavailable
smart_restart_configuration
smart_restart_application_logic
smart_restart_input_syntax
```

Smart/RESTART subcategories are synced from `../flow-studio/docs/smart-restart/messages.md` by [`scripts/sync-smart-restart-knowledge.sh`](../../scripts/sync-smart-restart-knowledge.sh).

---

## 6. Dataset Format

JSONL. Each row carries a hand-authored or Claude-generated training example:

```json
{
  "sample_id": "seed-spool-001",
  "category": "execution_abend",
  "source": "hand:starter",
  "request": "<sanitized spool dump text>",
  "expected_interpretation": { "summary": "...", "status": "abended", ... }
}
```

`category` aligns with the failureCategory enum plus a `completed` bucket for clean-completion samples.

Dataset split: 80% / 10% / 10%.

---

## 7. Dataset Categories

Balance across the 12 failureCategory values plus a `completed` clean-completion bucket. Smart/RESTART subcategories need extra hand-curation because Claude has limited prior exposure to those Smart/RESTART-specific patterns.

---

## 8. Dataset Size Targets

| Stage | Size |
|---|---:|
| Smoke test | 50–100 |
| First usable | 500–1,000 |
| Internal | 2,000–5,000 |

Bootstrap workflow: hand-author ~30 starter samples covering all categories → expand to 500+ by hand-authoring (or any out-of-band tool you trust) → keep every row gated by the 6-layer `validate_spool_result` check before merging → retrain.

---

## 9. System Prompt

In [`datasets/prompt_spec.json`](datasets/prompt_spec.json) `system` field. Stays consistent across runs.

---

## 10. Training Method

Supervised fine-tuning with LoRA. Same shape as FlowGraphGenerator + JCL Validator.

---

## 11. Training Configuration

| Setting | Value |
|---|---:|
| Epochs | 4 |
| Learning rate | 1e-4 |
| LoRA rank | 16 |
| LoRA alpha | 32 |
| LoRA dropout | 0.05 |
| Max sequence length | 4096 |
| Batch size | 2 |
| Grad accumulation | 8 |
| Precision | bf16 (autocast) |

---

## 12. Validation Requirements

The 6-layer Python validator at [`src/flow_ml/validation/spool_validator.py`](../../src/flow_ml/validation/spool_validator.py) enforces:

1. JSON parse
2. JSON schema (matches `spool_interpretation_schema.json`)
3. status enum
4. failureCategory enum (or null on completed)
5. Field shape (non-empty summary, rootCause, suggestedFix; returnCode is string or null)
6. Consistency (status=completed ⟹ failureCategory in {null, "other"})

---

## 13. Evaluation Metrics

| Metric | Target |
|---|---:|
| `json_parse_rate` | ≥ 0.95 |
| `schema_conformance_rate` | ≥ 0.90 |
| `status_accuracy` | ≥ 0.90 |
| `failure_category_accuracy` | ≥ 0.80 |
| `return_code_accuracy` | ≥ 0.85 (string match when present) |

---

## 14. Test Prompt Set

Maintain a fixed `test_prompt_set.jsonl` reused after every training run. Include:
- One `completed` clean-completion spool.
- One sample per failureCategory.
- A multi-failure spool where the model must pick the dominant cause.
- A truncated spool to test confidence calibration.

---

## 15. Artifact Requirements

Final `.fm` archive contains:

```text
model.safetensors
config.json
tokenizer.json
prompt_spec.json
spool_interpretation_schema.json
node_contracts.json
manifest.json
```

Merged safetensors at fp16. Total package ~3.4 GB.

---

## 16. Versioning

```text
spool-interpreter-1.7b-v0.1
spool-interpreter-1.7b-v0.2
spool-interpreter-4b-instruct-2507-v0.1
```
