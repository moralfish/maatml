# JCL Validator — Training Instructions

## 1. Purpose

Local AI inference model for Flow that validates JCL documents and emits a structured `JclValidationResult` JSON object. The model never executes a job, never reads credentials, never submits anything — it only labels what it sees.

The output is consumed by Flow's pre-submit validation gate; downstream code refuses to submit when `valid: false`.

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

**Starting point: Qwen3-1.7B.** Don't escalate until 500+ samples and the smaller base provably misses targets.

---

## 3. Training Objective

```
Sanitized JCL document  ->  JclValidationResult JSON
```

The model learns to:

- Parse a JCL document and detect syntactic/semantic defects.
- Emit a strict JSON object with `valid`, `errors[]`, and `confidence`.
- Pinpoint the offending line (1-indexed) for each error.
- Choose an `error_code` from the closed enum.
- Suggest fixes when obvious; otherwise leave `suggestion` absent.

The model must NOT:

- Execute or submit JCL.
- Invent new error codes.
- Return prose, markdown, or commentary outside the JSON.
- Set `valid: true` while populating `errors`.

---

## 4. Expected Output Format

Required shape:

```json
{
  "valid": false,
  "errors": [
    {
      "line": 7,
      "column": 1,
      "severity": "error",
      "code": "missing_dd",
      "message": "DD statement missing required DSN parameter.",
      "suggestion": "Add `DSN=PROJ.LOAD,DISP=SHR` after `//STEP01`."
    }
  ],
  "confidence": 0.94
}
```

Field rules:

- `valid` (bool, required) — true iff `errors` is empty.
- `errors` (array, required) — at most 5 entries; ordered from most to least severe.
- `confidence` (number, required) — float in [0.0, 1.0].

Per-error fields:

- `line` (int, required) — 1-indexed line in the input JCL.
- `column` (int, optional) — 1-indexed column.
- `severity` (string, required) — one of `error`, `warning`, `info`.
- `code` (string, required) — one of the 8 enum values listed in §5.
- `message` (string, required) — short, human-readable.
- `suggestion` (string, optional) — actionable fix.

Return JSON only. No markdown fences. No commentary.

---

## 5. Bounded Vocabulary

Allowed `severity` values:

```text
error
warning
info
```

Allowed `code` values (exhaustive):

```text
missing_dd
invalid_job_card
unresolved_symbolic_parameter
continuation_error
invalid_exec_statement
invalid_dataset_reference_structure
other
none
```

`none` is reserved for `valid: true` results with `errors: []`.

Full enumeration in [`datasets/node_contracts.json`](datasets/node_contracts.json).

---

## 6. Dataset Format

JSONL. Each row carries a hand-authored or Claude-generated training example:

```json
{
  "sample_id": "seed-jcl-001",
  "category": "missing_dd",
  "source": "hand:starter",
  "request": "<sanitized JCL document text>",
  "expected_validation_result": { "valid": false, "errors": [...], "confidence": 0.92 }
}
```

`category` lines up with the error codes plus `valid` (clean inputs).

Dataset split: 80% / 10% / 10% (train / val / test).

---

## 7. Dataset Categories

Balance across the 8 codes plus a `valid` clean-input bucket:

- `valid` — clean JCL, expect `valid: true`, `errors: []`.
- `missing_dd` — missing required DD statements.
- `invalid_job_card` — malformed `// JOB` cards.
- `unresolved_symbolic_parameter` — `&FOO` without a `// SET FOO=...`.
- `continuation_error` — broken continuation columns.
- `invalid_exec_statement` — malformed `// EXEC PGM=...`.
- `invalid_dataset_reference_structure` — bad `DSN=` syntax.
- `other` — defects that don't fit the above categories.
- Multi-error cases (mix two or more codes in one document).
- Edge cases: extremely long documents, near-valid edge cases, etc.

---

## 8. Dataset Size Targets

| Stage | Size |
|---|---:|
| Smoke test | 50–100 |
| First usable | 500–1,000 |
| Internal | 2,000–5,000 |

Bootstrap workflow: hand-author ~30 starter samples covering all categories → expand to 500–600 by hand-authoring (or any out-of-band tool you trust) → keep every row gated by the 6-layer `validate_jcl_result` check before merging → retrain.

---

## 9. System Prompt

In [`datasets/prompt_spec.json`](datasets/prompt_spec.json) `system` field. Stays consistent across runs. Instructs the model to:

- Return strict JSON only.
- Use only the 8 allowed error codes.
- Never invent codes or fields.
- Cap at 5 errors per result.
- Calibrate `confidence`.

---

## 10. Training Method

Supervised fine-tuning with LoRA. Same shape as FlowGraphGenerator:

- Base + LoRA adapter → merged safetensors → Candle runtime.
- 3-message conversations (system, user, assistant).
- bf16 autocast on MPS/CUDA; weights stay fp32.
- LoRA rank 16, alpha 32, dropout 0.05, attention-only target modules.
- 4 epochs default; bump to 8–12 if loss plateaus high.

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

Smoke profile uses Qwen3-0.6B for fast pipeline validation.

---

## 12. Validation Requirements

The 6-layer Python validator at [`src/flow_ml/validation/jcl_validator.py`](../../src/flow_ml/validation/jcl_validator.py) enforces:

1. **JSON parse** — output is valid JSON.
2. **JSON schema** — matches `jcl_validation_schema.json`.
3. **Severity enum** — every error.severity is in {error, warning, info}.
4. **Code enum** — every error.code is in the 8-value vocabulary.
5. **Field shape** — line ≥ 1, message non-empty.
6. **Consistency** — `valid: false` iff `errors` non-empty; `confidence` in [0, 1].

Outputs that fail any layer are rejected at training time (during corpus generation) and counted at evaluation time.

---

## 13. Evaluation Metrics

| Metric | Target |
|---|---:|
| `json_parse_rate` | ≥ 0.95 |
| `schema_conformance_rate` | ≥ 0.90 |
| `severity_accuracy` | ≥ 0.85 |
| `code_accuracy` | ≥ 0.85 |
| `valid_flag_accuracy` | ≥ 0.95 |
| `line_within_3_accuracy` | ≥ 0.70 |

`line_within_3` allows ±3 lines on the predicted error line — perfect line-level localisation is harder than the model needs to nail; ±3 is the practical UI tolerance.

---

## 14. Test Prompt Set

Maintain a fixed `test_prompt_set.jsonl` reused after every training run. Include:

- One clean `valid: true` JCL.
- One sample per error code.
- A multi-error sample.
- A near-valid edge case (e.g. column 71 vs column 72 continuation).

---

## 15. Artifact Requirements

Final `.fm` archive contains:

```text
model.safetensors
config.json
tokenizer.json
prompt_spec.json
jcl_validation_schema.json
node_contracts.json
manifest.json
```

Merged safetensors (LoRA folded in) at fp16. Total package ~3.4 GB.

---

## 16. Versioning

```text
jcl-validator-1.7b-v0.1
jcl-validator-1.7b-v0.2
jcl-validator-4b-instruct-2507-v0.1
```

Each version records: base model, dataset version, training config, eval results, known limitations.
