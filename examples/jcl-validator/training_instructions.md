# JCL Validator — Training Instructions

## 1. Purpose

Pre-submission validation of JCL syntax, parameter completeness, and common
error patterns. Emits a structured `JclValidationResult` JSON for downstream
consumers. Architecture: multi-head BERT classifier.

| | Current |
|---|---|
| Base | ModernBERT-base (~150 MB) |
| Method | Full fine-tune, multi-head |
| Size | ~150-200 MB fp16 |
| Latency | <500 ms target |
| Tokenizer | Custom JCL BPE with column-aware pre-tokenizer |

## 2. Base model

`answerdotai/ModernBERT-base` — encoder-only, 8K native context. The 8K
context handles realistic JCL decks without truncation; the model is
small enough that full fine-tune (no LoRA) fits on a developer laptop.

Smoke profile keeps the same base — there's no scale ladder for the
classifier. Smoke just trims epochs + dataset.

## 3. Training objective

Four classification heads sharing the BERT encoder:

| Head | Type | Output |
|---|---|---|
| `validity` | binary | `valid: bool` |
| `error_code` | 8-class softmax | one of the codes in §5; `none` when valid |
| `severity` | 3-class softmax | `error \| warning \| info`; `none` when valid |
| `line_localization` | per-token classification | each token classified as `error_line` / `not_error_line`; runtime takes the first-error-line span |

Loss = weighted sum across heads. Default weights `{validity: 1.0,
error_code: 1.0, severity: 0.5, line: 0.3}` — `validity` and `error_code`
dominate; `severity` is easier (4 of 8 codes are always `error`) so it
gets less weight; `line_localization` is the hardest and noisiest, so it
gets the smallest weight to avoid drowning the other heads.

The model produces logits per head; the runtime stitches them into the
`JclValidationResult` JSON shape (see §4).

## 4. Expected output format

```json
{
  "valid": false,
  "errors": [
    {
      "line": 1,
      "column": 17,
      "severity": "error",
      "code": "invalid_job_card",
      "message": "JOB card is malformed (account, class, or priority fields are missing or invalid).",
      "suggestion": "Re-check the JOB statement: accounting field in parentheses, then optional CLASS/MSGCLASS/PRIORITY."
    }
  ],
  "confidence": 0.93
}
```

The classifier emits one error at most per pass (matching the multi-head
shape — single primary code). For multi-error JCL the model still picks
the dominant code; future iterations can chain inferences for additional
errors.

`message` and `suggestion` come from a deterministic templated phrasebook
keyed by `code`, shipped in `node_contracts.json` as
`error_message_templates`. The model doesn't generate text; it picks the
code and the runtime fills the rest. Zero hallucination on suggestion text
is the explicit design choice.

`confidence` is the validity-head softmax score for the chosen
prediction. The frontend's confidence-band routing reads this against
the node's thresholds.

## 5. Bounded vocabulary

From `examples/jcl-validator/datasets/node_contracts.json`:

- `severities`: `["error", "warning", "info", "none"]`
- `error_codes`: 8 values — `missing_dd`, `invalid_job_card`,
  `unresolved_symbolic_parameter`, `continuation_error`,
  `invalid_exec_statement`, `invalid_dataset_reference_structure`,
  `other`, `none`
- `error_message_templates` (new in v2): per-code suggestion templates the
  runtime uses to fill `message`+`suggestion` from the predicted code.

## 6. Dataset format

Same JSONL shape the v1 corpus used — no regeneration needed. Each row:

```json
{
  "sample_id": "syn-missing_dd-001",
  "source": "synthetic:template",
  "category": "missing_dd",
  "request": "<sanitized JCL deck, multi-line>",
  "expected_validation_result": {
    "valid": false,
    "errors": [{"line": 2, "column": 1, "severity": "error", "code": "missing_dd",
                "message": "...", "suggestion": "..."}],
    "confidence": 0.93
  },
  "split": "train"
}
```

The classifier dataset class flattens this into per-head targets at load
time — `validity` from `valid`, `error_code` from `errors[0].code`,
`severity` from `errors[0].severity`, `line` token-level via the
character-offset-to-token mapping. Sample-level fields stay in the JSONL
for evaluation.

## 7. Splits

80 / 10 / 10 by sample-id hash (same as v1; deterministic across reruns).

## 8. Dataset size targets

| Stage | Samples |
|---|---|
| Smoke | 100-200 |
| MVP | 1000+ (current corpus is 1003 — sufficient) |
| Internal | 2000+ |
| Stronger | 5000+ |

The v1 retrain found the 1000-sample floor; v2 starts there and iterates
if eval gates miss.

## 9. System prompt

N/A — classifier doesn't take a system prompt. The model sees the raw
sanitised JCL via the custom JCL tokenizer (§13). `prompt_spec.json`
stays in the package for documentation purposes and so the manifest
file list is uniform with the other two models.

## 10. Training method

**Full fine-tune** — no LoRA. ModernBERT-base is small enough that full
fine-tune of 150 MB params fits comfortably on a 16-32 GB Mac. The
resulting safetensors merge straight into the package (no LoRA-adapter
step). Cleaner downstream when we revisit ONNX export.

## 11. Training config

```yaml
training:
  model_id: answerdotai/ModernBERT-base
  max_input_tokens: 2048      # ModernBERT supports 8K; 2K covers realistic JCL
  batch_size: 16
  grad_accum: 1
  learning_rate: 2.0e-5       # standard BERT fine-tune LR
  epochs: 4
  weight_decay: 0.01
  warmup_ratio: 0.06
  precision: bf16             # autocast; weights stay fp32
  grad_checkpointing: false
  eval_steps: 9999            # post-train eval only (MPS unified-memory tradeoff)
  save_steps: 250
  logging_steps: 25
  head_loss_weights:
    validity: 1.0
    error_code: 1.0
    severity: 0.5
    line: 0.3
```

Smoke overrides reduce to `epochs: 1, max_steps: 6, batch_size: 2`.

## 12. Training environment

- Python 3.13, `transformers`, `safetensors`, `tokenizers`.
- ModernBERT runs on standard BERT-style transformer attention; the
  saved safetensors checkpoint loads with Hugging Face `AutoModel`.
- Apple Silicon (MPS) primary target, CPU fallback via
  `PYTORCH_ENABLE_MPS_FALLBACK=1`.

## 13. Custom JCL tokenizer

`jcl_plugin/tokenizer.py` — two stages:

1. **Pre-tokenizer (column-aware)**: see `COLUMN_RULES.md` for the
   normative spec. Strips columns 73-80 (sequence numbers, ignored by
   the JCL parser), emits a `<COL1>` special token at the start of each
   line so the model can learn statement boundaries, preserves the
   column-72 continuation marker as a `<CONT>` token.
2. **BPE**: 30,000-vocab BPE trained on the synthetic corpus
   (10,000+ samples via `build_jcl_seeds.py --target 10000`). Output:
   `tokenizer.json` shipped in the package.

The pre-tokenizer is invoked at training time (`JclDataset`) and should be
mirrored at inference by any consumer that tokenizes JCL the same way. A
fixture test on ~20 hand-authored JCL strings locks the expected output.

## 14. Validation requirements

The existing 6-layer `maatml.validation.jcl_validator` JSON gate stays
unchanged — the runtime produces JSON matching the v1 schema. Layers:
JSON parse → schema → severity enum → code enum → field shape →
consistency.

## 15. Evaluation metrics

Required gates (test split):

| Metric | Target |
|---|---|
| `json_parse_rate` | ≥0.99 |
| `schema_conformance_rate` | ≥0.99 |
| `valid_flag_accuracy` | ≥0.95 |
| `code_accuracy` | ≥0.90 |
| `severity_accuracy` | ≥0.90 |
| `line_within_3_accuracy` | ≥0.70 |
| `p95_latency_ms` | <500 |

`p95_latency_ms` is new in v2 — required to validate the <500 ms target.

## 16. Test prompt set

`examples/jcl-validator/datasets/samples/test_prompt_set.jsonl` — fixed
anchors that survive corpus regeneration. Eight benchmark JCL decks
across the eight error codes plus two valid samples.

## 17. Repair dataset

Not applicable — classifier doesn't have the generative model's failure
mode of producing structurally-broken output. Repair-style data lives in
the synthetic corpus directly (each error category has its own
templates).

## 18. Artifact requirements

Training produces a checkpoint under `output/checkpoints/<run-name>/`:
`model.safetensors` (ModernBERT + 4 heads), `config.json`, the custom
`tokenizer.json`, plus the committed `jcl_validation_schema.json` and
`node_contracts.json` (bounded vocab + message templates). Packaging/export
of Hub-ready artifacts from checkpoints is future work.

## 19. Versioning

Bump `version` in `model.yml` (semver) on each retrain that changes behaviour
or schema. Checkpoint run names use `name@version`.
## 20. First training milestone

Success criteria for the first end-to-end run:
- Training completes without OOM / NaN gradients.
- `eval_loss` < 0.5 at end of training.
- All eval gates in §15 met on the test split.

## 21. Recommended sequence

```bash
# 1. Generate the BPE tokenizer training corpus (10k samples).
.venv/bin/python examples/jcl-validator/scripts/build_seeds.py --target 10000 \
    --out examples/jcl-validator/datasets/samples/tokenizer_corpus.jsonl

# 2. Train the custom JCL tokenizer (one-shot; commit the output).
.venv/bin/python examples/jcl-validator/scripts/build_tokenizer.py \
    --corpus examples/jcl-validator/datasets/samples/tokenizer_corpus.jsonl \
    --out examples/jcl-validator/datasets/tokenizer.json

# 3. Train the classifier.
maatml prepare examples/jcl-validator/
maatml train examples/jcl-validator/ --smoke  # ~30s sanity check
maatml train examples/jcl-validator/          # ~10 min on M5 Max

# 4. Evaluate.
maatml evaluate examples/jcl-validator/
```
