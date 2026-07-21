# Spool Interpreter — Training Instructions

## 1. Purpose

Post-execution interpretation of z/OS job spool output. Takes sanitised JES2
spool transcripts and emits a structured `SpoolInterpretation` JSON:

```json
{
  "summary": "...",
  "status": "failed|completed|...",
  "returnCode": "S0C7",
  "rootCause": "...",
  "suggestedFix": "...",
  "explanation": "...",
  "relatedDocs": ["abend-s0c7", "ile-data-validation"],
  "failureCategory": "execution_abend",
  "confidence": 0.92
}
```

Architecture: flan-t5-base encoder-decoder (seq2seq).

| | Current |
|---|---|
| Base | flan-t5-base (~250 MB fp32 / ~600 MB after fine-tune) |
| Method | Full fine-tune seq2seq |
| Size | ~600 MB fp16 target |
| Input cap | 1024 tokens (T5 native; chunking deferred) |
| Latency | <2 s target |
| Output schema | 8 fields (includes `explanation` + `relatedDocs`) |

## 2. Base model

`google/flan-t5-base` — encoder-decoder, ~250M params. Instruction-tuned
upstream which helps the structured-output task converge faster than vanilla
t5-base. fp16 packaging targets ~600 MB on disk; INT8 quantization is deferred.

Smoke profile keeps the same base — no scale ladder for seq2seq. Smoke
trims epochs + dataset.

## 3. Training objective

Seq2seq conditional generation. **Input**: sanitised spool transcript
(plain text, optionally prefixed with a task marker like `interpret spool:`).
**Target**: the canonical `SpoolInterpretation` JSON serialised as a string.

Loss: standard cross-entropy on target tokens with teacher forcing
(`Seq2SeqTrainingArguments` default). Pad tokens in `labels` masked to `-100`
so they don't contribute to loss.

The runtime greedily decodes the target string and parses it back to the
typed `SpoolInterpretation`. The validator (§14) catches malformed output.

## 4. Expected output format

```json
{
  "summary": "Job FAILED at step COPY01 with S0C7 (data exception).",
  "status": "failed",
  "returnCode": "S0C7",
  "rootCause": "S0C7 on packed-decimal arithmetic — input field contains non-numeric data.",
  "suggestedFix": "Validate input data for packed-decimal fields before arithmetic; recompile with NUMPROC(NOPFD) for development; check upstream extract for trailing blanks.",
  "explanation": "The job entered step COPY01 normally and began processing the input dataset. On the third record the COBOL program attempted a COMPUTE on a packed-decimal field that contained spaces, triggering an S0C7 data exception. Execution halted immediately; subsequent steps were flushed.",
  "relatedDocs": ["abend-s0c7", "cobol-data-exception", "numproc-options"],
  "failureCategory": "execution_abend",
  "confidence": 0.94
}
```

New in v2:
- `explanation` — 2-4 sentence narrative walking through the chain of events
  that led to the outcome. Distinct from `summary` (which is a 1-sentence
  recap). Required when `status != "completed"`; optional otherwise.
- `relatedDocs` — string array of internal doc keys per failure category
  (e.g. `abend-s0c7`, `dataset-not-cataloged`). Keys, not URLs — the
  frontend maps them to actual help links at render time.

## 5. Bounded vocabulary

From `examples/spool-interpreter/datasets/node_contracts.json`:

- `status_values`: `["completed", "failed", "abended", "skipped", "running"]`
- `failure_categories`: 8 values (dataset/allocation/permission/jcl-syntax/
  utility-parameter/execution-abend/scheduler/other)
- `related_docs_catalog`: per-category doc-key suggestions the seed builder
  draws from.

`returnCode` is free-text (MVS codes like `S0C7`, `RC=08`) so it has no
bounded enum.

## 6. Dataset format

JSONL — each row:

```json
{
  "sample_id": "syn-s0c7-001",
  "source": "synthetic:template",
  "category": "execution_abend",
  "request": "<sanitised spool transcript, multi-line>",
  "expected_interpretation": {
    "summary": "...",
    "status": "failed",
    "returnCode": "S0C7",
    "rootCause": "...",
    "suggestedFix": "...",
    "explanation": "...",
    "relatedDocs": ["abend-s0c7"],
    "failureCategory": "execution_abend",
    "confidence": 0.93
  },
  "split": "train"
}
```

The seq2seq dataset class concatenates the request as the source and
serialises `expected_interpretation` as compact JSON for the target.

## 7. Splits

80 / 10 / 10 by sample-id hash (same as v1; deterministic across reruns).

## 8. Dataset size targets

| Stage | Samples |
|---|---|
| Smoke | 100-200 |
| MVP | 1500+ |
| Internal | 3000+ |
| Stronger | 6000+ |

flan-t5 needs more data than the JCL classifier because the output space
is open-ended text (within JSON braces). The structured constraints help
convergence but don't shrink the target distribution the way 8-way
classification does.

## 9. System prompt

flan-t5 doesn't take a system prompt slot the way chat models do. The
source-side task marker (`interpret spool: <transcript>`) is fixed in
`prompt_spec.json` and prepended at both training and inference. The
runtime's `prompt_spec.json` lookup table still ships so the manifest
file list stays uniform with the other two models.

## 10. Training method

**Full fine-tune** — no LoRA. flan-t5-base at 250 M parameters fine-tunes
comfortably on a 16-32 GB Mac. The resulting safetensors merge straight
into the package (no LoRA-adapter merge step).

## 11. Training config

```yaml
training:
  model_id: google/flan-t5-base
  source_max_len: 1024        # T5 native cap; chunking deferred to v3
  target_max_len: 512         # JSON target rarely exceeds 300 tokens
  batch_size: 8
  grad_accum: 2
  learning_rate: 3.0e-5
  epochs: 6
  weight_decay: 0.01
  warmup_ratio: 0.06
  precision: bf16
  grad_checkpointing: false
  eval_steps: 9999            # post-train eval only
  save_steps: 250
  logging_steps: 25
  generation:
    num_beams: 1              # greedy at inference; beam search deferred
    max_new_tokens: 512
```

Smoke overrides reduce to `epochs: 1, max_steps: 8, batch_size: 2`.

## 12. Training environment

- Python 3.13, `transformers`, `safetensors`.
- flan-t5 lives in `transformers.T5ForConditionalGeneration`; weights load
  cleanly under MPS via `device_map="mps"` with the standard fp32 →
  bf16 autocast.

## 13. Tokenizer

flan-t5's SentencePiece tokenizer ships with the base model. No custom
pre-tokenizer (unlike JCL — spool transcripts are conventional text). The
SentencePiece model is included in the package as `tokenizer.json` (via
`AutoTokenizer.save_pretrained(use_fast=True)`).

## 14. Validation requirements

`maatml.validation.spool_validator` extends from the v1 six-layer gate
with two additional layers:

| Layer | Check |
|---|---|
| 1 | JSON parse |
| 2 | Schema conformance |
| 3 | Status enum (`completed`/`failed`/`abended`/`skipped`/`running`) |
| 4 | Failure-category enum (12 values) |
| 5 | Field-shape consistency (`returnCode` shape, `confidence` range) |
| 6 | Free-text non-empty (`rootCause`, `suggestedFix`) |
| **7** | `explanation` non-empty when `status != "completed"` |
| **8** | `relatedDocs` is an array (possibly empty) of strings |

Schema bump: `spool_interpretation_schema.json` adds `explanation` and
`relatedDocs` as optional properties.

## 15. Evaluation metrics

Required gates (test split):

| Metric | Target |
|---|---|
| `json_parse_rate` | ≥0.99 |
| `schema_conformance_rate` | ≥0.99 |
| `status_accuracy` | ≥0.90 |
| `failure_category_accuracy` | ≥0.80 |
| `return_code_match_rate` | ≥0.85 |
| `explanation_present_rate` | ≥0.95 (new in v2) |
| `related_docs_coverage_rate` | ≥0.80 (new in v2) |
| `p95_latency_ms` | <2000 |

## 16. Test prompt set

`examples/spool-interpreter/datasets/samples/test_prompt_set.jsonl` — fixed
anchors that survive corpus regeneration. Covers every `failureCategory`
plus two `completed` samples.

## 17. Repair dataset

Not separately materialised. Repair-style data (malformed transcripts that
should still parse to a coherent interpretation) lives inline in the
synthetic templates — each category has 1-2 templates that exercise
truncated / noisy spool fragments.

## 18. Artifact requirements

Training produces a checkpoint under `output/checkpoints/<run-name>/`:
`model.safetensors` (flan-t5-base encoder+decoder), `config.json`, the
SentencePiece `tokenizer.json` (+ `spiece.model` if not embedded),
`prompt_spec.json`, plus the committed `spool_interpretation_schema.json`
and `node_contracts.json`. Packaging/export of Hub-ready artifacts from
checkpoints is future work.

## 19. Versioning

Bump `version` in `model.yml` (semver) on each retrain that changes behaviour
or schema. Checkpoint run names use `name@version`.

## 20. First training milestone

Success criteria for the first end-to-end run:
- Training completes without OOM / NaN gradients.
- `eval_loss` < 1.0 at end of training.
- All eval gates in §15 met on the test split.

## 21. Recommended sequence

```bash
# 1. Regenerate seed samples with the v2 schema fields.
.venv/bin/python examples/spool-interpreter/scripts/build_seeds.py --target 1500 \
    --out examples/spool-interpreter/datasets/samples/seed_samples.jsonl

# 2. Train.
maatml prepare examples/spool-interpreter/
maatml train examples/spool-interpreter/ --smoke  # ~1 min sanity check
maatml train examples/spool-interpreter/          # ~30 min on M5 Max

# 3. Evaluate.
maatml evaluate examples/spool-interpreter/
```
