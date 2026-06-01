# Implementation and Training Plan

This repo is the Python-first training and evaluation workspace for the AI models
that power Flow Studio, and the **Model Hub** source that catalogs and distributes
them. Trained checkpoints become Hub artifacts (`gguf` / `mlx` / `safetensors`)
that flow-studio downloads and runs via its managed `llama-server` sidecar.
Producing those artifacts from checkpoints is future work — the in-process Candle
packaging path has been removed.

## Runtime split

| layer | tooling |
|---|---|
| Training & evaluation | Python — `src/flow_ml/` package, `flow_ml` CLI |
| Hub artifacts | `gguf` / `mlx` / `safetensors` (export from checkpoints — future) |
| Runtime | `llama-server` GGUF sidecar, inside flow-studio |

## Models

| model folder | task key | model id | architecture | base model |
|---|---|---|---|---|
| `models/jcl-validator/` | `jcl_validation` | `jcl-validator:v1` | `classifier` (4-head) | `answerdotai/ModernBERT-base` |
| `models/spool-interpreter/` | `spool_interpretation` | `spool-interpreter:v1` | `seq2seq` | `google/flan-t5-base` |
| `models/flow-graph-generator/` | `flow_graph_generation` | `flow-graph-generator:v1` | `generative` (LoRA SFT) | `Qwen/Qwen3-1.7B` |

**Each model uses a different architecture** — the CLI dispatches by
`model.yml::architecture` at every boundary (the v1 all-generative SFT path for
JCL and spool was retired for smaller/faster task-specific architectures). Each
model emits a fixed JSON shape gated by a per-task out-of-model validator. Per-model
details live in `models/<name>/training_instructions.md`.

## `model.yml` — single source of truth

Every model folder contains exactly one `model.yml`. It drives every CLI command:

```yaml
name: <folder-name>
model_id: <id>
task: <task-key>          # dispatch key for the CLI
architecture: <classifier|seq2seq|generative>   # selects the trainer
version: v1
base_model: <hf-repo>

data:                     # inputs for `flow_ml prepare`
  seed: 7331
  seed_samples: datasets/samples/seed_samples.jsonl
  split_ratios: [0.8, 0.1, 0.1]

training:                 # inputs for `flow_ml train` (typed per architecture)
  model_id: <hf-repo>
  max_input_tokens: 1024
  batch_size: 8
  grad_accum: 2
  learning_rate: 2.0e-5
  epochs: 4
  # `lora:` block applies to the generative trainer only

smoke:                    # overrides for `flow_ml train --smoke`
  base_model: <tiny-hf-repo>
  max_steps: 6

packaging:                # retained for the future export path (not used by the CLI today)
  weights_dtype: f16      # f32 | f16 | bf16
  confidence_thresholds:
    high: 0.9
    low: 0.6
```

`src/flow_ml/config.py` owns the schema (`ModelDefinition` + `PackagingSpec`) and
`load_model_def()`. The `data:` / `training:` / `smoke:` sections are
`dict[str, Any]` so each trainer typechecks its own subset.

## Python package layout

```
src/flow_ml/
  __init__.py
  config.py           # ModelDefinition, PackagingSpec, load_model_def()
  cli.py              # Typer app — prepare, train, evaluate, plan
  data/
    pipeline.py        # prepare_jcl / prepare_spool / prepare_flow_graph
    schemas.py         # Pydantic sample types for all three tasks
    sanitizer.py       # PII / secret redaction applied before tokenization
    sanitization.yaml  # redaction rule definitions
    synthetic/
      jcl_generator.py   # synthetic JCL corpus from templates + defect injection
  training/
    jcl_classifier.py        # ModernBERT 4-head classifier trainer
    spool_seq2seq.py         # flan-t5 seq2seq trainer
    flow_graph_generator.py  # Qwen3-1.7B + LoRA generative trainer
    sft_base.py              # shared SFT skeleton (collator, render, train loop)
  tokenization/
    jcl_tokenizer.py   # column-aware JCL BPE tokenizer
    COLUMN_RULES.md    # JCL column-sensitivity rules
    strategy.md        # tokenization decisions
  validation/
    flow_graph_validator.py # Flow Graph 7-layer out-of-model validator
    jcl_validator.py        # JCL validation result validator (6 layers)
    spool_validator.py      # Spool interpretation validator (8 layers)
  evaluation/
    runner.py        # evaluate_jcl / evaluate_spool / evaluate_flow_graph
  utils/
    io.py            # read_yaml / read_json / write_json / iter_jsonl / write_jsonl / stable_hash
```

## CLI lifecycle

All commands take a model folder (containing `model.yml`) as their primary
argument. Outputs land under `<model-dir>/output/` (gitignored).

```
flow_ml prepare  <model-dir>
flow_ml train    <model-dir> [--smoke] [--limit N] [--seed S] [--device auto|mps|cpu|cuda]
flow_ml evaluate <model-dir> [--checkpoint PATH] [--split test] [--baseline PATH] [--max-input-tokens N]
flow_ml plan     <model-dir>
```

Typical end-to-end run:

```
prepare  →  train --smoke  →  train  →  evaluate
```

## Data pipeline

### JCL Validator (`prepare_jcl`)

Generates a fully synthetic corpus from the JCL templates in `datasets/templates/`.
Defect injectors produce labelled samples for each error category, plus a valid
set, balanced 50/50 valid/error. Splits: 80/10/10. The custom column-aware JCL
tokenizer is trained separately (one-shot) and committed alongside the model.

### Spool Interpreter, Flow Graph Generator (`prepare_spool`, `prepare_flow_graph`)

Load hand-authored seed samples from `datasets/samples/seed_samples.jsonl` and
apply hash-stable deterministic 80/10/10 splitting. Each row carries
`{sample_id, source, category, request, expected_output, split?}`; samples are
authored by hand and human-reviewed before they land in the corpus.

All three prepare functions write `output/prepared/{train,val,test}.jsonl`.

## Training

Each trainer reads its section from `model.yml` into a typed config and runs a
different architecture:

| trainer | architecture | adapter | loss |
|---|---|---|---|
| `jcl_classifier` | ModernBERT 4-head sequence classifier | full fine-tune | weighted cross-entropy over the four heads (validity / error_code / severity / line) |
| `spool_seq2seq` | flan-t5 encoder-decoder | full fine-tune | seq2seq cross-entropy over the target JSON |
| `flow_graph_generator` | Qwen3-1.7B causal LM | LoRA r=16 α=32 | CLM (labels masked over system + user; unmasked over the assistant JSON) |

All trainers set `dataloader_num_workers=0` and use `eval_steps: 9999` to disable
mid-training evaluation on MPS (the unified-memory allocator does not release
val-set tensors between eval and training). `grad_checkpointing` defaults to
`false`. The generative trainer runs fp32 (with bf16 autocast disabled) for
numerical headroom on its longer, JSON-heavy sequences; the classifier and
seq2seq trainers run bf16 autocast over fp32 master weights.

## Evaluation

`evaluate_*` functions in `evaluation/runner.py` load a checkpoint, run inference
on the test split, and write `output/eval/<run-name>.{json,md}`.

| task | metrics |
|---|---|
| `jcl_validation` | json_parse_rate, schema_conformance_rate, severity_accuracy, code_accuracy, valid_flag_accuracy, line_within_3_accuracy |
| `spool_interpretation` | json_parse_rate, schema_conformance_rate, failure_category_accuracy, return_code_accuracy, status_accuracy |
| `flow_graph_generation` | json_parse_rate, schema_conformance_rate, node_type_validity_rate, edge_ref_validity_rate, node_contract_validity_rate, security_policy_pass_rate, forbidden_rejection_rate (= 1.00) |

Each evaluator runs the per-task out-of-model validator over the model's output
(JCL 6 layers / FlowGraph 7 / Spool 8). The `--baseline PATH` option loads a
previous report JSON and adds a `baseline_delta` section to the new report.

## Apple Silicon / MPS notes

- `bf16` autocast with fp32 master weights for the classifier and seq2seq
  trainers; the generative trainer runs fp32 (loading weights AT bf16 + autocast
  bf16 has produced NaN gradients on MPS).
- `dataloader_num_workers=0` everywhere (multi-worker + MPS deadlocks via fork pickling).
- The generative trainer forces `attn_implementation="eager"`.
- `eval_steps: 9999` disables mid-training eval on MPS (the allocator does not
  release val-set tensors between eval and training steps).
- `PYTORCH_ENABLE_MPS_FALLBACK=1` is set automatically by the CLI for unsupported ops.

## Core principles

- Structured outputs first — every model emits a fixed JSON schema, validated at eval time.
- Bounded model scope — each model does exactly one thing.
- Deterministic fallback behavior — if a model output fails schema validation, the consumer falls back to a safe default rather than surfacing raw model output.
- Sanitized inputs before inference — `sanitizer.py` strips PII and secrets at the data-prep stage; the same rules apply at inference for the transcript-like tasks.
- Fixed benchmark sets before UI integration — held-out test prompts are never regenerated; metrics are gated before a model ships.
