# Implementation and Training Plan

This repo is the Python-first training, evaluation, and packaging workspace for the AI models that power Flow Studio. Models are deployed to Candle (Rust) at runtime via `.fm` archives.

## Runtime split

| layer | tooling |
|---|---|
| Training & evaluation | Python — `src/flow_ml/` package, `flow_ml` CLI |
| Packaging | Python — writes safetensors + tokenizer + manifest into a `.fm` archive |
| Runtime | Candle (Rust) inside the flow-starter orchestration core |

## Models

| model folder | task key | model id | base model | runtime backend |
|---|---|---|---|---|
| `models/jcl-validator/` | `jcl_validation` | `jcl-validator:v1` | `google-bert/bert-base-uncased` | `CandleBackend` |
| `models/spool-interpreter/` | `spool_interpretation` | `spool-interpreter:v1` | `HuggingFaceTB/SmolLM2-360M-Instruct` + LoRA | `CandleGenerativeBackend` |
| `models/dsl-generator/` | `dsl_generation` | `dsl-generator:v1` | `Qwen/Qwen2.5-Coder-7B-Instruct` + LoRA | `CandleGenerativeBackend` |
| `models/agent-planner/` | `agent_planning` | `agent-planner:v1` | `Qwen/Qwen3-4B-Instruct-2507` + LoRA | `CandleGenerativeBackend` |

## `model.yml` — single source of truth

Every model folder contains exactly one `model.yml`. It consolidates what was previously spread across multiple config files and drives every CLI command:

```yaml
name: <folder-name>
model_id: <id>            # what Flow Studio sees
task: <task-key>          # dispatch key for the CLI
runtime: candle
version: v1
base_model: <hf-repo>

data:                     # inputs for `flow_ml prepare`
  seed: 7331
  seed_samples: datasets/samples/seed_samples.jsonl
  split_ratios: [0.6, 0.2, 0.2]
  # jcl: template_dir + n_per_class + n_valid
  # dsl: augment block (target_count, out)

training:                 # inputs for `flow_ml train`
  model_id: <hf-repo>
  max_input_tokens: 1024
  batch_size: 1
  grad_accum: 16
  learning_rate: 1.0e-4
  epochs: 3
  lora:
    enabled: true
    r: 16
    alpha: 32
    target_modules: [q_proj, k_proj, v_proj, o_proj]

smoke:                    # overrides for `flow_ml train --smoke`
  base_model: <tiny-hf-repo>
  max_steps: 6

packaging:                # inputs for `flow_ml package`
  max_input_tokens: 1024
  expected_latency_ms: 2000
  weights_dtype: f16      # f32 | f16 | bf16
  confidence_thresholds:
    high: 0.9
    low: 0.6
```

`src/flow_ml/config.py` owns the schema (`ModelDefinition` + `PackagingSpec`) and `load_model_def()`.

## Python package layout

```
src/flow_ml/
  __init__.py
  config.py           # ModelDefinition, PackagingSpec, load_model_def()
  cli.py              # Typer app — 6 commands (prepare, train, evaluate, package, verify, plan)
  data/
    pipeline.py       # prepare_jcl / prepare_spool / prepare_dsl / prepare_agent
    schemas.py        # Pydantic sample types for all four tasks
    sanitizer.py      # PII / secret redaction applied before tokenization
    sanitization.yaml # redaction rule definitions
    synthetic/
      jcl_generator.py   # synthetic JCL corpus from templates + defect injection
      dsl_generator.py   # rule-based DSL augmenter
  training/
    jcl_validator.py     # multi-head BERT fine-tune
    spool_interpreter.py # SmolLM2 + LoRA
    dsl_generator.py     # Qwen2.5-Coder + LoRA
    agent_planner.py     # Qwen3 + LoRA
  evaluation/
    runner.py        # evaluate_jcl / evaluate_spool / evaluate_dsl / evaluate_agent
    graph_diff.py    # DSL graph structure comparison utilities
  models/
    manifest.py      # ModelManifest + ConfidenceThresholds (runtime contract)
  packaging/
    package_model.py # package_jcl / package_spool / package_dsl / package_agent / verify_package
  tokenization/
    strategy.md      # tokenization decisions
  utils/
    io.py            # read_yaml / read_json / write_json / iter_jsonl / write_jsonl / stable_hash
```

## CLI lifecycle

All commands take a model folder (containing `model.yml`) as their primary argument. Outputs land under `<model-dir>/output/` (gitignored).

```
flow_ml prepare  <model-dir>
flow_ml train    <model-dir> [--smoke] [--limit N] [--seed S] [--device auto|mps|cpu|cuda]
flow_ml evaluate <model-dir> [--checkpoint PATH] [--split test] [--baseline PATH] [--max-input-tokens N]
flow_ml package  <model-dir> [--checkpoint PATH] [--version vN]
flow_ml verify   <fm-or-dir>
flow_ml plan     <model-dir>
```

Typical end-to-end run:

```
prepare  →  train --smoke  →  train  →  evaluate  →  package  →  verify
```

## Data pipeline

### JCL Validator (`prepare_jcl`)

Generates a fully synthetic corpus from the 8 JCL templates in `datasets/templates/`. Defect injectors produce labelled samples for each of the seven error categories, plus a valid set. Default: 2 000 samples per class + 2 000 valid ≈ 16 000 total. Splits: 80 / 10 / 10.

### Spool Interpreter and DSL Generator (`prepare_spool`, `prepare_dsl`)

Load hand-authored seed samples from `datasets/samples/seed_samples.jsonl`, apply hash-stable deterministic splitting. DSL Generator also runs the rule-based augmenter (`dsl_generator.py`) to expand the seed corpus to ~3 500 samples before splitting.

### Agent Planner (`prepare_agent`)

Same as spool / DSL: load seed + eval samples from JSONL, hash-stable split. `datasets/samples/eval_samples.jsonl` is always routed to the test split to keep the benchmark fixed across retrains.

All four prepare functions write `output/prepared/{train,val,test}.jsonl`.

## Training

Each trainer reads its section from `model.yml` into a typed config (`JclTrainConfig`, `SpoolTrainConfig`, `DslTrainConfig`, `AgentTrainConfig`) via Pydantic. The `--smoke` flag calls `model_def.merged_smoke()`, which overlays the `smoke:` block on top of `training:` and swaps in the smaller `smoke.base_model`.

| trainer | architecture | LoRA | loss |
|---|---|---|---|
| `jcl_validator` | BERT + 3 custom heads (seq, cat, line) | no | weighted sum of 3 CE losses |
| `spool_interpreter` | causal LM | yes | CLM (labels masked over prompt) |
| `dsl_generator` | causal LM | yes | CLM (labels masked over prompt) |
| `agent_planner` | causal LM | yes | CLM (labels masked over prompt) |

All trainers use HuggingFace `Trainer` / `TrainingArguments`, set `dataloader_num_workers=0`, and force `attn_implementation="eager"` on generative models (SDPA on MPS has known regressions). `eval_steps: 9999` in generative `model.yml` files disables mid-training evaluation on MPS to avoid the unified-memory runaway.

## Evaluation

`evaluate_*` functions in `evaluation/runner.py` load a checkpoint, run inference on the test split, and write `output/eval/<run-name>.{json,md}`.

| task | metrics |
|---|---|
| `jcl_validation` | seq accuracy + F1, cat macro-F1, line token-F1 |
| `spool_interpretation` | json_validity, category_accuracy, root_cause_rouge_l |
| `dsl_generation` | json_validity, parser_roundtrip, node_count_jaccard, edge_count_match |
| `agent_planning` | json_validity, schema_validity, intent_match, dsl_presence_accuracy, refusal_accuracy, action_jaccard |

The `--baseline PATH` option loads a previous report JSON and adds a `baseline_delta` section to the new report.

## Packaging and the `.fm` archive

`package_*` functions in `packaging/package_model.py`:

1. Copy the checkpoint's `config.json`, `tokenizer.json`, and related files.
2. Merge LoRA adapter weights back into the base model and save as `model.safetensors` (generative models). For JCL Validator, also writes `flow_heads.safetensors` + `labels.json`.
3. Convert weights to the dtype specified by `packaging.weights_dtype` (`f32` default; `f16` for DSL Generator and Agent Planner to keep 7B packages around 14 GB).
4. Write `manifest.json` from `ModelManifest` (model_id, task, runtime, version, base_model, max_input_tokens, expected_latency_ms, confidence_thresholds, weights_dtype).
5. Copy `prompt_spec.json` when present.
6. Zip everything into `<model_id>-<version>.fm` (deflated).

The `.fm` archive is a self-contained import artifact: Flow Studio's Models drawer loads it directly with no additional dependencies.

`verify_package` reloads the unpacked directory (or extracts a `.fm` archive) through `transformers` and runs a one-shot forward pass to confirm the weights and tokenizer are loadable.

## Manifest contract

`src/flow_ml/models/manifest.py` defines `ModelManifest`, which is the shared contract between this Python training workspace and the Candle runtime backends in flow-starter. Any field added here must have a corresponding reader in the Rust side.

## Apple Silicon / MPS notes

- Default precision is `fp32`; `bf16` is reserved for CUDA.
- `dataloader_num_workers=0` everywhere (multi-worker + MPS deadlocks via fork pickling).
- Generative trainers force `attn_implementation="eager"`.
- `eval_steps: 9999` disables mid-training eval on MPS (memory allocator does not release val-set tensors between eval and training steps).
- `PYTORCH_ENABLE_MPS_FALLBACK=1` is set automatically by the CLI for unsupported ops.

## Core principles

- Structured outputs first — every model emits a fixed JSON schema, validated at eval time.
- Bounded model scope — each model does exactly one thing; the runtime dispatches by task key.
- Deterministic fallback behavior — if a model output fails schema validation, the runtime falls back to a safe default rather than surfacing raw model output.
- Sanitized inputs before inference — `sanitizer.py` strips PII and secrets at the data-prep stage; the same rules apply to runtime inputs.
- Fixed benchmark sets before UI integration — `eval_samples.jsonl` is held out from training and never regenerated; metrics are gated before a model ships.
