# AGENTS.md

Guidance for AI coding agents working in this repository.

## Commands

All commands assume the project venv (`.venv/bin/...`). Training needs the
`[ml]` extra (torch, transformers, peft, tokenizers). CPU-free contributions
can use `pip install -e ".[dev]"`; unit tests run without torch.

```bash
# Install
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,ml]"

# Test / lint (run from repo root)
.venv/bin/python -m pytest tests/
ruff check src tests scripts

# Per-model lifecycle (any standalone folder with model.yml)
.venv/bin/flow_ml scaffold ~/models/my-task --architecture causal_sft
.venv/bin/flow_ml validate ~/models/my-task
.venv/bin/flow_ml prepare  models/<name>/
.venv/bin/flow_ml train    models/<name>/ [--smoke] [--device mps|cpu] [--seed N] [--limit N]
.venv/bin/flow_ml evaluate models/<name>/ [--checkpoint X] [--split test|val]
.venv/bin/flow_ml plugins

# Seed corpus regen (deterministic, validator-gated)
.venv/bin/python scripts/build_jcl_seeds.py   --target 1000
.venv/bin/python scripts/build_spool_seeds.py --target 1500

# Custom JCL BPE tokenizer (required before JCL training)
.venv/bin/python scripts/build_jcl_seeds.py --target 10000 \
    --out models/jcl-validator/datasets/samples/tokenizer_corpus.jsonl
.venv/bin/python -m flow_ml.tokenization.jcl_tokenizer train \
    --corpus models/jcl-validator/datasets/samples/tokenizer_corpus.jsonl \
    --out models/jcl-validator/datasets/tokenizer.json
```

## Architecture

flow-ml is a **plugin-based training/fine-tuning framework**. Core owns
registries, device profiles, generic prepare/train/eval harnesses, and
guards. Task-specific validators/metrics live under `flow_ml.contrib.*` and
register via entry points (`flow_ml.plugins`).

| Location | Architecture | Base |
|---|---|---|
| `models/jcl-validator/` | `classifier` / `multi_head_classifier` | ModernBERT-base |
| `models/spool-interpreter/` | `seq2seq` | flan-t5-base |
| `examples/support-ticket-triage/` | `causal_sft` (LoRA SFT) | Qwen3-0.6B |

**Standalone model folders.** Every path in `model.yml` resolves relative to
the model folder via `ModelDefinition.resolve()`. Core never falls back to
repo-relative paths. `pip install flow-ml` + `flow_ml scaffold ~/anywhere/...`
must work with no checkout present.

**Single source of truth.** `model.yml` carries `architecture`, `dataset:`,
`training:`, `smoke:`, `evaluation:`, `packaging:`. Version is semver
(`0.1.0`); identity is `name@version`.

### Dispatch (registry)

```
architecture → trainer registry (causal_sft / seq2seq / multi_head_classifier)
dataset.format → format registry (jsonl_seed → prepare)
evaluation.validator / metrics / predictor → plugin registries
```

`model.yml` may also list `plugins: [./plugin.py]` for folder-local extensions.

### Device profiles (`flow_ml.device`)

- `mps`: no mid-train eval, workers=0, no grad checkpointing, fp32 master weights
- `cuda`: mid-train eval on, workers>0, checkpointing allowed
- `cpu`: conservative defaults

### Guards (`flow_ml.training.guards`)

- `NanGuardCallback` aborts on non-finite loss/grad_norm
- `ensure_tokenizer_model_contract` + `training.embedding_strategy`
- `run_metadata.json` written into each checkpoint dir

### Out-of-model validators

JCL and Spool validators live under `flow_ml.validation` and are registered
by `flow_ml.contrib.{jcl,spool}`. Shared fence stripping / result types are in
`flow_ml.validation.base`.

## Data flow

```
datasets/samples/seed_samples.jsonl  ← scripts/build_*_seeds.py
        │
        │  flow_ml prepare
        ▼
output/prepared/{train,val,test}.jsonl
        │
        │  flow_ml train (registry by architecture)
        ▼
output/checkpoints/<name@version>/{*.safetensors, tokenizer, run_metadata.json, ...}
        │
        │  flow_ml evaluate
        ▼
output/eval/<run>.{json,md}
```

## Operational notes

- **JCL tokenizer required before training.** If
  `models/jcl-validator/datasets/tokenizer.json` is missing, training falls
  back to the stock ModernBERT tokenizer and eval gates miss. Set
  `training.embedding_strategy: resize` when using the custom tokenizer.
- **Seed regen stamps `family`.** Group-aware splits hash `family` (fallback
  `source` → `sample_id`). Re-run builders after schema changes.
- **Checkpoint selection.** `_latest_checkpoint()` picks the most recently
  modified dir under `output/checkpoints/`. Pass `--checkpoint` explicitly
  when needed.
- **Tests live under `tests/`.** Run from repo root.
