# flow-ml

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-blue.svg)](pyproject.toml)
[![CI](https://img.shields.io/badge/CI-GitHub_Actions-blue.svg)](.github/workflows/ci.yml)

A **training and fine-tuning framework** for task-specific language models.
Each model is a standalone folder with a `model.yml` that drives the full
lifecycle: prepare → train → evaluate. Licensed under **Apache-2.0**.

## Reference models

| Model | Task | Architecture | Base |
|-------|------|--------------|------|
| [JCL Validator](models/jcl-validator/) | `jcl_validation` | `classifier` (4-head) | ModernBERT-base |
| [Spool Interpreter](models/spool-interpreter/) | `spool_interpretation` | `seq2seq` | flan-t5-base |

Any directory with a valid `model.yml` works the same way — install flow-ml via
pip and point the CLI at the folder. Scaffold a new model folder with
`flow_ml scaffold` (see [CONTRIBUTING.md](CONTRIBUTING.md)).

## Requirements

- **Python** 3.10+ (developed against 3.13)
- **OS** macOS, Linux (Windows untested)
- **Disk / memory** ~3 GB for the ML stack; 16 GB unified memory is the design
  target for local training

## Installation

```bash
python -m venv .venv
source .venv/bin/activate

# Library + CLI (no torch)
pip install -e ".[dev]"

# Training / evaluation extras
pip install -e ".[dev,ml]"
```

## CLI overview

Each command takes a model folder (containing `model.yml`) as its first
argument. Outputs land under `<model-folder>/output/` (gitignored).

```
flow_ml prepare  <model-dir>                                   # builds output/prepared/{train,val,test}.jsonl
flow_ml train    <model-dir> [--smoke] [--limit N] [--seed S] # writes output/checkpoints/<run-name>/
flow_ml evaluate <model-dir> [--checkpoint X] [--split test]  # writes output/eval/<run-name>.{json,md}
flow_ml plan     <model-dir>                                   # prints the prepare/train/eval command plan
flow_ml plugins                                                # list discovered trainers/validators/metrics
flow_ml scaffold <dir> --architecture causal_sft [--name X]   # create a new model folder
flow_ml validate <model-dir>                                   # check model.yml + registered plugins
```

Run `flow_ml <command> --help` for options.

## End-to-end example (JCL Validator)

```bash
# 1. Seed samples live at
#    models/jcl-validator/datasets/samples/seed_samples.jsonl
#    (or regenerate via scripts/build_jcl_seeds.py)

# 2. Prepare splits
flow_ml prepare models/jcl-validator/

# 3. Smoke training, then full training
flow_ml train models/jcl-validator/ --smoke
flow_ml train models/jcl-validator/

# 4. Evaluate the most recent checkpoint
flow_ml evaluate models/jcl-validator/
```

JCL training also needs the custom tokenizer once:

```bash
python scripts/build_jcl_seeds.py --target 10000 \
  --out models/jcl-validator/datasets/samples/tokenizer_corpus.jsonl
python -m flow_ml.tokenization.jcl_tokenizer train \
  --corpus models/jcl-validator/datasets/samples/tokenizer_corpus.jsonl \
  --out models/jcl-validator/datasets/tokenizer.json
```

## Batch scripts (JCL + Spool)

```bash
# Deterministic seed corpora (no API calls)
python scripts/build_jcl_seeds.py
python scripts/build_spool_seeds.py

# Train / evaluate both models
python scripts/train_all.py --smoke
python scripts/train_all.py
python scripts/evaluate_all.py
```

## Apple Silicon / MPS notes

- Default precision is **bf16** autocast with fp32 master weights.
- Trainers set `eval_steps: 9999` to disable mid-training eval on MPS (unified
  memory does not release val-set tensors between eval and training).
- `grad_checkpointing` defaults to `false`; `dataloader_num_workers=0`
  everywhere (multi-worker + MPS can deadlock via fork pickling).
- `PYTORCH_ENABLE_MPS_FALLBACK=1` is set by the CLI for unsupported ops.

## Repository layout

```
models/                     # one folder per model
  jcl-validator/
    model.yml               # single source of truth
    README.md
    datasets/               # schemas, prompt specs, seed samples
    output/                 # gitignored prepared / checkpoints / eval
  spool-interpreter/        # same layout

src/flow_ml/                # Python package (config, CLI, trainers, validators)
scripts/                    # seed builders + batch train/eval
tests/
```

## Development

See [CONTRIBUTING.md](CONTRIBUTING.md) for setup, PR expectations, DCO
sign-off, and versioning policy. AI coding agents: [AGENTS.md](AGENTS.md).

Community: [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md) · Security:
[SECURITY.md](SECURITY.md) · Changes: [CHANGELOG.md](CHANGELOG.md)

## Licensing

- **flow-ml** is licensed under the [Apache License 2.0](LICENSE).
- This repository **does not redistribute base-model weights** — only Hugging
  Face Hub IDs. The reference bases (ModernBERT, flan-t5, and related Apache-2.0
  models such as Qwen3 when used) are Apache-2.0; **your fine-tuned checkpoints
  inherit the base model's license terms**.
- Seed corpora are **fully synthetic**, produced by deterministic builders under
  `scripts/` — no proprietary mainframe dumps are shipped.
