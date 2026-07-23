# maatml

[![PyPI](https://img.shields.io/pypi/v/maatml.svg)](https://pypi.org/project/maatml/)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-blue.svg)](pyproject.toml)
[![CI](https://github.com/moralfish/maatml/actions/workflows/ci.yml/badge.svg)](https://github.com/moralfish/maatml/actions/workflows/ci.yml)

**MaatML** fine-tunes small, task-specific models across **text, vision, and
vision-language**, and takes them from experimentation to production through a
single declarative `model.yml`: **prepare → train → evaluate → export → serve**.
Licensed under **Apache-2.0**.

**What makes it different:** correctness is checked *outside* the model by
**validators**. The same validator gates your synthetic **data**, your
**evaluation**, and your **live inference**, so a MaatML model ships with a
contract, not just weights. That validator-gated *data → eval → serving* loop,
now across modalities, is what general fine-tuning tools leave out.

Site: [maatml.pages.dev](https://maatml.pages.dev) ·
PyPI: [`maatml`](https://pypi.org/project/maatml/) ·
Source: [github.com/moralfish/maatml](https://github.com/moralfish/maatml)


## Installation

```bash
python -m venv .venv
source .venv/bin/activate

# Library + CLI (no torch)
pip install maatml

# Training / evaluation stack
pip install "maatml[ml]"

# Optional extras
pip install "maatml[ml,cuda]"    # QLoRA on NVIDIA CUDA (bitsandbytes)
pip install "maatml[ml,pref]"    # DPO / ORPO (TRL)
pip install "maatml[ml,vision]"  # torchvision + ONNX (examples/vision)
pip install "maatml[vllm]"       # Linux-only vLLM serving (examples/vision-vlm)
pip install "maatml[teacher]"    # OpenAI-compatible teacher for datagen
pip install "maatml[docs]"       # mkdocs site
```

Then:

```bash
maatml --help
maatml scaffold ~/models/my-task --architecture causal_sft --name my-task
maatml validate ~/models/my-task
```

For contributing to this repository (editable install), see
[CONTRIBUTING.md](CONTRIBUTING.md).

## Example models

Six reference models share the identical folder layout and CLI, from a
one-command support-ticket triage to a vLLM-servable vision-language model:

| Model | Task | Architecture | Base |
|-------|------|--------------|------|
| [Support Ticket Triage](examples/support-ticket-triage/) | triage → JSON | `causal_sft` (LoRA) | Qwen3-0.6B |
| [Vision VLM](examples/vision-vlm/) | describe a scene image | `vlm_sft` (vLLM-servable) | SmolVLM-256M-Instruct |
| [Vision](examples/vision/) | scene + detect + pose | `vision_multitask` | MobileNetV3-Large |
| [Vision Describer](examples/vision-describer/) | caption from vision JSON | `seq2seq` | flan-t5-small |
| [JCL Validator](examples/jcl-validator/) | `jcl_validation` | `classifier` (4-head) | ModernBERT-base |
| [Spool Interpreter](examples/spool-interpreter/) | `spool_interpretation` | `seq2seq` | flan-t5-base |

Any directory with a valid `model.yml` works the same way: install maatml from
PyPI and point the CLI at the folder. Scaffold a new model folder with
`maatml scaffold`.

## Where MaatML fits

MaatML **builds on** Hugging Face `transformers` / `peft` / `trl` and does the
one thing those building blocks leave to you: it wraps them in an opinionated,
validator-gated lifecycle for **small** task-specific models you can train on a
laptop and deploy to the edge or vLLM.

- **Complements** general fine-tuning tools (Axolotl, LLaMA-Factory, Unsloth,
  TRL) rather than competing on scale. Reach for those for large models,
  multi-node training, RL, or broad model coverage.
- **Runs its own fixed lifecycle** (`prepare → train → evaluate → export →
  verify` / `serve`), and will add a single-command `maatml run` for that path
  (see [ROADMAP.md](ROADMAP.md)). It is **not** a general-purpose workflow
  scheduler: no triggers, no arbitrary shell/Python steps, no remote executors.
  Drop `maatml train` into MLflow / Prefect / Metaflow when you need that.
- **Its niche:** local-first, multimodal, structured-output models with
  correctness gated *outside* the model, from data generation through serving.

## Requirements

- **Python** 3.10+ (developed against 3.13)
- **OS** macOS, Linux (Windows untested)
- **Disk / memory** ~3 GB for the ML stack; 16 GB unified memory is the design
  target for local training

## CLI overview

Each command takes a model folder (containing `model.yml`) as its first
argument. Outputs land under `<model-folder>/output/` (gitignored).

```
maatml prepare  <model-dir>                                   # builds output/prepared/{train,val,test}.jsonl
maatml train    <model-dir> [--smoke] [--resume auto|PATH] [--set K=V]
maatml sweep    <model-dir> --param K=a,b [--metric NAME] [--smoke] [--max-trials N]
maatml evaluate <model-dir> [--checkpoint X] [--gate]         # writes output/eval/<run>.{json,md}
maatml export   <model-dir> [--checkpoint X] [--format safetensors|gguf|mlx|onnx] [--parity]
maatml verify   <export-dir-or-manifest>                      # sha256 check vs manifest.json
maatml serve    <model-dir> [--checkpoint X] [--host HOST] [--port N]  # JSON inference API
maatml datagen  <model-dir> [--target N] [--teacher]          # validator-gated seed generation
maatml ingest   <model-dir> --input PATH [--map field=col] [--sanitize tag]
maatml runs     <model-dir>                                   # list training runs
maatml plan     <model-dir>                                   # prints the prepare/train/eval/export plan
maatml plugins                                                # list discovered trainers/validators/metrics
maatml scaffold <dir> --architecture causal_sft|dpo [--name X]
maatml validate <model-dir>                                   # check model.yml + registered plugins
```

Multi-GPU (CUDA): `accelerate launch -m maatml.cli train <model-dir>/` or
`torchrun --nproc_per_node=N -m maatml.cli train <model-dir>/`.

QLoRA (CUDA + `[cuda]`): set `training.quantization.load_in_4bit: true` in
`model.yml`. Preference data: `dataset.format: preference_jsonl` with
`{prompt, chosen, rejected}` rows; scaffold with `--architecture dpo`.

Export defaults to a safetensors bundle + `manifest.json`. GGUF/MLX need
external tooling (`llama.cpp` convert / `mlx_lm`). Pin base-model revisions
with `training.model_revision`.

Docs: [maatml.pages.dev](https://maatml.pages.dev) · Roadmap: [ROADMAP.md](ROADMAP.md) ·
In-repo docs: `docs/` (`pip install "maatml[docs]"` then `mkdocs serve`).

Run `maatml <command> --help` for options.

## End-to-end example (Support Ticket Triage)

The quickest model to run: a LoRA fine-tune of Qwen3-0.6B that turns a raw
support ticket into `{priority, category, team, summary}` JSON, gated by a
schema validator plus a `category → team` routing contract enforced *outside*
the model. Every reference model now registers a validator and declares
`evaluation.gates`.

```bash
git clone https://github.com/moralfish/maatml.git
cd maatml
pip install "maatml[ml]"

maatml prepare  examples/support-ticket-triage/
maatml train    examples/support-ticket-triage/ --smoke   # fast pipeline check
maatml train    examples/support-ticket-triage/
maatml evaluate examples/support-ticket-triage/ --gate    # enforce eval gates
maatml serve    examples/support-ticket-triage/           # JSON inference API
```

For a **multimodal** walkthrough (image → description, servable by vLLM) see
[examples/vision-vlm/](examples/vision-vlm/). For an **advanced** model with a
custom column-aware tokenizer, see [examples/jcl-validator/](examples/jcl-validator/).
Its tokenizer is built once up front:

```bash
python examples/jcl-validator/scripts/build_seeds.py --target 10000 \
  --out examples/jcl-validator/datasets/samples/tokenizer_corpus.jsonl
python examples/jcl-validator/scripts/build_tokenizer.py
```

## Batch scripts

```bash
# Deterministic seed corpora (no API calls)
python examples/jcl-validator/scripts/build_seeds.py
python examples/spool-interpreter/scripts/build_seeds.py

# Train / evaluate example models
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
examples/                   # reference task models (plugins + data)
  jcl-validator/
    model.yml               # single source of truth
    jcl_plugin/             # validator, metrics, predictor, tokenizer, …
    datasets/               # schemas, prompt specs, seed samples
    scripts/                # seed + tokenizer builders
    output/                 # gitignored prepared / checkpoints / eval
  spool-interpreter/        # same layout (uses core seq2seq)
  support-ticket-triage/

src/maatml/                 # core framework (architectures, CLI, harnesses)
scripts/                    # batch train/eval/validate
tests/                      # core unit tests
```

## Development

See [CONTRIBUTING.md](CONTRIBUTING.md) for setup, PR expectations, DCO
sign-off, and versioning policy. AI coding agents: [AGENTS.md](AGENTS.md).

Community: [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md) · Security:
[SECURITY.md](SECURITY.md) · Changes: [CHANGELOG.md](CHANGELOG.md)

## Licensing

- **maatml** is licensed under the [Apache License 2.0](LICENSE).
- This repository **does not redistribute base-model weights**, only Hugging
  Face Hub IDs. The reference bases (ModernBERT, flan-t5, and related Apache-2.0
  models such as Qwen3 when used) are Apache-2.0; **your fine-tuned checkpoints
  inherit the base model's license terms**.
- Seed corpora are **fully synthetic**, produced by deterministic builders under
  `examples/*/scripts/`, with no proprietary mainframe dumps shipped.
