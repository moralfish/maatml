# flow-ml

**flow-ml is the Flow Studio Model Hub** — the registry/distribution source for
the AI model artifacts Flow Studio runs locally, and the workspace that trains
the task models. The Hub serves general models in `gguf` / `mlx` / `safetensors`
formats; Flow Studio's hub client downloads them and runs inference on-device via
the managed `llama-server` sidecar (`~/.flow-studio/llms/`). Only artifact bytes
+ metadata cross the wire, never inference data.

This repo trains the task models:
- per-model definition files (`models/<name>/model.yml`)
- dataset schemas, prompt specs, and seed corpora
- sanitization, augmentation, and split preparation
- training and evaluation

> **Runtime + export.** Flow Studio's local runtime is the `llama-server` GGUF
> sidecar (see flow-studio `docs/architecture/model-runtime.md`). The in-process
> Candle task-model path has been removed. Producing Hub artifacts
> (GGUF / MLX / safetensors) for the trained models is **future work** and is not
> wired up here yet (roadmap epic **E2**).

## Task models

Each model is self-contained under `models/`, and **each uses a different
architecture** — dispatched at every CLI boundary by `model.yml::architecture`:

| model | task (manifest) | model id | base | architecture |
|-------|-----------------|----------|------|--------------|
| [JCL Validator](models/jcl-validator/) | `jcl_validation` | `jcl-validator:v1` | ModernBERT-base | `classifier` (4-head) |
| [Spool Interpreter](models/spool-interpreter/) | `spool_interpretation` | `spool-interpreter:v1` | flan-t5-base | `seq2seq` |
| [Flow Graph Generator](models/flow-graph-generator/) | `flow_graph_generation` | `flow-graph-generator:v1` | Qwen3-1.7B + LoRA | `generative` (SFT) |

## Repository layout

```
models/                     # one folder per model - the only thing that matters
  jcl-validator/
    model.yml               # SINGLE source of truth: data + training + smoke
    README.md               # data strategy, training notes, evaluation criteria
    datasets/               # tracked: schemas, prompt_specs, seed samples, JCL templates
      schema.json
      label_taxonomy.md
      templates/
      samples/
        seed_samples.jsonl  # tracked (small, hand-written)
    output/                 # gitignored:
      # prepared/{train,val,test}.jsonl
      # checkpoints/<run-name>/
      # eval/<report>.{json,md}

  spool-interpreter/      ...   # same layout
  flow-graph-generator/   ...   # same layout

src/flow_ml/                # Python package
  config.py                 # ModelDefinition + load_model_def(model_dir)
  cli.py                    # flow_ml CLI (prepare / train / evaluate / plan)
  data/
    pipeline.py             # prepare_jcl / prepare_spool / prepare_flow_graph
    sanitization.yaml       # global PII/secret redaction rules
    sanitizer.py
    schemas.py
    synthetic/
      jcl_generator.py      # synthetic JCL corpus
  training/
    jcl_classifier.py       # ModernBERT 4-head classifier trainer
    spool_seq2seq.py        # flan-t5 seq2seq trainer
    flow_graph_generator.py # Qwen3-1.7B + LoRA generative trainer (FlowGraphDto)
    sft_base.py             # shared SFT skeleton (collator, render, train loop)
  validation/               # per-task out-of-model validators (JCL 6 / FlowGraph 7 / Spool 8 layers)
    flow_graph_validator.py
    jcl_validator.py
    spool_validator.py
  evaluation/runner.py
  tokenization/jcl_tokenizer.py  # column-aware JCL BPE tokenizer
  utils/io.py
scripts/                    # CLI shims + deterministic seed builders
tests/                      # collected on every push and PR
```

## Requirements

- **Python** 3.10+ (developed against 3.13)
- **OS** macOS, Linux. Windows untested.
- **Disk** ~3 GB for the ML stack alone; add base-model weights as needed
  (Qwen3-1.7B ~3.4 GB at fp16, Qwen3-0.6B ~1.5 GB for smoke runs).
- **Memory** 16 GB unified memory is the design target.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e ".[dev]"          # base profile: pydantic, pyyaml, rich, typer, pytest
pip install -r requirements-ml.txt  # ML profile: torch, transformers, peft, ...
```

## CLI

Each command takes a model folder (containing `model.yml`) as its first
argument. Outputs land under `<model-folder>/output/` (gitignored).

```
flow_ml prepare  <model-dir>                                              # builds output/prepared/{train,val,test}.jsonl
flow_ml train    <model-dir> [--smoke] [--limit N] [--seed S]            # writes output/checkpoints/<run-name>/
flow_ml evaluate <model-dir> [--checkpoint X] [--baseline PATH]          # writes output/eval/<run-name>.{json,md}
                             [--max-input-tokens N] [--split test]
flow_ml plan     <model-dir>                                              # prints the prepare/train/eval command plan
```

Run `flow_ml <command> --help` for full options.

## End-to-end example (Flow Graph Generator)

```bash
# 1. Hand-author seed samples in
#    models/flow-graph-generator/datasets/samples/seed_samples.jsonl

# 2. Prepare splits (writes train/val/test.jsonl)
python -m flow_ml.cli prepare models/flow-graph-generator/

# 3. Smoke training (Qwen3-0.6B, 6 steps, ~5 seconds on M5 Max)
python -m flow_ml.cli train models/flow-graph-generator/ --smoke

# 4. Full training (Qwen3-1.7B, 4-12 epochs depending on dataset size)
python -m flow_ml.cli train models/flow-graph-generator/

# 5. Evaluate the most recent checkpoint
python -m flow_ml.cli evaluate models/flow-graph-generator/
```

## Batch scripts (all three models)

The three models have different architectures but share one CLI lifecycle, so
it's natural to drive them in one pass. Wrappers under [`scripts/`](scripts/)
iterate over `jcl-validator`, `spool-interpreter`, and `flow-graph-generator`,
isolate failures, and print a summary.

### Build (or refresh) the seed corpora

Each builder is deterministic (seeded RNG), template-based, and gates every
generated row through the per-task layer-validator before writing — no API
calls, fully reproducible.

```bash
.venv/bin/python scripts/build_jcl_seeds.py          # ~500 samples, 8 categories
.venv/bin/python scripts/build_spool_seeds.py        # ~500 samples, 13 categories
.venv/bin/python scripts/build_flow_graph_seeds.py   # ~500 samples, 13 categories
```

### Train / evaluate all three

```bash
.venv/bin/python scripts/train_all.py --smoke        # tiny bases, ~30 s total on M5 Max
.venv/bin/python scripts/train_all.py                # full training, ~25 min total
.venv/bin/python scripts/evaluate_all.py             # latest ckpt of each, test split
```

Each task trains into its own `output/checkpoints/`; no cross-contamination.

## Apple Silicon / MPS notes

- Default precision is **bf16** autocast with fp32 master weights; the Flow Graph
  Generator runs **fp32** for numerical headroom on its longer sequences.
- All trainers set `eval_steps: 9999` to disable mid-training eval on MPS (the
  unified-memory allocator does not release val-set tensors between eval and
  training). A single post-training eval pass runs after `trainer.train()` returns.
- `grad_checkpointing` defaults to `false`; `dataloader_num_workers=0` everywhere
  (multi-worker + MPS deadlocks via fork pickling).

## Development

- Lint: `ruff check src tests scripts`
- Tests: `.venv/bin/python -m pytest tests/`
- CI runs `validate_repo.py` + `pytest` on every push and PR.

## Further reading

- Per-model READMEs under [`models/`](models/) - data strategy, training instructions, evaluation criteria.
