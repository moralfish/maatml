# flow-ml

flow-ml is the **Flow Studio Model Hub** — the registry/distribution source for
the local AI model artifacts Flow Studio runs, and the workspace that trains the
task models. Sibling repo `../flow-studio` is the consumer (desktop app + hub
client + the managed `llama-server` GGUF sidecar).

## What this repo does

Trains three task models, **each a different architecture**, dispatched by
`model.yml::architecture`:

| Model | Task | Architecture | Base |
|---|---|---|---|
| jcl-validator | classify JCL syntax errors | 4-head classifier | ModernBERT-base |
| spool-interpreter | summarise z/OS spool output as JSON | seq2seq | flan-t5-base |
| flow-graph-generator | NL request → Flow Graph JSON | generative (LoRA SFT) | Qwen3-1.7B |

Each model folder is self-describing: `model.yml` (config), `datasets/` (seeds +
prompt_spec + schema + samples), `output/` (prepared splits + checkpoints + eval
reports).

> **Runtime + export.** Flow Studio's local runtime is the `llama-server` GGUF
> sidecar; the in-process Candle task-model path has been removed (see
> flow-studio `docs/architecture/model-runtime.md`, `model-hub.md`). As the Hub,
> flow-ml will catalog `gguf` / `mlx` / `safetensors` artifacts — producing those
> from the trained checkpoints is **future work**, not wired up here yet (E2).

## Workflow

```bash
flow_ml prepare  models/<name>/             # synth/load corpus → output/prepared/{train,val,test}.jsonl
flow_ml train    models/<name>/ [--smoke]   # → output/checkpoints/
flow_ml evaluate models/<name>/             # → output/eval/{ckpt}.{json,md}
```

CLI dispatcher: [src/flow_ml/cli.py](src/flow_ml/cli.py). Each subcommand routes
by `model.yml.task` **and** `model.yml.architecture`.

## Architecture dispatch

```
task=jcl_validation       + architecture=classifier → training/jcl_classifier.py     (ModernBERT)
task=spool_interpretation + architecture=seq2seq    → training/spool_seq2seq.py       (flan-t5)
task=flow_graph_generation (generative, default)     → training/flow_graph_generator.py (Qwen3-1.7B LoRA)
```

The v1 all-generative SFT path for JCL and spool was retired in favour of these
smaller, task-specific architectures.

## Key files (load-bearing)

```
src/flow_ml/
├── cli.py                          # `flow_ml <cmd>` dispatcher (by task + architecture)
├── config.py                       # ModelDefinition (model.yml loader)
├── data/
│   ├── pipeline.py                 # prepare_jcl / prepare_spool / prepare_flow_graph
│   ├── schemas.py                  # sample types for the three tasks
│   ├── sanitizer.py                # PII redaction for spool/jcl
│   └── synthetic/                  # rule-based JCL corpus generator
├── training/
│   ├── jcl_classifier.py           # ModernBERT 4-head classifier
│   ├── spool_seq2seq.py            # flan-t5 seq2seq
│   ├── flow_graph_generator.py     # Qwen3-1.7B LoRA SFT
│   └── sft_base.py                 # shared SFT skeleton
├── validation/                     # per-task out-of-model validators (JCL 6 / FlowGraph 7 / Spool 8)
├── evaluation/runner.py            # evaluate_{jcl,spool,flow_graph}
└── tokenization/jcl_tokenizer.py   # column-aware JCL BPE tokenizer (+ COLUMN_RULES.md)
```

## Local runnability (16 GB target)

- JCL classifier (ModernBERT): small + fast to train and evaluate.
- Spool interpreter (flan-t5): mid-size encoder-decoder.
- Flow Graph Generator (Qwen3-1.7B): fits a 16 GB Mac for training/eval.

## Cross-repo: flow-studio

The DSL spec/grammar that feeds the Flow Graph Generator's prompt originates in
flow-studio; flow-ml never writes back. There is no longer a packaged-model
runtime contract between the repos (the Candle task-model path was removed); the Hub
relationship is artifact distribution (GGUF/MLX/safetensors) via the catalog the
flow-studio client reads — see flow-studio `docs/architecture/model-hub.md`.

## Conventions

- **Determinism**: every prepare uses an explicit `seed` in `model.yml`; two runs
  with the same seed produce byte-identical splits.
- **Validator-gated corpora**: seed builders reject any row that fails the
  per-task out-of-model validator before it lands in `seed_samples.jsonl`.
- **Real validator at eval**: evaluators run the same per-task validator; the
  `flow_dsl_py` PyO3 binding is used when present and degrades gracefully when not.
- **Smoke profile**: tiny bases + a handful of steps for fast pipeline validation.

## Test surface

```bash
.venv/bin/python -m pytest tests/
```

`tests/`: `test_eval_report`, `test_flow_graph_pipeline`, `test_flow_graph_validator`,
`test_jcl_tokenizer`, `test_pipeline`, `test_sanitizer`, `test_schemas`.

## Environment

- Python 3.13+, venv at `.venv/`
- Heavy ML deps under `pip install -e ".[ml]"` (torch, transformers≥5.0, peft, accelerate, …)
- Dev deps under `pip install -e ".[dev]"` (pytest, ruff, mypy)
- `flow_dsl_py` PyO3 binding is optional; built from `flow-studio/crates/flow-dsl-py`
  via `maturin develop --release`. Eval gracefully degrades when missing.
