# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

All commands assume the project venv (`.venv/bin/...`) which carries the `[ml]` extra (torch, transformers, peft, tokenizers). Top-level orchestration runs through the `flow_ml` console-script entry point (installed by `pip install -e .`).

```bash
# Install
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,ml]"

# Test (run from repo root)
.venv/bin/python -m pytest tests/
.venv/bin/python -m pytest tests/test_jcl_tokenizer.py -k pretokenize    # single file / filter

# Lint
ruff check src tests scripts

# Per-model lifecycle (each step takes `models/<name>/`)
.venv/bin/flow_ml prepare  models/<name>/
.venv/bin/flow_ml train    models/<name>/ [--smoke] [--device mps|cpu] [--seed N] [--limit N]
.venv/bin/flow_ml evaluate models/<name>/ [--checkpoint X] [--split test|val]

# Seed corpus regen (deterministic, no API calls, gated by per-task validators)
.venv/bin/python scripts/build_jcl_seeds.py        --target 1000
.venv/bin/python scripts/build_spool_seeds.py      --target 1500
.venv/bin/python scripts/build_flow_graph_seeds.py --target 500

# Custom JCL BPE tokenizer (one-shot; required before JCL training)
.venv/bin/python scripts/build_jcl_seeds.py --target 10000 \
    --out models/jcl-validator/datasets/samples/tokenizer_corpus.jsonl
.venv/bin/python -m flow_ml.tokenization.jcl_tokenizer train \
    --corpus models/jcl-validator/datasets/samples/tokenizer_corpus.jsonl \
    --out models/jcl-validator/datasets/tokenizer.json

# Batch wrappers for all three models in one pass
.venv/bin/python scripts/train_all.py    [--only jcl spool flow_graph] [--smoke] [--skip-prepare]
.venv/bin/python scripts/evaluate_all.py [--only ...] [--limit N]
```

## Architecture

This repo is the **Flow Studio Model Hub** — the registry/distribution source for the model artifacts flow-studio runs — and it trains the three task models. **Each model has a different architecture** (not all SFT), dispatched by `model.yml::architecture` at every CLI boundary.

**Runtime + export.** flow-studio's local runtime is the managed `llama-server` GGUF sidecar; the in-process Candle task-model path has been **removed**. As the Hub, flow-ml will catalog GGUF / MLX / safetensors artifacts — producing those from the trained checkpoints is **future work**, not wired up here yet (roadmap epic E2). See flow-studio `docs/architecture/model-hub.md` and `model-runtime.md`.

| Model folder | Architecture | Base |
|---|---|---|
| `models/jcl-validator/` | `classifier` (4-head multi-task) | ModernBERT-base |
| `models/spool-interpreter/` | `seq2seq` (full fine-tune) | flan-t5-base |
| `models/flow-graph-generator/` | `generative` (LoRA SFT, default) | Qwen3-1.7B |

**Single source of truth per model.** `models/<name>/model.yml` (loaded by `flow_ml.config.ModelDefinition`) carries every knob: `architecture`, `data:`, `training:`, `smoke:`, `packaging:`. The `data:`, `training:`, `smoke:` sections are `dict[str, Any]` so each trainer typechecks its own subset (e.g. `SpoolSeq2SeqConfig.from_dict`). The `packaging:` section (`PackagingSpec`) is retained for the future export path and is not consumed by the current CLI. Pydantic forbids extra fields at the top level — fields like `architecture` belong at the top.

**Cross-repo coupling.** The DSL spec/grammar that feeds the Flow Graph Generator's prompt originates in flow-studio; flow-ml never writes back. There is no longer a packaged-model runtime contract (the Candle task-model path was removed); the Hub relationship is artifact distribution via the catalog the flow-studio client reads.

### Trainer dispatch (`src/flow_ml/cli.py::cmd_train`)

```
md.task == "jcl_validation"      + md.architecture == "classifier" → training/jcl_classifier.py::train_jcl_classifier
md.task == "spool_interpretation" + md.architecture == "seq2seq"   → training/spool_seq2seq.py::train_spool_seq2seq
md.task == "flow_graph_generation"                                  → training/flow_graph_generator.py::train_flow_graph
```

Each trainer's `model_def.merged_smoke()` overlays `smoke:` on `training:` so `--smoke` reuses the same code path with cheaper config.

### Evaluator dispatch (`src/flow_ml/evaluation/runner.py`)

- `evaluate_jcl()` reimplements the BERT forward inline: `AutoModel.from_pretrained` for the encoder, `safetensors.torch.load_file` for the heads sidecar, builds a `JclValidationResult` JSON via the `error_message_templates` table in `node_contracts.json`, then runs the 6-layer Python validator. `message`+`suggestion` come from templates keyed by the predicted code — **zero hallucination on user-facing copy** is the design intent.
- `evaluate_spool()` uses `AutoModelForSeq2SeqLM` with greedy decode + the `interpret spool: ` task prefix. **Note**: T5's SentencePiece maps `{` and `}` to `<unk>`, which `skip_special_tokens=True` strips — the model only emits the JSON interior, so the evaluator wraps the decoded text with `{...}` before parsing.
- `evaluate_flow_graph()` is the Qwen3 generative path.

### Out-of-model validators (`src/flow_ml/validation/`)

Every task has an N-layer JSON gate that the seed builder uses to reject rows and the evaluator uses to compute metrics. They're authoritative — if a row passes the validator, it's safe to train on. Layer counts: JCL=6, FlowGraph=7, Spool=**8** (layers 7-8 added in v2 for `explanation` non-empty when `status != "completed"` and `relatedDocs: list[str]`).

### Data flow summary

```
datasets/samples/seed_samples.jsonl  ← scripts/build_*_seeds.py (deterministic)
        │
        │  flow_ml prepare
        ▼
output/prepared/{train,val,test}.jsonl
        │
        │  flow_ml train (dispatches by architecture)
        ▼
output/checkpoints/<run-name>/{model.safetensors, classifier_heads.safetensors?, tokenizer.json, config.json}
        │
        │  flow_ml evaluate
        ▼
output/eval/<run-name>.{json,md}
```

## Apple Silicon / MPS knobs (set everywhere by default)

- `bf16` autocast with fp32 master weights. Loading weights AT bf16 + autocast bf16 NaNs on MPS.
- `eval_steps: 9999` to disable mid-training eval (MPS unified memory doesn't release val-set tensors between eval and training).
- `grad_checkpointing: false` on the 1.7B trainers (MPS bf16 recomputation has destabilised in past runs).
- `dataloader_num_workers=0` everywhere (multi-worker + MPS deadlocks via fork pickling).
- `PYTORCH_ENABLE_MPS_FALLBACK=1` is set in `cli.py` for unsupported ops.
- JCL classifier full-attention on `max_input_tokens=2048, batch_size=16` OOMs at ~85 GB on M5 Max; the model.yml ships with `1024 × 8 × grad_accum=2` instead (same effective batch).

## Operational notes that aren't obvious

- **JCL classifier requires a tokenizer-training step before training**. The `train_jcl_classifier` falls back silently to the stock ModernBERT tokenizer if `models/jcl-validator/datasets/tokenizer.json` is missing — eval gates then miss because `<COL1>`/`<CONT>` never appear in the input. Always check that file exists before launching JCL training.
- **Seed corpus regen invalidates older checkpoints when schemas change.** Spool's v2 schema added `explanation` + `relatedDocs`; the validator's layers 7-8 reject seeds that lack them. After any `expected_*` field add, `build_*_seeds.py` must be re-run before `flow_ml prepare`.
- **Stale checkpoints get auto-picked.** `flow_ml.cli._latest_checkpoint()` selects the most recently modified dir under `output/checkpoints/`. When swapping architectures, `rm -rf` the old checkpoint dirs first or pass `--checkpoint` explicitly.
- **Tests live under `tests/`, not next to source.** Run from repo root.
