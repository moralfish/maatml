# MaatML Roadmap

MaatML is a **machine learning models framework from experimentation to
production**: data pipelines (validator-gated seed generation and ingestion) →
training → evaluation → export → deploy, with plugins that **optimize, distill,
and distribute** models to run anywhere. Purpose: task-specific domain language
models with a small footprint — from a multi-gigabyte base model to a small
subject-matter expert. Correctness is checked outside the model via validators
(the Maat weighing thesis).

**Design rule:** core owns architectures (`causal_sft`, `seq2seq`,
`multi_head_classifier`); examples own task semantics (validators, metrics,
tokenizers, generators, sanitizer rules, seed builders).

## Status

| Tranche | Theme | Status |
|---|---|---|
| Phase 0 | Rename to MaatML + examples-first restructure | Done |
| v0.2 | Experiment layer | Done |
| v0.3 | Methods + scale | Done |
| v0.4 | Product surface (export / deploy / flywheel) | Done |

## v0.2 — Experiment layer

Infrastructure everything else consumes.

- **Run registry** — `output/runs.jsonl`, per-run checkpoint dirs, `maatml runs`,
  replace mtime-based `_latest_checkpoint`
- **Resume** — `maatml train --resume [auto|PATH]`
- **Tracking passthrough** — `training.report_to` / `run_name` → HF
  `TrainingArguments` (W&B / TensorBoard if installed)
- **Per-head loss logging** — generic `loss_<head>` from `multi_head`
- **Adapter-only artifacts** — `training.lora.save_mode: merged|adapter|both`
- **Tokenize-once cache** + `group_by_length`
- **Format adapters** — `alpaca` / `sharegpt` as format plugins; multi-turn
  loss masking in `causal_sft`
- **Eval gates** — `evaluation.gates:` + `maatml evaluate --gate` (non-zero exit)
- **Hygiene** — wire or remove dead `dataset.group_by`

## v0.3 — Methods + scale

- QLoRA / quantized bases (`[cuda]` extra, bitsandbytes; CUDA-only)
- DPO / ORPO via TRL (`[pref]` extra); preference pairs mintable from validator
  outcomes
- CUDA profile maturation (wired `weights_dtype_policy`, flash-attn passthrough,
  `dataloader_workers` override)
- Multi-GPU via accelerate / torchrun (rank-0 run-registry writes)
- HPO / sweep hooks (`--set` overrides, `maatml sweep`)

## v0.4 — Product surface

- **Export** — `maatml export --format gguf|mlx|safetensors` + manifest from
  `PackagingSpec`; post-export parity gate on pinned benchmarks ✅
- **Checksums / signing / revision pinning** — `maatml verify`;
  `training.model_revision` ✅
- **Data flywheel** — `maatml datagen` / `ingest`; shared gated builder;
  optional OpenAI-compatible teacher behind the validator gate (`[teacher]`) ✅
- **Docs site** — mkdocs-material (`[docs]`), plugin-author guide ✅

## Non-goals (for now)

Competing with Axolotl / Unsloth on throughput, model coverage, or GUI.
MaatML wins on reproducible pipelines for domains where correctness is
decidable — not as a general fine-tuning speed race.
