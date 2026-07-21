# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html)
for the Python package and per-model versions under `examples/`.

## [0.4.0] - 2026-07-21

### Added

- **Product surface (v0.4):**
  - `maatml export` — safetensors bundle (+ optional `gguf` / `mlx` when
    tooling is installed) with `manifest.json` from `PackagingSpec`
  - `maatml verify` — recompute sha256 of manifest-listed files
  - Post-export `--parity` gate against `dataset.benchmark_samples` +
    `evaluation.gates`
  - `training.model_revision` passed to `from_pretrained` / tokenizer load;
    recorded in `run_metadata.extra`
  - Data flywheel: `maatml.data.gated.build_gated_corpus`,
    `maatml datagen` (registered generators or `--teacher`),
    `maatml ingest` (map / sanitize / validate / dedupe)
  - Optional teacher client (`MAATML_TEACHER_BASE_URL` /
    `MAATML_TEACHER_API_KEY`); extra `pip install maatml[teacher]`
  - Registries: `EXPORTERS` / `GENERATORS` (`register_exporter`,
    `register_generator`); jcl / spool example generators
  - Lightweight docs site: `mkdocs.yml` + `docs/`; extra
    `pip install maatml[docs]`

## [0.3.0] - 2026-07-21

### Added

- **Methods + scale (v0.3):**
  - QLoRA / quantized bases for `causal_sft` via `training.quantization:`
    (bitsandbytes; CUDA-only). Extra: `pip install maatml[cuda]` (with `[ml]`).
  - `DeviceProfile.allow_quantized_load` (True only for `cuda`); hard error on
    mps/cpu when quantization is requested
  - Wired `weights_dtype_policy`: `fp32_master` (mps/cpu) vs `native` bf16/fp16
    master weights on CUDA when `training.precision` matches
  - `training.attn_implementation` passthrough (`flash_attention_2` / `sdpa` /
    `eager`) and `training.dataloader_workers` override across trainers
  - Multi-GPU via accelerate / torchrun: distributed detection, HF Trainer owns
    placement, rank-0-only run-registry / `run_metadata` writes
  - Architectures `dpo` / `orpo` (TRL); format `preference_jsonl`; helper
    `mint_preference_pairs`. Extra: `pip install maatml[pref]` (with `[ml]`)
  - `maatml train --set KEY=VALUE` and offline `maatml sweep --param KEY=a,b`
    (cartesian grid, no Optuna)

## [0.2.0] - 2026-07-21

### Added

- **Experiment layer (v0.2):**
  - Run registry (`output/runs.jsonl`, `maatml runs`) with per-run checkpoint
    dirs under `output/checkpoints/<run_id>/`
  - `maatml train --resume [auto|PATH]` wired into all trainers
  - `training.report_to` / `run_name` passthrough to HuggingFace TrainingArguments
  - Per-head `loss_<name>` logging for `multi_head_classifier`
  - `training.lora.save_mode: merged|adapter|both` (adapter-aware CausalSFTPredictor)
  - Tokenize-once dataset cache (`output/cache/`) + `training.group_by_length` (causal_sft)
  - Format adapters `alpaca` / `sharegpt`; multi-turn loss masking in causal SFT
  - `evaluation.gates` + `maatml evaluate --gate` (non-zero exit on failure)
  - `dataset.group_by` wired into group-aware splits

### Changed

- **Renamed the project from flow-ml to MaatML.** Package / CLI / entry-point
  group are now `maatml` / `maatml` / `maatml.plugins`. Resolves the name
  collision with [MLflow](https://mlflow.org/). GitHub repo:
  [moralfish/maatml](https://github.com/moralfish/maatml).
- **Examples-first restructure:** `jcl-validator` and `spool-interpreter` live
  under `examples/` with folder-local plugins (`jcl_plugin` / `spool_plugin`).
  Core owns architectures (`causal_sft`, `seq2seq`, `multi_head_classifier`);
  examples own validators, metrics, tokenizers, generators, and sanitizer rules.
- Sanitizers and text transforms are registries (`register_sanitizer`,
  `register_transform`); `load_model_plugins` loads package directories.

## [0.1.0] - 2026-07-21

### Added

- Public contribution surface: Apache-2.0 LICENSE, CONTRIBUTING, CODE_OF_CONDUCT,
  SECURITY, CHANGELOG, CODEOWNERS, issue/PR templates, Dependabot, pre-commit
- Plugin registry (`maatml.registry`) with trainers, validators, metrics,
  predictors, formats, and scaffold hooks; entry-point group `maatml.plugins`
- Registry-driven CLI: `prepare`, `train`, `evaluate`, `scaffold`, `validate`,
  `plugins`, `plan`
- Standalone model folders: all paths resolve relative to the model dir; no
  repo-relative fallbacks; wheel package-data for sanitization rules and fixtures
- Device profiles (`mps` / `cuda` / `cpu`) and training guards (NaN abort,
  tokenizer↔embedding contract, `run_metadata.json` provenance)
- Shared validation base + generic evaluation harness
- Reference contrib plugins: `maatml.contrib.jcl`, `maatml.contrib.spool`
- Example model: `examples/support-ticket-triage` (`causal_sft` — ticket → triage JSON)
- Reference models: JCL Validator (ModernBERT multi-head classifier) and Spool
  Interpreter (flan-t5 seq2seq), versioned at `0.1.0`
- Group-aware (`family`) dataset splits and family-stamped seed corpora
- CI: lint (ruff/mypy), Python 3.10–3.12 test matrix, wheel standalone install job
