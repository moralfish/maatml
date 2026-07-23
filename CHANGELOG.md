# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html)
for the Python package and per-model versions under `examples/`.

## [Unreleased]

Truth and safety II: make the "the same validator gates your data, evaluation,
and live inference" claim true, stop data-destroying commands, add a serve and
plugin security floor, and config honesty.

### Added

- **Serve enforcement:** `maatml serve --enforce` returns HTTP 422 when the
  configured validator rejects a prediction (gates live inference).
  `/predict?validate=1` stays a non-blocking annotation.
- **Ungated datagen escape hatch:** `maatml datagen --allow-ungated` runs
  without a validator and marks the run and a new `*.datagen_card.md` as
  UNGATED; the summary line reports GATED / UNGATED.
- **Untrusted-folder linting:** `maatml validate --no-plugins` checks schema and
  paths without importing model-folder plugin code.
- **Serve debug:** `maatml serve --debug` includes the exception and traceback
  in 500 responses (off by default).
- **Trust boundary** documented in README, SECURITY.md, and docs/plugins.md: a
  model folder is executable code and every command that reads `model.yml`
  (including `validate` and `plan`) runs its plugins.

### Changed

- **Behavior change:** `maatml datagen` now **fails** when no
  `evaluation.validator` is configured, instead of silently accepting every
  generated row. Pass `--allow-ungated` to keep the old accept-all behavior.
- **Behavior change:** `maatml scaffold` **refuses** to overwrite an existing
  `model.yml` or seed corpus; pass `--force` to regenerate.
- **Behavior change:** `--set` overrides are now validated (semver, `gt=0`,
  types); an invalid override exits non-zero instead of being applied silently.
- `maatml ingest` counts rows missing the gold field as `skipped_unvalidated`
  (instead of accepting them unvalidated) and errors when a `--map` source
  column matches zero input rows.
- `maatml evaluate` prints a notice when no validator is configured; a
  configured-but-unresolvable `evaluation.validator` now errors instead of
  silently degrading to JSON-parse-only scoring.
- Declaring `dataset.sanitize` with the alpaca / sharegpt / preference formats
  now errors (those paths cannot sanitize) rather than the dataset card falsely
  claiming a sanitizer ran; the card reports only tags actually applied.
- `maatml validate` warns on unrecognized `dataset:` / `evaluation:` keys.

### Fixed

- **Resume:** `maatml train --resume auto|<run_id>` now resolves to the newest
  `checkpoint-*` directory (previously it passed the run root, which current
  transformers rejects).
- **Run registry:** a torn or unparseable line in `runs.jsonl` is skipped with a
  warning and quarantined to `runs.jsonl.corrupt` instead of failing every
  command that reads the registry; records are written in a single append.
- **Seed safety:** `maatml datagen` writes seed files atomically and never
  truncates a non-empty seed file when nothing was accepted.

### Security

- **Serve:** 500 responses no longer leak the exception message or traceback to
  the client (server-side log only; opt in with `--debug`); a warning is printed
  when binding a non-loopback host.
- **Tokenized cache** loads with `torch.load(weights_only=True)`, closing a
  pickle code-execution sink under `output/cache/`.
- **GGUF export** resolves the convert script only from `MAATML_LLAMA_CONVERT`
  or `extensions.gguf.convert_script`; it no longer searches `PATH` or the cwd
  for a generic `convert.py`.
- **Vision predictors** confine request-supplied image paths to the model
  directory, rejecting absolute paths, `..` segments, and symlink escapes
  (closes a serve-time arbitrary-file-read).

## [0.5.1] - 2026-07-23

### Added

- **Triage contract:** `examples/support-ticket-triage` ships a real validator
  (`triage_plugin`) with a JSON → schema → **routing contract** (`category →
  team`) → summary-quality pipeline, plus enforced `evaluation.gates`, a fixed
  benchmark (`test_prompt_set.jsonl`), and CPU-free tests. It was the only
  reference example without a validator.
- **Gates everywhere:** `jcl-validator` and `spool-interpreter` promote their
  README target tables into enforced `evaluation.gates`, so every bundled
  example now gates.
- **Serve hardening:** opt-in CORS via `--cors` / `MAATML_SERVE_CORS` and a
  request-body size cap via `--max-body-bytes` (default 1 MiB → `413`).
- **Honest manifests:** export `manifest.json` records `weights_dtype` read from
  the exported safetensors tensors (`weights_dtype_verified: true`) alongside the
  declared `weights_dtype_declared` hint; mixed-precision exports list every
  observed dtype.
- **CI:** Python 3.13 in the test matrix; a CPU `ml-smoke` job that runs
  `prepare → train --smoke → evaluate` on triage through the real `[ml]` stack.

### Changed

- **Behavior change:** `maatml evaluate --gate` now **fails (exit non-zero)**
  when a model declares no `evaluation.gates`, instead of passing vacuously.
  Scripts relying on the old exit-0 must add a `gates:` block or drop `--gate`.
- **Security:** `maatml serve` no longer sends a wildcard
  `Access-Control-Allow-Origin: *` by default, cross-origin access is now
  opt-in.

## [0.5.0] - 2026-07-22

### Added

- **Serving:** `maatml serve` runs a dependency-light HTTP inference API
  (`/health`, `/info`, `/predict`); `/predict?validate=1` re-runs the registered
  validator inline. Light enough for edge / Jetson.
- **Vision:** `vision_multitask` architecture (MobileNetV3: scene classification,
  shape detection, pose) with ONNX export. Extra: `pip install maatml[vision]`.
- **Vision-language:** `vlm_sft` architecture; `examples/vision-vlm` fine-tunes
  SmolVLM-256M and exports HF-format checkpoints servable by vLLM. Extra:
  `pip install maatml[vllm]` (Linux-only).
- **Captioning:** `examples/vision-describer` (flan-t5 seq2seq) turns the
  multitask vision output into a short description.
- `maatml export --format onnx`; VLM processor assets bundled for vLLM serving.

### Fixed

- CPU-free CI: move SFT config models to `training/sft_config.py` so
  `tests/test_quantization.py` collects without torch; mypy assignment /
  `TrainingArguments` stub mismatches.
- `maatml serve` builds a serve context without torch (falls back to a plain
  device string), so its tests pass on CPU-free CI.

### Changed

- Docs overhaul: README and site now lead with the validator-gated
  *data → eval → serving* differentiator across text / vision / VLM; new
  `docs/lifecycle.md` and `docs/serving.md`; CONTRIBUTING gains a
  "Where to start" good-first-issues section.
- Docs and README lead with `pip install maatml` (PyPI); canonical site is
  [maatml.pages.dev](https://maatml.pages.dev).
- PyPI Trusted Publishing workflow (`.github/workflows/publish.yml`) + GitHub
  Environment `pypi`.

## [0.4.0] - 2026-07-21

### Added

- **Product surface (v0.4):**
  - `maatml export`: safetensors bundle (+ optional `gguf` / `mlx` when
    tooling is installed) with `manifest.json` from `PackagingSpec`
  - `maatml verify`: recompute sha256 of manifest-listed files
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
- Example model: `examples/support-ticket-triage` (`causal_sft`, ticket → triage JSON)
- Reference models: JCL Validator (ModernBERT multi-head classifier) and Spool
  Interpreter (flan-t5 seq2seq), versioned at `0.1.0`
- Group-aware (`family`) dataset splits and family-stamped seed corpora
- CI: lint (ruff/mypy), Python 3.10-3.12 test matrix, wheel standalone install job
