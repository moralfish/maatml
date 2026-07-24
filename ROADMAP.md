# MaatML Roadmap

MaatML is a **machine learning models framework from experimentation to
production**: data pipelines (validator-checked seed generation and ingestion) →
training → evaluation → export → deploy, with plugins that **optimize, distill,
and distribute** models to run anywhere. Purpose: task-specific domain language
models with a small footprint, from a multi-gigabyte base model to a small
subject-matter expert. Correctness is checked outside the model via validators
(the Maat weighing thesis).

**Design rule:** core owns architectures (`causal_sft`, `seq2seq`,
`multi_head_classifier`, `dpo` / `orpo`); examples own task semantics
(validators, metrics, tokenizers, generators, sanitizer rules, seed builders).
Plugin-owned architectures (`vision_multitask`, `vlm_sft`) register from their
model folders.

**Positioning:** MaatML runs its own **fixed** validator-gated lifecycle. It is
not a general-purpose workflow scheduler or arbitrary DAG engine (no
triggers/cron, no shell/python steps, no remote executors). Source-mutating
operations (`datagen`, `ingest`, planned `distill`) stay explicit and outside
the default runner.

**Claims rule:** the guarantee register ("gates", "cannot", "never") ships only
together with the mechanism that enforces it and a test that proves it. Exit
criteria in this file are testable assertions; wording that cannot be tested
does not belong in them.

## Status

Version numbers are assigned only when a tranche ships. Planned tranches below
are ordered but unversioned.

| Tranche | Theme | Status |
|---|---|---|
| Phase 0 | Rename to MaatML + examples-first restructure | Done |
| v0.2 | Experiment layer | Done |
| v0.3 | Methods + scale | Done |
| v0.4 | Product surface (export / deploy / flywheel) | Done |
| v0.5 | Serving + multimodal | Done |
| v0.5.1 | Truth and safety | Done |
| v0.6 | Truth and safety II: gates tell the truth | Done |
| v0.7 | Silent-failure hardening + test floor | Done |

## Non-goals

- Arbitrary shell or Python pipeline steps
- Broad model zoo competing with Axolotl / LLaMA-Factory / Unsloth

Completed release detail lives in [CHANGELOG.md](CHANGELOG.md). This file
carries outcomes, dependencies, and measurable exit criteria.

## v0.2: Experiment layer

Infrastructure everything else consumes.

- **Run registry**: `output/runs.jsonl`, per-run checkpoint dirs, `maatml runs`,
  replace mtime-based `_latest_checkpoint`
- **Tracking passthrough**: `training.report_to` / `run_name` → HF
  `TrainingArguments` (W&B / TensorBoard if installed)
- **Per-head loss logging**: generic `loss_<head>` from `multi_head`
- **Adapter-only artifacts**: `training.lora.save_mode: merged|adapter|both`
- **Tokenize-once cache** + `group_by_length`
- **Format adapters**: `alpaca` / `sharegpt` as format plugins; multi-turn
  loss masking in `causal_sft`
- **Eval gates**: `evaluation.gates:` + `maatml evaluate --gate` (non-zero exit)
- **Hygiene**: wire or remove dead `dataset.group_by`

## v0.3: Methods + scale

- QLoRA / quantized bases (`[cuda]` extra, bitsandbytes; CUDA-only)
- DPO / ORPO via TRL (`[pref]` extra); preference pairs mintable from validator
  outcomes (`mint_preference_pairs` library helper)
- CUDA profile maturation (wired `weights_dtype_policy`, flash-attn passthrough,
  `dataloader_workers` override)
- Multi-GPU via accelerate / torchrun (rank-0 run-registry writes)
- HPO / sweep hooks (`--set` overrides, `maatml sweep`)

## v0.4: Product surface

- **Export**: `maatml export --format gguf|mlx|safetensors` + manifest from
  `PackagingSpec`; post-export parity gate on pinned benchmarks
- **Checksums / revision pinning**: `maatml verify` (sha256 integrity);
  `training.model_revision`. Cryptographic *signing* is deferred (see Later)
- **Data flywheel**: `maatml datagen` / `ingest`; shared gated builder;
  optional OpenAI-compatible teacher behind the validator gate (`[teacher]`)
- **Docs site**: mkdocs-material (`[docs]`), plugin-author guide

## v0.5: Serving + multimodal

Shipped in 0.5.0 (see [CHANGELOG.md](CHANGELOG.md)).

- **Serving**: `maatml serve` (`/health`, `/info`, `/predict`;
  `/predict?validate=1` re-runs the registered validator when configured)
- **Vision**: `vision_multitask` + ONNX export (`[vision]`)
- **Vision-language**: `vlm_sft` (SmolVLM; vLLM-servable; `[vllm]` Linux-only)
- **Captioning**: `examples/vision-describer` (seq2seq over vision JSON)
- **Trusted Publishing**: PyPI OIDC publish workflow

## v0.5.1: Truth and safety (Done)

The thesis is "ships with a contract"; make the public claims true before
building on them.

**Depends on:** v0.5.

- Support-ticket-triage gained a real validator plugin (`triage_plugin`:
  JSON → schema → `category → team` routing contract → summary quality) +
  `evaluation.gates` + a fixed benchmark + tests. It was the only reference
  example without a validator 
- jcl-validator and spool-interpreter promoted their README target tables into
  enforced `evaluation.gates` in `model.yml`, so **every** example now gates
  before the strict `--gate` change and the lifecycle runner depend on them 
- `evaluate --gate` with no gates configured now fails (exit non-zero) instead
  of passing vacuously, a behavior change (exit code 0 → non-zero), noted in
  CHANGELOG 
- Manifest `weights_dtype` verified from the actual exported tensors, with the
  declared value kept as `weights_dtype_declared` and a `weights_dtype_verified`
  flag 
- Serve: dropped the wildcard CORS default (opt-in `--cors` /
  `MAATML_SERVE_CORS`); added a request-body size cap (`--max-body-bytes`) 
- CI: Python 3.13 in the test matrix; a `[ml]` CPU `ml-smoke` job running
  `prepare → train --smoke → evaluate` on triage . **Follow-up (maintainer):**
  publish-workflow rehearsal against TestPyPI needs a TestPyPI trusted-publisher
  configured first (OIDC), so it is left as a maintainer setup step.

## v0.6: Truth and safety II: gates tell the truth (Done)

**Tracking:** #13

Every place the validator contract can quietly
degrade, every command that can destroy user data, a security floor for serve
and plugin execution, and config honesty. Small diffs, high stakes. This
tranche is what makes the README's three-stage gating sentence true as written.

**Depends on:** v0.5.1.

**Gating is real at all three stages**

- `datagen` fails closed when no `evaluation.validator` is configured
  (reuse the `GateConfigError` pattern); explicit `--allow-ungated` escape
  hatch marks the summary line and dataset card `UNGATED` (test)
- `serve --enforce`: validation failures return HTTP 422; `?validate=1`
  remains as annotation mode. README / docs index / lifecycle keep the
  "gates serving" wording only once this ships (test)
- Sanitize tags apply on **all** format paths (`alpaca` / `sharegpt` /
  preference) or the command errors on an unsupported combination; the
  dataset card records what actually ran, not what was declared (test)
- `ingest`: rows missing the gold field are rejected or separately counted
  (`skipped_unvalidated`); `--map` errors when a source column matches zero
  input rows (test)
- Serve validator call shape resolved once at startup (no TypeError fallback
  that silently weakens validation to a raw-output call)
- A configured `evaluation.validator` that does not resolve to a registered
  validator is an error at evaluate / datagen time (`GateConfigError`
  pattern), not a silent downgrade to JSON-parse-only scoring

**No command destroys user data**

- `scaffold` refuses to overwrite an existing `model.yml` or seed corpus
  without `--force` (test)
- `datagen` writes seed files atomically (temp + `os.replace`) and refuses to
  truncate a non-empty seed file on a zero-accepted run (test)
- `runs.jsonl`: single-write appends; readers skip-and-warn unparseable lines
  (quarantine to `runs.jsonl.corrupt`) instead of failing every consumer (test)

**Broken as shipped → working**

- `--resume auto|RUN_ID` resolves the newest `checkpoint-*`
  (`get_last_checkpoint`) instead of the run root

**Security floor**

- serve 500s return a generic message; tracebacks go to the server log only
  (client tracebacks solely under `--debug`); warn on non-loopback bind
- Trust boundary documented in README + SECURITY.md + docs/plugins.md:
  running any command against a model folder executes its `plugins:`, including `validate` and `plan`; add `validate --no-plugins` for schema and
  path checks without code execution
- Tokenized cache loads with `weights_only=True` (or moves to JSON), no
  pickle sink
- GGUF converter resolved from explicit config/env only (drop the generic
  `convert.py` PATH / cwd lookup)
- Vision predictors reject absolute and `..` image paths and confine reads to
  the model directory

**Config honesty**

- `--set` overrides re-validate (`validate_assignment` or re-parse after
  apply), the CLI can no longer bypass schema constraints it advertises (test)
- Known-keys warning pass for `dataset:` / `evaluation:` sections; `evaluate`
  prints "no validator configured, scoring JSON parse only" when that is
  what is happening
- docs/serving.md describes `verify` as corruption detection (listed files
  match their recorded sha256), not tamper evidence, guarantee wording
  returns when injected-file detection and signing ship

**Exit criteria:** the README three-stage gating sentence is true as worded,
each stage backed by a test (datagen fail-closed; serve `--enforce` → 422;
eval `--gate` already covered); `scaffold` and `datagen` cannot remove user
content without `--force` (tests); a torn last line in `runs.jsonl` is
quarantined and `maatml runs` still lists prior runs (test); `--resume auto`
resumes to completion from an interrupted smoke run that saved a mid-run
checkpoint (CI test; the smoke overlay sets `save_steps` low enough to
checkpoint within the smoke budget); `--set` of a schema-invalid value exits
non-zero (test); serve 500 response bodies contain no traceback (test).

## v0.7: Silent-failure hardening + test floor (Done)

**Tracking:** #14

The remaining "looks green, did nothing" paths, and tests for the parts of the
codebase that had none, the trainers (4 of 6 architectures) and the CLI.

**Depends on:** Truth and safety II.

**Honest numbers**

- Per-class eval numbers are `pass_rate` / `passed` / `n`. The old shape
  reported a real precision alongside a literal `recall: 1.0` and `f1: 0.0`
  for every category, so two of four numbers were invented
- seq2seq brace repair is opt-in (`evaluation.repair_braces`), counted, and
  recorded in `Report.extras`; both seq2seq examples set it
- `evaluate` defaults its token budget to `packaging.max_input_tokens` (parity
  with serve and `export --parity`) and counts truncated inputs
- `evaluation.metrics` as a list runs every entry and merges results; two
  plugins claiming one metric key is an error (test)

**Coercion, now loud failure**

- Fractional epochs honoured in seq2seq / multi_head; `precision` validated
  where it is parsed (test)
- Unknown gold labels are scanned before training and fail with the offending
  values counted; boolean gold maps through the declared label order (test)
- seq2seq rows with a missing or empty target are dropped and counted, never
  serialised to the literal `"{}"` (test)
- Preference rows serialise structured chosen/rejected as JSON (test); an
  identical pair warns; DPO / ORPO save through `_save_sft_artifacts`, so
  `lora.save_mode` means the same thing everywhere
- alpaca / sharegpt drop and count rows with no user or assistant content; an
  all-degenerate corpus fails (test)
- `multi_head` legacy JCL fallback fires only when the legacy keys are present;
  absent or malformed `training.heads` is an error (test)
- Teacher failures are counted, the first error is reported on the datagen
  card, and five consecutive failures abort instead of burning the attempts
  cap (test)
- A group key covering ~the whole corpus splits per row with a loud warning;
  teacher and ingest rows carry a per-row family; an empty split is reported
  (test)
- `prepare` refuses a benchmark whose rows share a group key with the training
  splits (test). The vision examples were copying the first N seed rows as
  their benchmark, so the pinned score measured memorisation; both builders now
  generate held-out rows in a `bench_*` family namespace
- Trainer bodies mark the run `aborted` for fallible work that used to sit
  outside the finish handler; `--resume auto` skips `running` records with no
  checkpoint (test)
- The SFT tokenize cache key includes `val.jsonl` content
- `sweep` records a failed trial, continues, and exits non-zero at the end;
  ranking compares only trials reporting the chosen metric, with the direction
  taken from the metric name (test)
- Sanitizer warns once per rule when length-preserving truncation cuts a
  redaction short, and rejects a fixed replacement that cannot fit its
  pattern's shortest match at load (test)
- `datagen` append dedups by content / `sample_id` and clears a stale reject
  report (test)
- Plugin lifecycle: `load_model_plugins` is the single idempotent owner (test);
  per-source failures are recorded, listed by `maatml plugins`, and named in
  `Unknown … plugin` errors (test); `discover_plugins` no longer wipes the
  registries when the trainer registry looks empty (test); the `Validator`
  protocol matches the real call sites
- Known CLI user errors print one actionable line; `maatml --debug` restores
  the traceback (test)

**Test floor**

- `typer.testing.CliRunner` suite: `evaluate --gate` exit codes, `verify` on a
  tampered manifest, `scaffold` refusal, per-command argument parsing, and a
  torn `runs.jsonl` (torch-free)
- Torch-gated unit tests for all four previously untested trainers (SFT label
  masking, seq2seq label padding, multi_head per-head targets, preference row
  loading), plus torch-free config-parsing tests
- Shared `tests/conftest.py` snapshots and restores registries through a public
  API (`snapshot_registries` / `restore_registries` / `reset_registries` /
  `Registry.unregister`); tests no longer touch `_entries`
- The ml job installs `[vision]` and runs `pytest tests/ examples/`, so the
  torch-gated tests and the vision end-to-end test execute; the matrix runs
  with `-rs`
- `macos-latest` torch-free job. The optional weekly scheduled MPS `[ml]` smoke
  was not added: it is an optimization rather than a claims item, and local
  development exercises MPS continuously
- `gh-action-pypi-publish` pinned to a commit SHA (Dependabot keeps it bumped)

**Exit criteria (met):** each of the four previously-untested trainer modules
has a torch-gated unit test exercising config parsing or label masking; the
CliRunner suite covers `evaluate --gate` exit codes, `verify` on a tampered
manifest, and `scaffold` refusal; the ml job runs pytest with the vision
end-to-end test green; preparing a datagen-produced corpus yields non-empty
val/test splits (regression test for the degenerate-group fix); CI includes a
macOS job.

## Fixed lifecycle runner: `maatml run` (Planned)

**Tracking:** #15

```
seed corpus
    │
    ▼
prepare ──► train ──► evaluate --gate ──► export ──► verify
                           │
                           └── gates fail → stop (non-zero)
```

Explicit source ops (`datagen` / `ingest` / later `distill`) append to the seed
corpus and stay **outside** the default runner; changing seeds makes `prepare`
stale.

**Depends on:** Truth and safety II + Silent-failure hardening (every step the
runner chains must fail loud first; `runs.jsonl` must tolerate a torn line).

- **`maatml run <model-dir>`**: executes stale steps in order:
  prepare → train → evaluate (gates enforced) → export → verify.
  Flags: `--smoke`, `--force`, `--from STEP`, `--until STEP`, `--dry-run`,
  `--device`, `--set` (overrides feed the fingerprint)
- **Smoke-tier gates**: the `smoke:` overlay may override `evaluation.gates`;
  a smoke-gated run is marked as such in the run record and manifest, so a
  green smoke line stays distinguishable from a real gate pass
- **Fingerprints = idempotence, not speed.** Each step hashes effective config
  after smoke/overrides, declared input assets, prepared/eval data, checkpoint
  or manifest content, MaatML version / git SHA, plugin source hashes,
  device/profile, and exporter identity. Stored atomically in
  `output/pipeline.json`. Skip only when the fingerprint matches, the prior
  step completed, and expected outputs still verify. Train's fingerprint
  populates `RunRecord.spec_hash`. `--set` values are validated (Truth and
  safety II) before they enter a fingerprint
- **`maatml plan` becomes `run --dry-run`**: per-step fresh/stale status with
  the reason (which hash changed)
- Typed config where the runner fingerprints it (step sections), rather than a
  big-bang config rewrite. The typed surface includes `evaluation.validator`
  and gate keys

**Exit criteria:** `maatml run examples/support-ticket-triage/ --smoke` goes
seeds → verified export in one command in CI using smoke-tier gates, with the
run record marked smoke-gated; re-run with no changes does no work; a run with
a misspelled `evaluation.validator` exits non-zero; `run --set` with a
schema-invalid value exits non-zero and leaves `output/pipeline.json`
unchanged.

## Distill + reviewed flywheel + serve contract (Planned)

**Tracking:** #16

**Depends on:** Truth and safety II + Silent-failure hardening (hardened
`TeacherClient` / `build_gated_corpus` / ingest), and preferably the lifecycle
runner (staleness).

- **`maatml distill`**: teacher responses for an unlabeled prompt pool; every
  row validator-gated before entering the seed corpus (reuse `TeacherClient`,
  `build_gated_corpus`, ingest field-mapping). Typed stage config (prompt
  source, teacher model/revision, retry/budget limits, output, provenance), no new untyped `dict[str, Any]` surface. Accepted rows carry teacher
  model/revision, prompt hash, source, family; rejection reports retained;
  offline replay supported. Worked example on triage
- **Serve capture**: opt-in, sanitized, size/retention-capped, local.
  Captured model predictions are **not** automatically gold; human/teacher
  correction and explicit approval are required before ingestion. Approved
  rows fold into seeds and make `prepare` stale, the retrain loop is
  `serve --capture` → review → `ingest` → `run`. Capture requires the serve
  auth token (below)
- **Serve contract** (promoted from Later: `--capture` and non-loopback binds
  need auth, and `--enforce` from Truth and safety II pairs with enforcement
  during generation):
  - serve auth token (required for `--capture`, recommended for non-loopback
    binds)
  - jsonschema-constrained decoding at serve time via a constrained-decoding
    logits processor (outlines or lm-format-enforcer, behind a serve extra);
    generative architectures only; structure enforced *during* generation
    instead of only rejected after
  - bounded retry-with-feedback: on validation failure, feed the error back
    to the model and re-ask once; retries counted and reported, never silent
- **Preference minting CLI**: surface `mint_preference_pairs` as an optional
  source operation (not a default `run` step) feeding `dpo` / `orpo`

**Exit criteria:** a validator-rejected teacher row is absent from the
produced seed corpus (test); `ingest` rejects a captured row lacking explicit
approval (test); `serve --capture` without the auth token refuses to start
(test); distill replay with network disabled reproduces the accepted corpus
(CI); a serve request failing validation under `--enforce` returns 422 with
the retry count in the response metadata (test).

## Slim artifact distribution (Planned)

**Tracking:** #17

**Depends on:** v0.5.1 (verified manifest dtype), and preferably the lifecycle
runner (verified export as a runner step). `verify`'s own blind spots close
inside this tranche, before publish/pull builds on it.

- **`verify` closes its blind spots first**: distribution multiplies bundles
  in the wild, so the verb it rests on must be exact:
  - files present in the export dir but absent from `manifest.json` are
    verification errors (injected-file detection)
  - `export --parity` writes its artifacts outside the bundle or re-emits the
    manifest afterwards, a documented flag must not invalidate the bundle it
    documents
  - with these shipped, docs may describe `verify` as integrity checking of
    the full bundle contents; tamper evidence still waits for signing (Later)
- **Canonical bundle digest**: stable digest from identity + formats + sorted
  file hashes in `manifest.json`
- **`maatml publish` / `pull`** with two backends: `local` (network-free
  reference, tested in CPU-free CI) and `hf` (Hub, `[hub]` extra); verify
  before upload and after download; immutable references; credentials
  delegated to native Hub auth
- **Auto `MODEL_CARD.md` at export**: recorded eval/gates + *explicitly
  declared* license metadata only; export must not network-lookup or infer a
  license

**Deferred until demand:** OCI backend, signing / SBOM, promotion channels /
ledger, `verify --require-signature`.

**Exit criteria:** `export → verify → publish → pull → verify` round-trips
through the local backend in CPU-free CI preserving the digest; a file
injected into a bundle after export fails `verify` (test).

## Later

- Precomputed top-k logit KD (teacher is training-only; eval/serve keep
  trusting validators); online KD CUDA-only and optional
- QAT (torchao) for the edge path: post-hoc GGUF/MLX/ONNX quantization loses
  accuracy that quantization-aware training recovers
- Signing (`verify --require-signature`, fail closed), OCI backend, promotion
  channels
- OpenAI-compatible `/v1/chat/completions`
- lm-eval-harness bridge: declare a public-benchmark smoke suite in
  `model.yml` next to task gates, for regression numbers alongside the
  validators
- Validator adapters: run Guardrails Hub or plain Pydantic validators inside
  the MaatML validator contract
- SetFit-style few-shot classifier path: ~8 labels/class at low compute (per
  published SetFit results), as a cheap entry point before a full
  `multi_head_classifier` fine-tune
- `maatml render-config`: effective config with all defaults and overrides
  materialized (pairs with the known-keys warnings from Truth and safety II)
- Quantized export (GGUF `--outtype` levels, ONNX int8); generic core ONNX
  exporter (promote out of the vision plugin)
- 1.0: frozen config / plugin / manifest APIs, migration guarantees,
  reproducible benchmarks
