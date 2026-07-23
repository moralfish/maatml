# MaatML Roadmap

MaatML is a **machine learning models framework from experimentation to
production**: data pipelines (validator-gated seed generation and ingestion) →
training → evaluation → export → deploy, with plugins that **optimize, distill,
and distribute** models to run anywhere. Purpose: task-specific domain language
models with a small footprint — from a multi-gigabyte base model to a small
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

## Status

| Tranche | Theme | Status |
|---|---|---|
| Phase 0 | Rename to MaatML + examples-first restructure | Done |
| v0.2 | Experiment layer | Done |
| v0.3 | Methods + scale | Done |
| v0.4 | Product surface (export / deploy / flywheel) | Done |
| v0.5 | Serving + multimodal | Done |
| v0.5.1 | Truth and safety | Done |
| v0.6 | Fixed lifecycle runner (`maatml run`) | Planned |
| v0.7 | Distill + reviewed flywheel | Planned |
| v0.8 | Slim artifact distribution | Planned |

## Non-goals

- Prefect-style scheduling / triggers / remote executors
- Arbitrary shell or Python pipeline steps
- Workspace-level multi-model DAGs (possible later)
- Distributed compute beyond existing accelerate / torchrun training
- Broad model zoo competing with Axolotl / LLaMA-Factory / Unsloth
- GUI

Completed release detail lives in [CHANGELOG.md](CHANGELOG.md). This file
carries outcomes, dependencies, and measurable exit criteria.

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
  outcomes (`mint_preference_pairs` library helper)
- CUDA profile maturation (wired `weights_dtype_policy`, flash-attn passthrough,
  `dataloader_workers` override)
- Multi-GPU via accelerate / torchrun (rank-0 run-registry writes)
- HPO / sweep hooks (`--set` overrides, `maatml sweep`)

## v0.4 — Product surface

- **Export** — `maatml export --format gguf|mlx|safetensors` + manifest from
  `PackagingSpec`; post-export parity gate on pinned benchmarks
- **Checksums / revision pinning** — `maatml verify` (sha256 integrity);
  `training.model_revision`. Cryptographic *signing* is deferred (see Later)
- **Data flywheel** — `maatml datagen` / `ingest`; shared gated builder;
  optional OpenAI-compatible teacher behind the validator gate (`[teacher]`)
- **Docs site** — mkdocs-material (`[docs]`), plugin-author guide

## v0.5 — Serving + multimodal

Shipped in 0.5.0 (see [CHANGELOG.md](CHANGELOG.md)).

- **Serving** — `maatml serve` (`/health`, `/info`, `/predict`;
  `/predict?validate=1` re-runs the registered validator when configured)
- **Vision** — `vision_multitask` + ONNX export (`[vision]`)
- **Vision-language** — `vlm_sft` (SmolVLM; vLLM-servable; `[vllm]` Linux-only)
- **Captioning** — `examples/vision-describer` (seq2seq over vision JSON)
- **Trusted Publishing** — PyPI OIDC publish workflow

## v0.5.1 — Truth and safety (Done)

The thesis is "ships with a contract"; make the public claims true before
building on them.

**Depends on:** v0.5.

- Support-ticket-triage gained a real validator plugin (`triage_plugin`:
  JSON → schema → `category → team` routing contract → summary quality) +
  `evaluation.gates` + a fixed benchmark + tests. It was the only reference
  example without a validator ✅
- jcl-validator and spool-interpreter promoted their README target tables into
  enforced `evaluation.gates` in `model.yml`, so **every** example now gates
  before the strict `--gate` change and the v0.6 runner depend on them ✅
- `evaluate --gate` with no gates configured now fails (exit non-zero) instead
  of passing vacuously — a behavior change (exit code 0 → non-zero), noted in
  CHANGELOG ✅
- Manifest `weights_dtype` verified from the actual exported tensors, with the
  declared value kept as `weights_dtype_declared` and a `weights_dtype_verified`
  flag ✅
- Serve: dropped the wildcard CORS default (opt-in `--cors` /
  `MAATML_SERVE_CORS`); added a request-body size cap (`--max-body-bytes`) ✅
- CI: Python 3.13 in the test matrix; a `[ml]` CPU `ml-smoke` job running
  `prepare → train --smoke → evaluate` on triage ✅. **Follow-up (maintainer):**
  publish-workflow rehearsal against TestPyPI needs a TestPyPI trusted-publisher
  configured first (OIDC), so it is left as a maintainer setup step.

**Exit criteria:** README's flagship validator claim is true; `--gate` cannot
pass with zero gates; flagship example has tests. **Met.**

## v0.6 — Fixed lifecycle runner: `maatml run` (Planned)

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

**Depends on:** v0.5.1 (so gates are meaningful).

- **`maatml run <model-dir>`** — executes stale steps in order:
  prepare → train → evaluate (gates enforced) → export → verify.
  Flags: `--smoke`, `--force`, `--from STEP`, `--until STEP`, `--dry-run`,
  `--device`, `--set` (overrides feed the fingerprint)
- **Fingerprints = idempotence, not speed.** Each step hashes effective config
  after smoke/overrides, declared input assets, prepared/eval data, checkpoint
  or manifest content, MaatML version / git SHA, plugin source hashes,
  device/profile, and exporter identity. Stored atomically in
  `output/pipeline.json`. Skip only when the fingerprint matches, the prior
  step completed, and expected outputs still verify. Train's fingerprint
  populates `RunRecord.spec_hash`
- **`maatml plan` becomes `run --dry-run`** — per-step fresh/stale status with
  the reason (which hash changed)
- Typed config where the runner fingerprints it (step sections), rather than a
  big-bang config rewrite

**Exit criteria:** `maatml run examples/support-ticket-triage/ --smoke` goes
seeds → verified export in one command in CI; re-run with no changes does no
work.

## v0.7 — Distill + reviewed flywheel (Planned)

**Depends on:** v0.5.1 (real validators) and preferably v0.6 (runner staleness).

- **`maatml distill`** — teacher responses for an unlabeled prompt pool; every
  row validator-gated before entering the seed corpus (reuse `TeacherClient`,
  `build_gated_corpus`, ingest field-mapping). Typed stage config (prompt
  source, teacher model/revision, retry/budget limits, output, provenance) —
  no new untyped `dict[str, Any]` surface. Accepted rows carry teacher
  model/revision, prompt hash, source, family; rejection reports retained;
  offline replay supported. Worked example on triage
- **Serve capture** — opt-in, sanitized, size/retention-capped, local.
  Captured model predictions are **not** automatically gold; human/teacher
  correction and explicit approval are required before ingestion. Approved
  rows fold into seeds and make `prepare` stale — the retrain loop is
  `serve --capture` → review → `ingest` → `run`
- **Preference minting CLI** — surface `mint_preference_pairs` as an optional
  source operation (not a default `run` step) feeding `dpo` / `orpo`

**Exit criteria:** no teacher or captured output enters training without
validation plus explicit acceptance; distillation is replayable offline from
provenance.

## v0.8 — Slim artifact distribution (Planned)

**Depends on:** v0.5.1 (honest manifests) and preferably v0.6 (verified export
as a runner step).

- **Canonical bundle digest** — stable digest from identity + formats + sorted
  file hashes in `manifest.json`
- **`maatml publish` / `pull`** with two backends: `local` (network-free
  reference, tested in CPU-free CI) and `hf` (Hub, `[hub]` extra); verify
  before upload and after download; immutable references; credentials
  delegated to native Hub auth
- **Auto `MODEL_CARD.md` at export** — recorded eval/gates + *explicitly
  declared* license metadata only; export must not network-lookup or infer a
  license

**Deferred until demand:** OCI backend, signing / SBOM, promotion channels /
ledger, `verify --require-signature`.

**Exit criteria:** `export → verify → publish → pull → verify` round-trips
through the local backend in CPU-free CI preserving the digest.

## Later / pre-1.0

- Precomputed top-k logit KD (teacher is training-only; eval/serve keep
  trusting validators); online KD CUDA-only and optional
- Signing (`verify --require-signature`, fail closed), OCI backend, promotion
  channels
- OpenAI-compatible `/v1/chat/completions`; jsonschema-constrained decoding;
  serve auth token
- Quantized export (GGUF `--outtype` levels, ONNX int8); generic core ONNX
  exporter (promote out of the vision plugin)
- 1.0: frozen config / plugin / manifest APIs, migration guarantees,
  reproducible benchmarks

## Hygiene backlog (non-tranche)

- Ship `py.typed`; `maatml doctor`; `runs --compare`
- Scaffold creation path for plugin-owned `vision_multitask` / `vlm_sft`;
  add missing scaffold tests for `seq2seq` / `multi_head_classifier`
- Windows CPU-free CI job; reconcile any stray `requirements*.txt` with
  pyproject extras; remove or fold orphaned root docs (`maatml.md`)
