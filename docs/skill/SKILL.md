---
name: maatml
description: Build, gate, export and serve a task-specific model with maatml. Use when working in a model folder (model.yml, datasets/, output/, a *_plugin directory) or on maatml itself - building a corpus, distilling from a teacher, deriving gates, running the lifecycle, deciding whether a run ships, exporting GGUF, or serving a model behind its validator.
---

# maatml

Fine-tunes small task-specific models across text, vision and vision-language,
and takes them to production through one declarative `model.yml`:
**prepare -> train -> evaluate -> export -> serve**.

The organizing idea: **correctness is judged outside the model, by a validator**
that checks output against a contract. The same validator gates the synthetic
data, the evaluation, and optionally live inference. A maatml model ships with a
contract, not just weights.

```
                    validator  (per task, registered by a plugin)
          ┌─────────────────┼─────────────────┐
     gates data        gates eval        guards serving
      datagen           evaluate          serve --enforce
      distill            --gate           ?validate=1
      ingest
```

## Commands

```bash
maatml scaffold <dir> --architecture causal_sft --name my-task
maatml validate <model-dir>          # config, declared paths, plugin registration
maatml audit    [model-dir]          # read-only pre-flight; exits 1 on anything broken

maatml prepare  <model-dir>          # train/val/test splits
maatml train    <model-dir>          # LoRA / QLoRA / full / DPO / ORPO / vision / VLM
maatml evaluate <model-dir> --gate   # validator + metrics + gates, non-zero on failure
maatml export   <model-dir> --format gguf
maatml verify   <export-dir>         # sha256 against manifest.json
maatml serve    <model-dir>

maatml run  <model-dir>              # the whole walk, stopping at the first failure
maatml run  <model-dir> --smoke      # same walk, smoke tier
maatml plan <model-dir>              # what is stale and why (= run --dry-run)
maatml runs <model-dir> [--compare]  # what has been recorded
```

`run` skips steps whose fingerprint still matches — effective config, declared
inputs, upstream fingerprint, maatml version and git SHA, plugin sources, device
profile, exporter. That is **idempotence, not speed**: a step is skipped only if
it completed last time and its outputs are still there. `output/pipeline.json`
holds it; `plan` prints which component changed.

The source operations — `datagen`, `distill`, `ingest`, `mint`, reviewed
`serve --capture` — stay **outside** the runner. They change the seed corpus,
which is exactly what makes `prepare` stale next run.

Depth, in this skill's own folder: `references/model-yml.md` for the config
surface, `references/plugins.md` for registering a validator, predictor,
metrics or generator, and `references/flywheel.md` for growing a corpus.

## Starting a new model folder

maatml is domain-agnostic: the task changes, the shape does not. In order, and
resisting the urge to train early:

1. **Name the contract before the model.** What makes an output correct, checked
   by a program rather than by reading it? If nothing can check it, the rest of
   the loop has nothing to stand on — that is the question to answer first, not
   which base model to use.
2. `maatml scaffold <dir> --architecture causal_sft --name my-task` writes a
   folder whose sections already agree with each other.
3. **Write the validator first**, before any corpus exists, and point
   `evaluation.validator` at it. It gates the data you are about to make, so it
   has to exist before the data. See `references/plugins.md`.
4. **Extract the contract's ground truth from the running system** rather than
   transcribing it: dump the schema the application actually sends, walk the
   live API, read the vocabulary out of the real files.
5. **Seed a small corpus by hand**, then grow it with `datagen` / `distill` /
   `ingest` — every added row gated by that same validator. See
   `references/flywheel.md`.
6. `maatml run <dir> --smoke` to prove the lifecycle walks end to end on real
   hardware, gated at the smoke tier. Expect it to fail its production gates;
   that run exists to prove plumbing, not quality.
7. Only then a full run, and derive the gates **from its report** rather than
   guessing them up front.

Steps 1 and 3 are the ones people skip, and skipping them is what turns a
fine-tune into an unfalsifiable claim.

## The rule that prevents most damage

**The CLI owns the lifecycle. Read the module before working around it.** Every
stage has a reason for its order and the file it writes; hand-rolling one
produces artifacts the next stage cannot read, and it surfaces three steps later
as something else.

Configuration lives in `model.yml`, once. A knob passed on a command line, set
in a launch script or defaulted in a plugin is a knob the run record cannot
report, and a run that cannot say what trained it is not a measurement.

## Gates are measured, never chosen

A floor is the **Wilson 95% lower bound of the observed rate at that metric's
own denominator**, floored to two places, with the measurement beside it:

```yaml
routing_refusal_recall_rate: 0.84  # 66/71 = 0.930, w95 0.846
conversation_honesty_rate:   0.80  # 16/16 = 1.000, w95 0.806
all_layers_pass_rate:        0.94  # 401/414 = 0.969, w95 0.947
```

Derive them from the accepted release's own report after any run; a folder
should carry a `scripts/derive_gates.py --write` rather than have floors typed
by hand. Denominators differ by an order of magnitude in a multi-family model,
so one shared floor is either unreachable for the small families or vacuous for
the large ones.

Wilson because the floor's distance from the observation should be set by how
much evidence stands behind it: 401/414 yields 0.947, two points of slack, while
16/16 yields 0.806, nineteen. `references/model-yml.md` has why not the observed
rate, why not Wald, and the independence caveat that makes these floors slightly
optimistic.

**Never gate on an aggregate alone.** A pooled rate stays flat while the
composition underneath it moves: the dominant family's own metric climbs as the
displaced families' safety metrics fall, and the summary can read *highest* at
the worst arm.

A `--smoke` run enforces `smoke.gates` instead, and the pass is recorded as
smoke-gated in the run record and in the export manifest's `gate_evidence`, so a
rehearsal never reads later as a production gate pass. `output_nonempty_rate` is
always reported and is what a smoke tier can honestly gate on: it says the
checkpoint saved, reloaded and produced output, not that the output was good.

## Whether a run ships

Three parts, in order. Skipping the third makes release decisions wrong.

1. **Absolute** — every gated metric at or above its floor, at production tier.
2. **Delta** — no gated metric regresses against the accepted release. Exempt
   moves smaller than one row at n>=30; one row is not evidence of decay.
3. **Controlled replay** — when the benchmark changed, replay *both* checkpoints
   over identical rows. A raw delta across a changed benchmark reads benchmark
   hardening as model decay and rejects candidates that are actually better.

A gate is a regression test, so it is silent on any defect the candidate and the
baseline **share**. Only a growing benchmark finds those — which also means
floors must be re-derived from the benchmark in use, or they describe a
population that no longer exists.

## The contract is extracted, not authored

Generate the validator's ground truth from the running system: dump the tool
catalogue from the application's own request builder, walk the live API surface,
read the vocabulary out of the files the application writes. A list transcribed
from documentation is a different version's contract, and it drifts silently.

## Serving

- **Match the training protocol.** If the corpus rendered tools inline (a
  catalogue in the message text, a `{"calls":[...]}` object ending the reply),
  serve with `--server-option tool_style=inline`. The string trained on has to
  be the string served; a server's own tool template renders definitions the
  model never saw.
- **Switch thinking off twice.** `chat_template_kwargs.enable_thinking: false`
  reaches llama.cpp; `reasoning_effort: "none"` reaches LM Studio. No single key
  reaches every server, and a thinking model is not the model that was gated.
- **Put the validator in the path** with `--enforce --max-retries N`: a rejected
  reply goes back to the model carrying the validator's own message and is
  re-asked. That is the decline-with-a-note correction, automated.
- **Q5_K_M over Q4_K_M** for anything citation-shaped. Quantization costs
  citation fidelity specifically, and an aggregate hides it.
- `serve --capture` needs `--auth-token`; captured rows are proposals, and
  `ingest` refuses any still marked unapproved.

## Traps that cost real time

- **The model is right and the format is wrong.** Suspect the serialiser before
  the corpus. `distill.target_format: text` writes *string* targets; anything
  that `json.dumps` them teaches the model to answer inside a quoted literal —
  correct content that no validator accepts.
- **A metric that cannot observe its own failure.** Derive the family a row is
  graded under from the **gold target**, never the prediction. A broken call
  produces no calls, falls to the prose family, leaves the denominator, and the
  metric reads 0 of 0 exactly when it stops working.
- **One word, two meanings.** `family` is both the split group key and the
  contract name. Check which one a site means.
- **Train elsewhere, export at home.** `runs.jsonl` is written where training
  ran and does not travel with the weights; without it an export records
  `gated: false` for a run that passed. Carry the whole run directory
  (`--exclude 'checkpoint-*'`), never a list of name patterns — a pattern list
  carries only what someone remembered to name, and a missing
  `chat_template.jinja` is an adapter that cannot build a prompt.
- **A flag that changes nothing.** A backend taking `**_ignored` accepts flags
  it does not honour, silently. Confirm the behaviour, not the exit code.
- **Silence is not success.** An empty reply, a quiet log, a stalled stream.
  Check the artifact on disk, never the absence of output.

## Before any change to maatml itself lands

```bash
python -m pytest
```

Read the run record before asking why a number moved: `maatml runs <dir>
--compare` puts two side by side.
