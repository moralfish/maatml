# The data flywheel: growing a corpus that stays gated

Training data is the scarce input, and a validator-gated model turns its own
contract into a way to make more of it. Every operation here is **explicit** and
every row it produces is gated by the same `evaluation.validator` that grades
the model — so a row that enters the corpus is a row the model is measured
against.

None of these are steps of `maatml run`. They change the seed corpus, which is
exactly what makes `prepare` stale on the next run:

```
datagen / distill / ingest / mint / (reviewed capture)
        |  append validator-gated rows to the seed corpus
        v
maatml run          # prepare is stale, so the loop retrains
```

## Contents

- [Choosing an operation](#choosing-an-operation)
- [datagen: generate whole rows](#datagen-generate-whole-rows)
- [distill: label a prompt pool](#distill-label-a-prompt-pool)
- [ingest: import external rows](#ingest-import-external-rows)
- [mint: preference pairs](#mint-preference-pairs)
- [Reviewed capture](#reviewed-capture)
- [Making a teacher earn its keep](#making-a-teacher-earn-its-keep)

## Choosing an operation

| You have | Use |
|---|---|
| nothing but a contract | `datagen` |
| prompts, but no labels | `distill` |
| rows from somewhere else | `ingest` |
| candidate completions to rank | `mint` |
| real traffic worth learning from | `serve --capture`, then review, then `ingest` |

## `datagen`: generate whole rows

Runs a registered generator (or a teacher) and keeps only rows the validator
accepts. It **fails closed** when no validator is configured unless you pass
`--allow-ungated` — ungated synthetic data teaches a contract nobody checks.

## `distill`: label a prompt pool

Where `datagen` invents whole rows, `distill` starts from prompts you already
have and asks a teacher only for the label. Every response is gated before it
enters the corpus, so a wrong label is dropped rather than trusted.

```bash
maatml distill <model-dir> --prompts prompts.jsonl
maatml distill <model-dir> --replay      # reproduce the corpus offline, no network
```

Accepted rows carry provenance (teacher model and revision, prompt hash, source,
family); rejections are kept in a report next to the corpus, which is worth
reading — a cluster of identical rejections is usually a briefing problem, not a
teacher problem.

Responses are cached on prompt hash **plus teacher model and revision**, so
`--replay` reproduces exactly the same accepted corpus and a different teacher
never silently reuses another's labels.

The teacher is any OpenAI-compatible endpoint, named by
`MAATML_TEACHER_BASE_URL` and `MAATML_TEACHER_API_KEY` — so serving one locally
is a base URL, not a code path. Keep the key in `.env`, never committed.

Configure it in `model.yml` under `distill:` rather than on the command line, so
the run record can say what labelled the corpus. See `model-yml.md`, especially
`target_format`: left at `json`, a non-JSON label is refused as unparseable and
the gate that decides never sees it.

## `ingest`: import external rows

Maps external columns into the seed shape, optionally sanitizes, and validates
gold targets when a validator is configured. It also guards the capture loop: a
`serve_capture` row is refused unless a reviewer approved it.

## `mint`: preference pairs

Turns candidate completions into `{prompt, chosen, rejected}` pairs for DPO or
ORPO. For each prompt the validator splits candidates into pass and fail; a
prompt with both yields one pair.

```bash
maatml mint <dpo-model-dir> --input candidates.jsonl
```

So a minted pair means "this completion passes the contract and that one does
not" — not a hand-labelled guess. Input is JSONL of
`{prompt, candidates: [completion, ...]}`; pairs append stamped `source: mint`.

Preference training needs volume and balance. A hundred approvals against nine
declines is too thin and too imbalanced to mint from; filtering supervised
training to approved actions is the better use of that signal.

## Reviewed capture

`maatml serve --capture` records real traffic. A captured prediction is **not**
training data — it is a proposal a human or teacher must correct and approve.

```bash
maatml serve <model-dir> --auth-token "$TOKEN" --capture captures.jsonl
# review captures.jsonl: fix the target, set "approved": true on keepers
maatml ingest <model-dir> --input captures.jsonl    # refuses unapproved rows
maatml run <model-dir>                               # retrains on the new seeds
```

Capture requires the auth token (an open capture endpoint is an unbounded write
sink and a way to poison the corpus), the file is size-capped, and `ingest`
refuses any row still marked unapproved.

## Making a teacher earn its keep

Teacher configuration is not a detail — it is often the difference between a
corpus and a pile of rejects. Measured on one 80-prompt pool:

| change | accepted |
|---|---|
| `max_tokens: 4096`, thinking flag set | 11 / 80 |
| `max_tokens: 12288` | 27 / 80 |
| `reasoning_effort: none`, one worked example in the brief | 56 / 80 |

Three things to take from that:

- **A thinking teacher can spend its whole budget thinking.** One prompt used
  10,706 of 12,135 completion tokens on hidden reasoning with the "disable"
  flag set. `reasoning_effort: none` zeroed it and cut that prompt from 117s to
  4s. Check where the tokens went before raising the budget.
- **Budget failures look like refusals.** Answers that are simply cut off arrive
  as unparseable or empty, and read as a teacher that cannot do the task.
- **The briefing is part of the contract surface.** A flat rendering of the tool
  signature produced flat arguments on 21 of 80 replies; one *worked call* in
  the brief roughly doubled acceptance. Show the shape you want, do not describe
  it.

If acceptance is low, read the rejection report before changing the model. The
pattern in the rejects usually names the fix.
