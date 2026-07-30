# The validator-gated lifecycle

MaatML's organizing idea is the **Maat weighing thesis**: a model's correctness
is judged *outside* the model, by a **validator** that checks its output against
a contract: a JSON schema, a grammar, node contracts, or task rules. The same
validator is reused at every stage. That reuse is what ties the lifecycle
together, and it is the main thing general fine-tuning tools leave to you.

## One validator, three jobs

```
                    ┌───────────────┐
                    │   validator   │   (per task, registered by a plugin)
                    └───────┬───────┘
          ┌─────────────────┼─────────────────┐
     gates data        gates eval        guards serving
          │                 │                 │
          ▼                 ▼                 ▼
       datagen           evaluate      serve ?validate=1
        ingest            --gate         /predict
```

- **Data:** `maatml datagen` / `maatml ingest` keep only rows whose output
  passes the validator (`build_gated_corpus`). Bad synthetic data never reaches
  training.
- **Evaluation:** `maatml evaluate` scores predictions and enforces
  `evaluation.gates`; `--gate` exits non-zero on failure, so it drops straight
  into CI.
- **Serving:** `maatml serve` re-runs the *same* validator inline when a
  request hits `/predict?validate=1`. It annotates each response with the
  validator result by default, and with `--enforce` it rejects failing outputs
  (HTTP 422), so the contract can hold in production too.

The payoff: a MaatML model ships with a contract, not just weights.

## Registering a validator

A validator is a plugin registration in your model folder (see the
[plugin author guide](plugins.md)):

`ok` is not something you set. It is derived: a result is `ok` once every
required layer has passed, which is what makes a partial pass reportable rather
than a bare true/false.

```python
import json

from maatml.registry import register_validator
from maatml.validation.base import ValidationError, ValidationResult

@register_validator("my_task")
def validate_my_task(raw_output, *, schema_path=None, **kwargs):
    result = ValidationResult(raw_output=raw_output, required_layers={1, 2})

    # Layer 1: it parses.
    try:
        result.parsed = json.loads(raw_output)
    except json.JSONDecodeError as exc:
        result.errors.append(
            ValidationError(layer=1, code="invalid_json", message=str(exc))
        )
        return result
    result.passed_layers.add(1)

    # Layer 2: it satisfies the contract.
    if isinstance(result.parsed, dict) and result.parsed.get("answer"):
        result.passed_layers.add(2)
    else:
        result.errors.append(
            ValidationError(layer=2, code="missing_answer", message="answer required")
        )
    return result
```

Point `model.yml` at it:

```yaml
plugins: [./my_plugin]
evaluation:
  validator: my_task
  metrics: my_task
  gates:
    accuracy: 0.9
```

## Even free text can be gated

Gating is not limited to strict JSON. The
[`vision-vlm`](serving.md) example gates a natural-language image description
with *proxy* metrics (scene-mention rate, shape-mention F1, and brevity),
proving the pattern extends to open-ended output. The validator still decides
what "correct enough to ship" means; only the checks change.

## The lifecycle commands

```bash
maatml prepare   <model-dir>   # build train/val/test splits
maatml train     <model-dir>   # fine-tune (LoRA/QLoRA/full/DPO/ORPO/vision/VLM)
maatml evaluate  <model-dir>   # validator + metrics + gates
maatml export    <model-dir>   # safetensors/gguf/mlx/onnx + manifest.json
maatml verify    <export-dir>  # sha256 check vs manifest
maatml serve     <model-dir>   # JSON inference API (validator inline)
```

## One command for the whole lifecycle

```bash
maatml run <model-dir>            # prepare -> train -> evaluate -> export -> verify
maatml run <model-dir> --smoke    # the same walk at the smoke tier
maatml run <model-dir> --dry-run  # what is stale, and why
```

`maatml run` walks the fixed pipeline in order and stops non-zero at the first
failure, so one green line means every stage passed rather than "the last
command I happened to type passed". Gates are enforced at the evaluate step,
which is what makes the line worth anything.

Steps that are already fresh are skipped. Freshness is decided by a fingerprint
over what the step consumed: effective config after `--smoke` and `--set`, the
declared input files, the upstream step's fingerprint, the maatml version and
git SHA, the plugin sources, the device profile, and the exporter. It lives in
`output/pipeline.json` and exists for **idempotence, not speed**: a step is
skipped only when its fingerprint matches, the step completed last time, and
its outputs are still there. `--dry-run` prints which component changed.

```
step      action   reason
prepare   skip     up to date
train     run      changed: training_config
evaluate  run      changed: upstream
```

The source operations (`datagen`, `distill`, `ingest`, `mint`, and reviewed
`serve --capture`) stay outside the runner. They change the seed corpus, and
that is precisely what makes `prepare` stale on the next run. See
[the data flywheel](flywheel.md) for how each one gates what it adds.

`--from` / `--until` restrict the walk, `--force` re-runs everything selected,
and `--set` overrides feed the fingerprint (an invalid one exits non-zero
before any step runs and leaves `output/pipeline.json` untouched).

### Smoke-tier gates

A `--smoke` rehearsal cannot meet production thresholds, and holding it to them
would only teach people to ignore a red gate. A `smoke:` block may declare its
own `gates:`, which `--smoke` runs enforce instead. The pass is recorded as
smoke-gated in the run record and in the export manifest's `gate_evidence`, so
a rehearsal never reads as a production gate pass later.

maatml always reports `output_nonempty_rate` alongside a model's own metrics,
which is what a smoke tier can honestly gate on: it says the checkpoint saved,
reloaded, and produced output, without claiming the output was any good.

`maatml plan <model-dir>` is the same view as `run --dry-run`.

## What each stage refuses to do quietly

- `prepare` splits by group key (`dataset.group_by`, else `family` → `source` →
  `sample_id`). A key covering nearly the whole corpus cannot be split, so
  those rows are split individually with a warning: group-level leakage
  protection does not apply to them. An empty split is reported, and a
  `benchmark_samples` row sharing a group key with the training splits is an
  error, because a benchmark is pinned to test.
- `train` fails on gold labels no head declares, on a seq2seq corpus with no
  targets, and on an unsupported `training.precision`. Any failure marks the
  run `aborted` in `runs.jsonl`.
- `evaluate` uses `packaging.max_input_tokens` as its token budget (the same
  budget serve and `export --parity` enforce) and records how many inputs it
  truncated. Per-class output is a pass rate with its sample count, not a
  confusion matrix.
- `sweep` records a failed trial and keeps going, then exits non-zero. It ranks
  only trials that reported the metric being ranked.
