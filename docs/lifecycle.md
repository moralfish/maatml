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

```python
from maatml.registry import register_validator
from maatml.validation.base import ValidationResult

@register_validator("my_task")
def validate_my_task(raw_output, *, schema_path=None, **kwargs):
    ...  # parse raw_output, check the contract
    return ValidationResult(ok=True, errors=[])
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

`maatml plan <model-dir>` prints this sequence for any model folder.

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
