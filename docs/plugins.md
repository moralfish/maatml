# Plugin author guide

MaatML core owns architectures and harnesses. **Examples / model folders own
task semantics** and register them via decorators.

## Registries

| Kind | Decorator | Typical use |
|------|-----------|-------------|
| trainer | `@register_trainer` | Architecture training loop |
| validator | `@register_validator` | Out-of-model JSON / contract gate |
| metrics | `@register_metrics` | Eval scoring |
| predictor | `@register_predictor` | Checkpoint → text / structured output |
| format | `@register_format` | Dataset prepare adapters |
| sanitizer | `@register_sanitizer` | Regex PII / domain scrubbing |
| transform | `@register_transform` | Text pre-tokenization |
| generator | `@register_generator` | `maatml datagen` candidate factories |
| exporter | `@register_exporter` | `maatml export --format …` |

List everything with `maatml plugins`.

## Folder-local plugins

In `model.yml`:

```yaml
plugins: [./jcl_plugin]
dataset:
  generator: jcl
evaluation:
  validator: jcl
  metrics: jcl
  predictor: jcl_classifier
```

`load_model_plugins` imports the package (or `.py` file); side-effect
registrations run at import time. It is the single owner of that import and is
idempotent, so a folder's plugin code runs once per process no matter how many
commands (or library calls) ask for it. Pass `force=True` to re-execute it.

A plugin source that fails to import is recorded rather than skipped in
silence: `maatml plugins` lists the failures under `unavailable`, and an
`Unknown … plugin` error names them, which is usually why a name is missing.

> **Trust boundary.** These imports run arbitrary Python at load time. Because
> every command reads `model.yml`, even `maatml validate` and `maatml plan`
> execute a folder's plugins. Only point maatml at model folders you trust, or
> use `maatml validate --no-plugins` to check schema and paths without importing
> plugin code.

## Generators (`maatml datagen`)

A generator is a factory:

```python
from maatml.registry import register_generator

@register_generator("my_task")
def my_generator(model_def, *, seed: int = 0, **kwargs):
    def generate_fn():
        return {"sample_id": "...", "request": "...", "target": {...}}
    return generate_fn
```

Core runs `build_gated_corpus(generate_fn, validate_fn, target_n=…)` and
appends accepted rows to `dataset.seed_samples`, skipping rows already in the
corpus (matched by `sample_id` or content). Returning `None` means "no
candidate this time" and costs one attempt; raising is recorded as a rejected
row. Raise `maatml.data.gated.GenerationAbort` to stop the run immediately
(the teacher client does this after five consecutive request failures).

Stamp a `family` (or whatever `dataset.group_by` names) on generated rows.
Rows that share one group key cannot be split, so a corpus where every row
carries the same key is split per row with a warning instead.

Optional teacher path: `maatml datagen --teacher` uses
`MAATML_TEACHER_BASE_URL` / `MAATML_TEACHER_API_KEY` (`pip install maatml[teacher]`).

## Exporters (`maatml export`)

Built-ins: `safetensors` (always), `gguf` / `mlx` (optional tooling). Custom:

```python
@register_exporter("my_fmt")
def export_my_fmt(model_def, checkpoint_dir, out_dir, *, run_id=None):
    ...
    return out_dir
```

Always write / update `manifest.json` via `maatml.export.manifest`.

## Testing plugins

Registries are process-global. Snapshot and restore them around a test through
the public API instead of touching registry internals:

```python
from maatml.registry import restore_registries, snapshot_registries

snapshot = snapshot_registries()
try:
    ...  # register, load a model folder, assert
finally:
    restore_registries(snapshot)
```

`reset_registries()` wipes every registry (and forgets which model-folder
plugins have run) for a blank slate; `reset_registries(rediscover=True)`
re-imports the built-ins afterwards. `REGISTRY.unregister(name)` drops one
entry. `discover_plugins()` only adds registrations, so it never removes what a
model folder registered.

## Deprecation policy

- Semver for the `maatml` package; model folders version independently in
  `model.yml`.
- Registry names are sticky once published in an example; rename with a
  temporary dual-register period.
- CLI flags may gain aliases; removals land in a minor with a CHANGELOG note.
