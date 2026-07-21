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
registrations run at import time.

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
appends accepted rows to `dataset.seed_samples`.

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

## Deprecation policy

- Semver for the `maatml` package; model folders version independently in
  `model.yml`.
- Registry names are sticky once published in an example; rename with a
  temporary dual-register period.
- CLI flags may gain aliases; removals land in a minor with a CHANGELOG note.
