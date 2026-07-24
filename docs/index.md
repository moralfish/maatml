# MaatML

MaatML fine-tunes small, task-specific models across **text, vision, and
vision-language**, and takes them from experimentation to production through one
declarative `model.yml`: **prepare → train → evaluate → export → serve**.

**What makes it different:** correctness is checked *outside* the model by
[validators](lifecycle.md). The same validator gates your synthetic **data** and
your **evaluation**, and guards your **live inference**: `maatml serve` runs it
on every response, reporting the result by default and rejecting outputs that
fail under `--enforce`. So a MaatML model ships with a contract, not just
weights. That validator-gated *data → eval → serving* loop, now across
modalities, is what general fine-tuning tools leave out.

**Install from PyPI** (not from source):

```bash
pip install maatml
pip install "maatml[ml]"       # training stack
pip install "maatml[ml,vision]"  # + torchvision and ONNX (vision / VLM examples)
pip install "maatml[docs]"     # this site: mkdocs serve
```

- Site: [maatml.pages.dev](https://maatml.pages.dev)
- PyPI: [pypi.org/project/maatml](https://pypi.org/project/maatml/)
- Source: [github.com/moralfish/maatml](https://github.com/moralfish/maatml)

## Documentation

- [Get started](getting-started.md): install and serve your first model in 5 minutes
- [The validator-gated lifecycle](lifecycle.md): the core idea, end to end
- [Serving & deployment](serving.md): `maatml serve`, ONNX/edge, and vLLM (VLMs)
- [Plugin author guide](plugins.md): register trainers, validators, metrics, exporters, generators
- [Examples](examples/index.md): six reference models, from support-ticket triage to a vLLM-servable VLM

Canonical references in the repo:

- [README.md](https://github.com/moralfish/maatml/blob/main/README.md): install, CLI overview, examples
- [ROADMAP.md](https://github.com/moralfish/maatml/blob/main/ROADMAP.md): tranche status
- [CHANGELOG.md](https://github.com/moralfish/maatml/blob/main/CHANGELOG.md): release notes

## Quick CLI map

| Command | Purpose |
|---------|---------|
| `maatml prepare` | Build train/val/test splits |
| `maatml train` | Fine-tune (LoRA / QLoRA / full / DPO / ORPO / vision / VLM) |
| `maatml evaluate` | Validator + metrics + eval gates (`--gate` fails CI) |
| `maatml export` | Bundle checkpoint (safetensors / gguf / mlx / onnx) + `manifest.json` |
| `maatml verify` | Recompute sha256 of manifest files |
| `maatml serve` | JSON inference API, validator inline (`/predict?validate=1`) |
| `maatml datagen` / `ingest` | Validator-gated data flywheel |

See the [Plugin author guide](plugins.md) for registering generators, exporters,
validators, and metrics.
