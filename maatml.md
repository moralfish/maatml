# maatml architecture overview

MaatML is a **training and fine-tuning framework** for task-specific language
models — from experimentation to production. Each model is a self-describing
folder with a `model.yml` that drives prepare → train → evaluate.

**Design rule:** core owns architectures; examples own task semantics.

## Example models

| Model | Task | Architecture | Base |
|---|---|---|---|
| jcl-validator | classify JCL syntax errors | multi-head classifier | ModernBERT-base |
| spool-interpreter | summarise z/OS spool output as JSON | seq2seq | flan-t5-base |
| support-ticket-triage | triage support tickets | causal SFT (LoRA) | Qwen3-0.6B |

Each example folder carries `model.yml`, optional `*_plugin/`, `datasets/`,
`scripts/`, and `output/` (prepared splits, checkpoints, eval — gitignored).

## Workflow

```bash
maatml prepare  examples/<name>/             # → output/prepared/{train,val,test}.jsonl
maatml train    examples/<name>/ [--smoke]   # → output/checkpoints/
maatml evaluate examples/<name>/             # → output/eval/{ckpt}.{json,md}
maatml plugins                               # list discovered plugin registrations
```

CLI dispatcher: [src/maatml/cli.py](src/maatml/cli.py). Subcommands route by
`model.yml.architecture` and registered plugins.

## Architecture dispatch

```
architecture=classifier / multi_head_classifier → training/multi_head.py
architecture=seq2seq                            → training/seq2seq.py
architecture=causal_sft                         → training/sft_base.py
```

Task validators / metrics / sanitizers register from example
`plugins: [./jcl_plugin]` (or `./spool_plugin`) packages — not from core
entry points.

## Package layout

```
src/maatml/
├── cli.py                          # `maatml <cmd>` dispatcher
├── config.py                       # ModelDefinition (model.yml loader)
├── registry.py                     # plugin registry / discovery
├── data/
│   ├── pipeline.py                 # generic prepare (SANITIZERS registry)
│   ├── schemas.py                  # Split enum only
│   └── sanitizer.py                # generic regex rule engine
├── training/
│   ├── multi_head.py               # config-driven multi-head classifier
│   ├── seq2seq.py                  # generic encoder-decoder
│   └── sft_base.py                 # causal-SFT skeleton
├── validation/base.py              # shared ValidationResult / fence strip
└── evaluation/                     # harness + generic predictors

examples/
├── jcl-validator/jcl_plugin/       # validator, metrics, predictor, tokenizer, …
└── spool-interpreter/spool_plugin/ # validator, metrics, sanitizer
```

## Conventions

- **Determinism**: prepare uses an explicit `seed` in `model.yml`.
- **Validator-gated corpora**: seed builders reject rows that fail the
  per-task validator before they land in `seed_samples.jsonl`.
- **Smoke profile**: cheaper `smoke:` overlays on `training:` for fast
  pipeline checks.

## Tests

```bash
.venv/bin/python -m pytest tests/ examples/ -q
```

See [README.md](README.md) for install and CLI details.
