# flow-ml architecture overview

flow-ml is a **training and fine-tuning framework** for task-specific language
models. Each model is a self-describing folder with a `model.yml` that drives
prepare → train → evaluate.

## Reference models

| Model | Task | Architecture | Base |
|---|---|---|---|
| jcl-validator | classify JCL syntax errors | 4-head classifier | ModernBERT-base |
| spool-interpreter | summarise z/OS spool output as JSON | seq2seq | flan-t5-base |

Each model folder carries `model.yml`, `datasets/` (seeds + schemas + specs),
and `output/` (prepared splits, checkpoints, eval reports — gitignored).

## Workflow

```bash
flow_ml prepare  models/<name>/             # → output/prepared/{train,val,test}.jsonl
flow_ml train    models/<name>/ [--smoke]   # → output/checkpoints/
flow_ml evaluate models/<name>/             # → output/eval/{ckpt}.{json,md}
flow_ml plugins                             # list discovered plugin registrations
```

CLI dispatcher: [src/flow_ml/cli.py](src/flow_ml/cli.py). Subcommands route by
`model.yml.task` and `model.yml.architecture`.

## Architecture dispatch

```
task=jcl_validation       + architecture=classifier → training/jcl_classifier.py
task=spool_interpretation + architecture=seq2seq    → training/spool_seq2seq.py
```

Contrib plugins register via the `flow_ml.plugins` entry-point group
(`flow_ml.contrib.jcl`, `flow_ml.contrib.spool`).

## Package layout

```
src/flow_ml/
├── cli.py                          # `flow_ml <cmd>` dispatcher
├── config.py                       # ModelDefinition (model.yml loader)
├── registry.py                     # plugin registry / discovery
├── contrib/                        # reference task plugins (jcl, spool)
├── data/
│   ├── pipeline.py                 # prepare_jcl / prepare_spool
│   ├── schemas.py                  # sample types for the two tasks
│   ├── sanitizer.py                # PII redaction for spool/jcl
│   └── synthetic/                  # rule-based JCL corpus generator
├── training/
│   ├── jcl_classifier.py           # ModernBERT 4-head classifier
│   ├── spool_seq2seq.py            # flan-t5 seq2seq
│   └── sft_base.py                 # shared causal-SFT skeleton
├── validation/                     # per-task out-of-model validators
├── evaluation/runner.py            # evaluate_{jcl,spool}
└── tokenization/jcl_tokenizer.py   # column-aware JCL BPE (+ COLUMN_RULES.md)
```

## Conventions

- **Determinism**: prepare uses an explicit `seed` in `model.yml`.
- **Validator-gated corpora**: seed builders reject rows that fail the
  per-task validator before they land in `seed_samples.jsonl`.
- **Smoke profile**: cheaper `smoke:` overlays on `training:` for fast
  pipeline checks.

## Tests

```bash
.venv/bin/python -m pytest tests/
```

See [README.md](README.md) for install and CLI details.
