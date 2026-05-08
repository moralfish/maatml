# JCL Validator

Multi-head BERT classifier that flags JCL syntax/semantic errors before a job
is submitted to z/OS. Trained on synthetically generated JCL with injected
defects.

- **Task string (manifest):** `jcl_validation`
- **Model id:** `jcl-validator:v1`
- **Base model:** `google-bert/bert-base-uncased`
- **Runtime backend:** `CandleBackend` (BERT-arch with three flow-specific heads)

## Heads

The model emits three classification heads simultaneously, scored against
`packaging.confidence_thresholds`:

- **seq** - is this JCL valid as a whole? (binary)
- **cat** - error category (one of seven; see `datasets/label_taxonomy.md`).
- **line** - line-level token classification (which line carries the defect).

`training.head_weights` controls how much each head contributes to the loss.

## Dataset

- `datasets/schema.json` - sample shape (sanitized JCL + per-head labels).
- `datasets/label_taxonomy.md` - the seven error categories with examples.
- `datasets/templates/*.jcl` - 8 valid JCL templates (single-step, multi-step,
  proc call, with set, IEBGENER, COBOL compile, concat DD, sort step) used by
  `flow_ml.data.synthetic.jcl_generator` to render valid samples and inject
  defects.

`prepare jcl` runs the synthetic generator and writes train/val/test JSONLs
into `output/prepared/`. Sizes default to 2000 per error class plus 2000 valid.

## End-to-end commands

```bash
# 1. Prepare splits (synthetic generation, ~16,000 samples by default)
python -m flow_ml.cli prepare models/jcl-validator/

# 2. Smoke training (tiny BERT, 10 steps)
python -m flow_ml.cli train models/jcl-validator/ --smoke

# 3. Full training (BERT-base, 3 epochs)
python -m flow_ml.cli train models/jcl-validator/

# 4. Evaluate the most recent checkpoint
python -m flow_ml.cli evaluate models/jcl-validator/

# 5. Package -> .fm archive
python -m flow_ml.cli package models/jcl-validator/ --version v1

# 6. Verify the .fm
python -m flow_ml.cli verify models/jcl-validator/output/dist/jcl-validator-v1.fm
```

## Evaluation criteria

`evaluate_jcl` computes per-head metrics:

- **seq** accuracy and F1.
- **cat** macro-F1 across the seven categories.
- **line** token-level F1.

Reports land in `output/eval/<run-name>.{json,md}`.

## Runtime contract

The Candle backend at `flow-starter/crates/flow-model-runtime/src/backends/jcl.rs`
loads `model.safetensors` plus the auxiliary `flow_heads.safetensors` and
`labels.json`, runs a single forward pass, and returns the three head outputs.
A `JclValidation` Tauri command in flow-starter wraps this for the canvas's
"Validate JCL" node.
