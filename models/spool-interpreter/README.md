# Spool Interpreter

Local generative model that interprets sanitized z/OS job spool output into a
structured failure summary so Flow Studio can present "what failed and how to
fix it" without a cloud round-trip.

- **Task string (manifest):** `spool_interpretation`
- **Model id:** `spool-interpreter:v1`
- **Base model:** SmolLM2-360M-Instruct (LoRA fine-tune via PEFT)
- **Runtime backend:** `CandleGenerativeBackend`
- **Response schema:**
  ```json
  {
    "failure_category": "S0C7|JCL_ERROR|...",
    "root_cause": "string",
    "suggested_fix": "string"
  }
  ```

## Dataset

- `datasets/prompt_spec.json` - system prompt + ChatML template + decoding
  params.
- `datasets/schema.json` - sample shape (`raw_spool`, target categories,
  `root_cause`, `suggested_fix`).
- `datasets/label_taxonomy.md` - failure category taxonomy.
- `datasets/samples/seed_samples.jsonl` - hand-authored spool excerpts paired
  with gold interpretations.

## End-to-end commands

```bash
# 1. Prepare splits (60/20/20 hash-stable split of the seed corpus)
python -m flow_ml.cli prepare models/spool-interpreter/

# 2. Smoke training (SmolLM2-135M, 6 steps)
python -m flow_ml.cli train models/spool-interpreter/ --smoke

# 3. Full training (SmolLM2-360M, 5 epochs)
python -m flow_ml.cli train models/spool-interpreter/

# 4. Evaluate the most recent checkpoint
python -m flow_ml.cli evaluate models/spool-interpreter/

# 5. Package -> .fm archive
python -m flow_ml.cli package models/spool-interpreter/ --version v1

# 6. Verify the .fm
python -m flow_ml.cli verify models/spool-interpreter/output/dist/spool-interpreter-v1.fm
```

## Evaluation criteria

`evaluate_spool` computes:

- **`json_validity`** - fraction of generations that parse as the target schema.
- **`category_accuracy`** - exact match on `failure_category`.
- **`root_cause_rouge_l`** - ROUGE-L F1 against gold `root_cause`.

Reports land in `output/eval/<run-name>.{json,md}`.

## Runtime contract

flow-studio's `CandleGenerativeBackend` loads the merged base+LoRA weights,
renders the user message through `prompt_spec.json`'s template, decodes greedy
to the configured stop sequence, then JSON-extracts the response payload.
