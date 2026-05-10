# JCL Validator

LoRA-tuned `Qwen/Qwen3-1.7B` that validates a sanitized JCL document and emits a structured `JclValidationResult` JSON object. Pure SFT on 3-message conversations (system, user, assistant). Replaces the legacy multi-head BERT classifier — same task, better interpretability, structured output that downstream code can act on.

Authoritative spec: [`training_instructions.md`](training_instructions.md).

## Targets

- **Cross-platform local inference**: Mac, Windows, Linux with 16 GB RAM minimum.
- **Final artefact**: merged safetensors loaded by `flow-model-runtime` (Candle).
- **Disk footprint**: ~3.4 GB at fp16. Same envelope as FlowGraphGenerator.

## Output shape

```json
{
  "valid": false,
  "errors": [
    {
      "line": 7,
      "severity": "error",
      "code": "missing_dd",
      "message": "IEBGENER step missing SYSUT1 and SYSUT2 DD statements.",
      "suggestion": "Add `//SYSUT1 DD ...` and `//SYSUT2 DD ...`."
    }
  ],
  "confidence": 0.91
}
```

`valid: true` iff `errors: []`. Cap of 5 errors per result. `code` is one of:
`missing_dd`, `invalid_job_card`, `unresolved_symbolic_parameter`,
`continuation_error`, `invalid_exec_statement`,
`invalid_dataset_reference_structure`, `other`, `none`.

Full enumeration in [`datasets/node_contracts.json`](datasets/node_contracts.json) and the JSON Schema at [`datasets/jcl_validation_schema.json`](datasets/jcl_validation_schema.json).

## Layout

```
models/jcl-validator/
├── README.md
├── training_instructions.md
├── model.yml
└── datasets/
    ├── prompt_spec.json
    ├── jcl_validation_schema.json
    ├── node_contracts.json
    └── samples/
        ├── seed_samples.jsonl
        └── test_prompt_set.jsonl
```

## Workflow

```bash
flow_ml prepare  models/jcl-validator/
flow_ml train    models/jcl-validator/ --smoke   # ~2 min on Qwen3-0.6B
flow_ml train    models/jcl-validator/           # ~30-40 min on Qwen3-1.7B (M5 Max bf16)
flow_ml evaluate models/jcl-validator/
flow_ml package  models/jcl-validator/ --version v0.1
```

Expand the seed corpus by hand-authoring rows in
`datasets/samples/seed_samples.jsonl` (each row:
`{sample_id, source, category, request, expected_validation_result, split?}`)
and re-running `flow_ml prepare` before training.

## Quality gates

| Metric | Target |
|---|---|
| `json_parse_rate` | ≥ 0.95 |
| `schema_conformance_rate` | ≥ 0.90 |
| `severity_accuracy` | ≥ 0.85 |
| `code_accuracy` | ≥ 0.85 |
| `valid_flag_accuracy` | ≥ 0.95 |
| `line_within_3_accuracy` | ≥ 0.70 |

`line_within_3` allows ±3 lines of slack on the predicted error line — practical UI tolerance.

## Latency note

Generative inference on Qwen3-1.7B is ~1–2 s per sample on M5 Max bf16. The legacy BERT classifier was ~100 ms. Acceptable trade-off for the structured-output + pinpoint-line + suggestion gains, but real-time use cases (sub-200 ms SLA) should batch validations or run async.
