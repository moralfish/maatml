# JCL Validator

ModernBERT-base multi-head classifier that validates a sanitized JCL document
and emits a structured `JclValidationResult` JSON object. Uses a custom
column-aware JCL BPE tokenizer. Full fine-tune (no LoRA).

Version: **0.1.0** (`model.yml`). Bump major for breaking output-schema changes,
minor for retrain/data/config changes, patch for metadata-only edits.

## Targets

- **Cross-platform local inference**: Mac, Windows, Linux with 16 GB RAM minimum.
- **Final artefact**: safetensors checkpoint (encoder + classifier heads sidecar).
- **Disk footprint**: ~150–200 MB.

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

Full enumeration in [`datasets/node_contracts.json`](https://github.com/moralfish/maatml/blob/main/examples/jcl-validator/datasets/node_contracts.json)
and the JSON Schema at
[`datasets/jcl_validation_schema.json`](https://github.com/moralfish/maatml/blob/main/examples/jcl-validator/datasets/jcl_validation_schema.json).

## Layout

```
examples/jcl-validator/
├── README.md
├── model.yml
└── datasets/
    ├── prompt_spec.json
    ├── jcl_validation_schema.json
    ├── node_contracts.json
    ├── tokenizer.json          # custom JCL BPE (required before training)
    └── samples/
        ├── seed_samples.jsonl
        └── test_prompt_set.jsonl
```

## Workflow

```bash
maatml prepare  examples/jcl-validator/
maatml train    examples/jcl-validator/ --smoke
maatml train    examples/jcl-validator/
maatml evaluate examples/jcl-validator/
```

Expand the seed corpus by hand-authoring rows in
`datasets/samples/seed_samples.jsonl` (each row:
`{sample_id, source, category, request, expected_validation_result, split?}`)
and re-running `maatml prepare` before training — or regenerate via
`examples/jcl-validator/scripts/build_seeds.py`.

## Quality gates

| Metric | Target |
|---|---|
| `json_parse_rate` | ≥ 0.95 |
| `schema_conformance_rate` | ≥ 0.90 |
| `severity_accuracy` | ≥ 0.85 |
| `code_accuracy` | ≥ 0.85 |
| `valid_flag_accuracy` | ≥ 0.95 |
| `line_within_3_accuracy` | ≥ 0.70 |

`line_within_3` allows ±3 lines of slack on the predicted error line — practical
UI tolerance.
