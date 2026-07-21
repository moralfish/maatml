# Spool Interpreter

flan-t5-base seq2seq model that reads sanitized z/OS spool output and emits a
structured `SpoolInterpretation` JSON object. Full fine-tune (no LoRA).

Version: **0.1.0** (`model.yml`). Bump major for breaking output-schema changes,
minor for retrain/data/config changes, patch for metadata-only edits.

## Targets

- **Cross-platform local inference**: Mac, Windows, Linux with 16 GB RAM minimum.
- **Final artefact**: safetensors checkpoint.
- **Disk footprint**: ~600 MB at fp16.

## Output shape

```json
{
  "summary": "Build job ABENDed at STEP02 (S0C7).",
  "status": "abended",
  "returnCode": null,
  "rootCause": "Data exception (S0C7) reading numeric field with invalid packed-decimal data in INFILE record 17.",
  "suggestedFix": "Validate INFILE before submission; clean or reject record 17.",
  "explanation": "STEP02 failed with S0C7 while reading INFILE. Record 17 contained invalid packed-decimal data in a numeric field.",
  "relatedDocs": ["s0c7-data-exception"],
  "failureCategory": "execution_abend",
  "confidence": 0.91
}
```

`status` is one of: `completed`, `failed`, `abended`, `skipped`, `running`.
`failureCategory` is one of 8 enum values (see `node_contracts.json`)
or `null` on `status: completed`. `explanation` must be non-empty when
`status != "completed"`; `relatedDocs` is a list of doc keys.

Full enumeration in [`datasets/node_contracts.json`](datasets/node_contracts.json)
and the JSON Schema at
[`datasets/spool_interpretation_schema.json`](datasets/spool_interpretation_schema.json).

## Layout

```
examples/spool-interpreter/
├── README.md
├── model.yml
└── datasets/
    ├── prompt_spec.json
    ├── spool_interpretation_schema.json
    ├── node_contracts.json
    └── samples/
        ├── seed_samples.jsonl
        └── test_prompt_set.jsonl
```

## Workflow

```bash
maatml prepare  examples/spool-interpreter/
maatml train    examples/spool-interpreter/ --smoke
maatml train    examples/spool-interpreter/
maatml evaluate examples/spool-interpreter/
```

Expand the seed corpus by hand-authoring rows in
`datasets/samples/seed_samples.jsonl` (each row:
`{sample_id, source, category, request, expected_interpretation, split?}`)
and re-running `maatml prepare` before training — or regenerate via
`examples/spool-interpreter/scripts/build_seeds.py`.

## Quality gates

| Metric | Target |
|---|---|
| `json_parse_rate` | ≥ 0.95 |
| `schema_conformance_rate` | ≥ 0.90 |
| `status_accuracy` | ≥ 0.90 |
| `failure_category_accuracy` | ≥ 0.80 |
| `return_code_accuracy` | ≥ 0.85 (when present) |
| `explanation_present_rate` | ≥ 0.95 (when status ≠ completed) |
| `related_docs_coverage_rate` | ≥ 0.90 |
