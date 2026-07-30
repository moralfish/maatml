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
  "summary": "Job abended with S0C7 before completion.",
  "status": "abended",
  "returnCode": null,
  "rootCause": "IEF450I PA700043 GO - ABEND S0C7 U0000 - TIME=21.52.07",
  "suggestedFix": "Inspect the abend code and the failing step's input data, correct, and rerun the step.",
  "failureCategory": "execution_abend",
  "confidence": 0.93
}
```

## Corpus

The seed corpus is **real JES2 output**: 1,200 job documents captured from
MVS 3.8j running under Hercules (public-domain OS, public container), with a
600-document benchmark captured from a disjoint generation seed and verified
to share no normalized document text with the seed side. Gold labels are
extracted from the system's own messages (completion codes, abend codes,
converter diagnostics), so every row's `status`/`returnCode`/`failureCategory`
is what actually happened. The capture kit lives outside this repo;
`scripts/build_seeds.py` regenerates the legacy synthetic corpus and will
overwrite the captured one, so do not run it casually.

`status` is one of: `completed`, `failed`, `abended`, `skipped`, `running`.
`failureCategory` is one of 8 enum values (see `node_contracts.json`)
or `null` on `status: completed`. `explanation` must be non-empty when
`status != "completed"`; `relatedDocs` is a list of doc keys.

Full enumeration in [`datasets/node_contracts.json`](https://github.com/moralfish/maatml/blob/main/examples/spool-interpreter/datasets/node_contracts.json)
and the JSON Schema at
[`datasets/spool_interpretation_schema.json`](https://github.com/moralfish/maatml/blob/main/examples/spool-interpreter/datasets/spool_interpretation_schema.json).

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
and re-running `maatml prepare` before training, or regenerate via
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
