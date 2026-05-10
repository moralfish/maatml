# Spool Interpreter

LoRA-tuned `Qwen/Qwen3-1.7B` that reads sanitized z/OS spool output and emits a structured `SpoolInterpretation` JSON object. Pure SFT on 3-message conversations. Replaces the legacy SmolLM2-360M baseline — same task, much sharper categorisation and confidence calibration.

Authoritative spec: [`training_instructions.md`](training_instructions.md).

## Targets

- **Cross-platform local inference**: Mac, Windows, Linux with 16 GB RAM minimum.
- **Final artefact**: merged safetensors loaded by `flow-model-runtime` (Candle).
- **Disk footprint**: ~3.4 GB at fp16. Same envelope as FlowGraphGenerator + JCL Validator.

## Output shape

```json
{
  "summary": "Build job ABENDed at STEP02 (S0C7).",
  "status": "abended",
  "returnCode": null,
  "rootCause": "Data exception (S0C7) reading numeric field with invalid packed-decimal data in INFILE record 17.",
  "suggestedFix": "Validate INFILE before submission; clean or reject record 17.",
  "failureCategory": "execution_abend",
  "confidence": 0.91
}
```

`status` is one of: `completed`, `failed`, `abended`, `skipped`, `running`.
`failureCategory` is one of 12 enum values (incl. 4 Smart/RESTART subcategories) or `null` on `status: completed`.

Full enumeration in [`datasets/node_contracts.json`](datasets/node_contracts.json) and the JSON Schema at [`datasets/spool_interpretation_schema.json`](datasets/spool_interpretation_schema.json).

## Layout

```
models/spool-interpreter/
├── README.md
├── training_instructions.md
├── model.yml
└── datasets/
    ├── prompt_spec.json
    ├── spool_interpretation_schema.json
    ├── node_contracts.json
    └── samples/
        ├── seed_samples.jsonl       (39 hand-authored; expand by hand-authoring more rows)
        └── test_prompt_set.jsonl    (8 fixed eval anchors)
```

## Workflow

```bash
flow_ml prepare  models/spool-interpreter/
flow_ml train    models/spool-interpreter/ --smoke
flow_ml train    models/spool-interpreter/
flow_ml evaluate models/spool-interpreter/
flow_ml package  models/spool-interpreter/ --version v0.1
```

Expand the seed corpus by hand-authoring rows in
`datasets/samples/seed_samples.jsonl` (each row:
`{sample_id, source, category, request, expected_interpretation, split?}`)
and re-running `flow_ml prepare` before training.

## Quality gates

| Metric | Target |
|---|---|
| `json_parse_rate` | ≥ 0.95 |
| `schema_conformance_rate` | ≥ 0.90 |
| `status_accuracy` | ≥ 0.90 |
| `failure_category_accuracy` | ≥ 0.80 |
| `return_code_accuracy` | ≥ 0.85 (when present) |

## Smart/RESTART knowledge sync

The 4 Smart/RESTART subcategories (resource_unavailable, configuration, application_logic, input_syntax) come from `../flow-studio/docs/smart-restart/messages.md`. Re-sync via [`scripts/sync-smart-restart-knowledge.sh`](../../scripts/sync-smart-restart-knowledge.sh) when that source updates.
