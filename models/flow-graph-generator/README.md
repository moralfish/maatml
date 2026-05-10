# flow-graph-generator

LoRA-tuned `Qwen/Qwen3-1.7B` that converts natural-language workflow requests into Flow Graph JSON proposals matching `FlowGraphDto` in flow-studio. Pure SFT on 3-message conversations (system, user, assistant). No tool calling, no agent loop. The model emits proposals only — flow-studio validates and decides whether to execute.

Authoritative spec: [`../../docs/flow_inference_model_training_instructions.md`](../../docs/flow_inference_model_training_instructions.md).

## Targets

- **Cross-platform local inference**: Mac, Windows, Linux with 16 GB RAM minimum.
- **Final artefact**: merged safetensors loaded by `flow-model-runtime` (Candle backend in flow-studio).
- **Disk footprint**: ~3.4 GB at fp16. Comfortably under the 5 GB `.fm` archive cap and the 16 GB RAM target on every supported OS.

## Layout

```
models/flow-graph-generator/
├── README.md                          (this file)
├── model.yml                          (training/packaging config)
└── datasets/
    ├── prompt_spec.json               (system prompt + response schema + stop tokens)
    ├── flow_graph_schema.json         (FlowGraphDto JSON Schema; synced from flow-studio)
    ├── node_contracts.json            (closed vocabulary: node kinds, action triples, refusal categories)
    ├── intent_aliases.json            (semantic verb → real (type, adapter, actionId) map)
    └── samples/
        ├── seed_samples.jsonl         (hand-authored + Claude-converted training pairs)
        └── test_prompt_set.jsonl      (the 8 doc-canonical eval prompts)
```

## Workflow

```bash
flow_ml prepare  models/flow-graph-generator/         # split seeds into train/val/test
flow_ml train    models/flow-graph-generator/         # LoRA fine-tune on Qwen3-1.7B
flow_ml evaluate models/flow-graph-generator/         # 7-layer validation + 8 metrics
flow_ml package  models/flow-graph-generator/ --version v0.1
```

## Bounded vocabulary (v1)

Allowed `(type, adapter, actionId)` triples:

| type | adapter | actionId |
|---|---|---|
| action | shell | run-command, git, npm, pnpm, cargo, kubectl, curl |
| action | zowe | cli-raw |
| action | mri-toolkit | prepare |
| utility | — | sleep, log, set-variable |
| ai | — | (modelId): jcl-validator, spool-interpreter, flow-graph-generator |
| cloud_ai | — | (provider): claude, openai, gemini |

Forbidden adapters (mock placeholders, never emit): `ssh`, `zosmf`, `mock`.

Forbidden operation classes (refused with `warnings`):
- `credential.read_secret`
- `shell.exec_unrestricted`
- `external.http_post`
- `network.upload_file`

Full enumeration in [`datasets/node_contracts.json`](datasets/node_contracts.json).

## Quality gates (from §14 of the spec)

| Metric | Target |
|---|---|
| `json_parse_rate` | ≥ 0.95 |
| `schema_conformance_rate` | ≥ 0.90 |
| `node_type_validity_rate` | ≥ 0.98 |
| `edge_ref_validity_rate` | ≥ 0.98 |
| `forbidden_rejection_rate` | **= 1.00** |
| `unsafe_refusal_rate` | ≥ 0.95 |

`semantic_match_rate` and `ambiguity_handling_rate` are observational in v1 (need human / LLM-judge scoring).

## Versioning

`flow-graph-1.7b-v0.1` is the first checkpoint name. Each release records:

- Base model
- Dataset version (sample count, category distribution, paraphrase pass count)
- Training configuration
- Evaluation results against the 8 quality gates
- Known limitations
- Runtime compatibility notes (Candle + Mac/CUDA/CPU)

## Schema sync

The Flow Graph schema lives canonically in flow-studio at `apps/shared-types/src/graph.ts`. The committed [`datasets/flow_graph_schema.json`](datasets/flow_graph_schema.json) and [`datasets/node_contracts.json`](datasets/node_contracts.json) files are produced by [`scripts/sync-spec-from-flow-studio.sh`](../../scripts/sync-spec-from-flow-studio.sh) and stamped with the source SHA in `prompt_spec.json._provenance`. Drift is guarded by a CI check.
