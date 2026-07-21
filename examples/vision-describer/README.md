# Vision Result Describer

Small **flan-t5-small** seq2seq model that reads a structured
[`examples/vision`](../vision/) multitask response and emits one short
factual description:

```json
{"description": "A striped scene contains two circles and a square, with the centered figure raising both arms."}
```

The model is **standalone** — train, evaluate, export, and serve it on its
own. A tiny stdlib client chains it with a running vision server.

## Output shape

```json
{ "description": "<≤30-word sentence>" }
```

Input is linearized vision JSON (low-confidence detections dropped, floats
rounded) so train-time rows match serve-time tokens.

## Layout

```
examples/vision-describer/
├── model.yml
├── vision_describer_plugin/   # linearize, describe, validator, metrics, generator
├── datasets/
│   ├── schema.json
│   ├── node_contracts.json
│   ├── prompt_spec.json
│   └── samples/               # seed + benchmark JSONL
├── scripts/
│   ├── build_seeds.py
│   └── compose_client.py      # vision → describer orchestration
└── tests/
```

## Lifecycle

```bash
pip install -e ".[dev,ml]"

# Optional: regenerate / grow the corpus
python examples/vision-describer/scripts/build_seeds.py --target 400

maatml validate examples/vision-describer
maatml prepare examples/vision-describer
maatml train examples/vision-describer --smoke --device mps   # or cpu|cuda
maatml train examples/vision-describer --device mps
maatml evaluate examples/vision-describer --gate
maatml export examples/vision-describer --format safetensors
maatml verify examples/vision-describer/output/export/<run_id>
```

## Compose with the vision model

Terminal A — vision (ONNX or torch checkpoint):

```bash
maatml serve examples/vision --port 8080
```

Terminal B — describer:

```bash
maatml serve examples/vision-describer --port 8081
```

Then:

```bash
python examples/vision-describer/scripts/compose_client.py \
  examples/vision/datasets/samples/images/syn-striped-0002-c601.png
```

The client POSTs the image to vision `/predict`, linearizes the structured
result with the same function used for training seeds, POSTs
`{"request": ...}` to the describer, and prints:

```json
{
  "vision": { "scene": ..., "detections": ..., "pose": ... },
  "description": "A striped scene contains no shapes, with the centered figure ...",
  "latency_ms": { "vision": 12.3, "describer": 45.6 }
}
```

## Quality gates

| Metric | Gate | Meaning |
|---|---|---|
| `json_parse_rate` | ≥ 0.98 | valid JSON object |
| `schema_conformance_rate` | ≥ 0.95 | matches `datasets/schema.json` |
| `conciseness_rate` | ≥ 0.90 | ≤ 30 words |
| `scene_grounding_rate` | ≥ 0.80 | mentions the scene label |
| `object_grounding_rate` | ≥ 0.70 | reflects detection labels / “no shapes” |

Eval gates cover the describer alone (pre-baked linearized JSON). End-to-end
vision→text quality is exercised via `compose_client.py` after both models
are trained.
