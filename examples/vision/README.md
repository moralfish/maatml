# Vision multitask

One MobileNetV3-Large checkpoint that jointly predicts:

- **scene** classification (background style)
- **object detection** (colored shapes — CenterNet-style, NMS-free ONNX)
- **pose** estimation (single stick figure, 12 keypoints)

Trained on deterministic synthetic scenes (PIL) — fully offline, no Hub downloads.
Deploy with `maatml export --format onnx` and `maatml serve`.

## Output shape

```json
{
  "scene": {"label": "striped", "confidence": 0.94},
  "detections": [
    {"label": "circle", "box": [0.1, 0.2, 0.3, 0.4], "confidence": 0.88}
  ],
  "pose": {
    "keypoints": [
      {"name": "head", "x": 0.5, "y": 0.2, "confidence": 1.0}
    ]
  }
}
```

## Layout

```
examples/vision/
├── model.yml
├── vision_plugin/          # trainer, predictor, metrics, ONNX exporter, synth
├── datasets/
│   ├── schema.json
│   └── samples/            # seed_samples.jsonl + images/ + benchmark
├── scripts/build_seeds.py
└── tests/
```

## Lifecycle

```bash
# Install (once)
pip install -e ".[dev,ml,vision]"

# Optional: grow the corpus (starter 16-row set is committed)
python examples/vision/scripts/build_seeds.py --target 2000
# or: maatml datagen examples/vision --target 500

maatml validate examples/vision
maatml prepare examples/vision
maatml train examples/vision --smoke --device mps   # or cpu|cuda
maatml train examples/vision --device mps
maatml evaluate examples/vision --gate
maatml export examples/vision --format onnx --parity
maatml verify examples/vision/output/export/<run_id>
```

## Serve (makes "→ deploy" real)

Same command on a Mac (onnxruntime CPU) and a Jetson (TensorRT/CUDA EP):

```bash
maatml serve examples/vision --checkpoint output/export/<run_id> --host 0.0.0.0 --port 8080

# health / info
curl -s localhost:8080/health | jq
curl -s localhost:8080/info | jq

# predict (path relative to model dir, or base64 / data-URI)
python examples/vision/output/export/<run_id>/deploy/client.py \
  examples/vision/datasets/samples/images/<sample>.png
```

Endpoints: `GET /health`, `GET /info`, `POST /predict` (`?validate=1` optional).

## Deploy to Jetson

1. Copy the export directory to the device.
2. Install NVIDIA's `onnxruntime-gpu` wheel for your JetPack.
3. Run `maatml serve … --checkpoint <export-dir>` — providers prefer TensorRT → CUDA → CPU.
4. Optional power-user path: `./deploy/build_engine.sh` (`trtexec --fp16`) for a standalone engine.

Int8 calibration is out of scope for this example.

## Bring real data

Point `dataset.seed_samples` at your own JSONL (same schema: `image` + `expected`)
or use `maatml ingest examples/vision --input PATH --map …`. Prefer a small COCO
subset via ingest rather than streaming 20 GB in core.

## Quality gates

| Metric | Gate (default) | Meaning |
|---|---|---|
| `scene_accuracy` | ≥ 0.85 | background-style classification |
| `map_50` | ≥ 0.15 | VOC-style mAP @ IoU 0.5 |
| `pck_0_2` | ≥ 0.15 | pose PCK @ 0.2 × person diagonal |

Raise the gates after a longer train on a larger corpus (`--target 2000`).
