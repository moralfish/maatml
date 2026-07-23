# Vision VLM

Small vision-language model ([SmolVLM-256M-Instruct](https://huggingface.co/HuggingFaceTB/SmolVLM-256M-Instruct))
fine-tuned to look at a synthetic scene and emit one short factual description:

```json
{"description": "A checkerboard background with two circles and a star; the figure stands with arms raised and a wide stance."}
```

The safetensors export is an HF-format directory; **vLLM loads it directly**
(`vllm serve <export-dir>`). Locally (Mac/CPU) use `maatml serve` with the
transformers backend; on Linux/Jetson use the vLLM deploy kit.

## Layout

```
examples/vision-vlm/
├── model.yml
├── vlm_plugin/               # trainer, predictor, metrics, synth, describe()
├── datasets/
│   ├── schema.json
│   ├── prompt_spec.json
│   └── samples/              # seed_samples.jsonl + images/ + benchmark
└── scripts/
    ├── build_seeds.py
    ├── serve_vllm.sh         # Linux / Jetson container
    └── client_openai.py      # stdlib client (maatml serve or vLLM)
```

## Lifecycle

```bash
pip install -e ".[dev,ml,vision]"   # training stack (vllm extra is Linux-only)

# Optional: grow the corpus (starter 16-row set is committed)
python examples/vision-vlm/scripts/build_seeds.py --target 300

maatml validate examples/vision-vlm
maatml prepare examples/vision-vlm
maatml train examples/vision-vlm --smoke --device cpu
maatml train examples/vision-vlm --device cpu
maatml evaluate examples/vision-vlm --gate
maatml export examples/vision-vlm
maatml verify examples/vision-vlm/output/export/<run_id>
```

## Serve locally (transformers)

```bash
maatml serve examples/vision-vlm --checkpoint output/export/<run_id> --host 0.0.0.0 --port 8080
python examples/vision-vlm/scripts/client_openai.py path/to.png --maatml http://127.0.0.1:8080/predict
```

## Serve with vLLM (Linux / Jetson)

```bash
# Linux host with CUDA
pip install "maatml[vllm]"
./examples/vision-vlm/scripts/serve_vllm.sh examples/vision-vlm/output/export/<run_id>

# Jetson Orin (container: recommended)
USE_JETSON_CONTAINER=1 ./examples/vision-vlm/scripts/serve_vllm.sh \
  examples/vision-vlm/output/export/<run_id>

# Client
python examples/vision-vlm/scripts/client_openai.py path/to.png \
  --vllm http://127.0.0.1:8000 --model vision-vlm
```

### Evaluate against a live vLLM endpoint

```bash
export MAATML_VLLM_ENDPOINT=http://127.0.0.1:8000
maatml evaluate examples/vision-vlm --gate
```

The predictor switches to the OpenAI-compatible chat-completions API when
`MAATML_VLLM_ENDPOINT` is set (image sent as a base64 data URL).

## Quality gates

| Metric | Gate (default) | Meaning |
|---|---|---|
| `scene_mention_rate` | ≥ 0.5 | background style word appears |
| `shape_mention_f1` | ≥ 0.3 | mentioned shape types vs gt |
| `brevity_rate` | ≥ 0.8 | ≤ 40 words, single line |
| `all_layers_pass_rate` | ≥ 0.7 | validator layers pass |

Raise gates after a longer train on a larger corpus (`--target 300+`).
