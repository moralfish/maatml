# Serving & deployment

A trained MaatML model has three deployment paths, from lightweight-local to
production throughput. All of them serve the *same* exported checkpoint, and all
can re-apply the model's [validator](lifecycle.md).

## 1. `maatml serve`, built-in HTTP API

A dependency-light server (Python stdlib, no FastAPI/uvicorn) that loads the
predictor once and exposes:

| Route | Purpose |
|-------|---------|
| `GET /health` | liveness + identity |
| `GET /info` | model summary + packaging hints |
| `POST /predict` | dataset-shaped JSON row → prediction |
| `POST /predict?validate=1` | prediction **plus** the inline validator result |

```bash
maatml serve examples/support-ticket-triage/ --host 0.0.0.0 --port 8080
```

It is intentionally simple, a single model, one request at a time, which keeps
it light enough for edge / single-instance use (including Jetson/JetPack). For
higher throughput, use the vLLM path below.

## 2. ONNX / edge (vision)

`vision_multitask` exports to **ONNX** for `onnxruntime` (CPU on Mac) or the
TensorRT execution provider on Jetson:

```bash
maatml export examples/vision/ --format onnx
```

## 3. vLLM, vision-language models

VLM checkpoints (`vlm_sft`) export as an HF-format directory that **vLLM loads
directly** for OpenAI-compatible, higher-throughput serving:

```bash
vllm serve examples/vision-vlm/output/export/<run_id>
```

The [`vision-vlm` example](https://github.com/moralfish/maatml/tree/main/examples/vision-vlm)
ships `serve_vllm.sh` (Linux / Jetson container) and an OpenAI-compatible client.
Its evaluator can even score against a live vLLM endpoint, set
`MAATML_VLLM_ENDPOINT` and the predictor switches to the chat-completions API
(the image is sent as a base64 data URL).

```bash
pip install "maatml[vllm]"     # Linux-only
export MAATML_VLLM_ENDPOINT=http://127.0.0.1:8000
maatml evaluate examples/vision-vlm --gate
```

## Verifying an export

Every export writes a `manifest.json`. `maatml verify <export-dir>` recomputes
the sha256 of each listed file, so you can confirm an artifact is intact, and
unchanged since export, before you ship it.

```bash
maatml export examples/support-ticket-triage/
maatml verify examples/support-ticket-triage/output/export/<run_id>
```
