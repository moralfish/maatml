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
| `GET /info` | model summary + packaging hints, and which of enforce / auth / capture / retries are on |
| `POST /predict` | dataset-shaped JSON row → prediction |
| `POST /predict?validate=1` | prediction **plus** the inline validator result |

```bash
maatml serve examples/support-ticket-triage/ --host 0.0.0.0 --port 8080
```

It is intentionally simple, a single model, one request at a time, which keeps
it light enough for edge / single-instance use (including Jetson/JetPack). For
higher throughput, use the vLLM path below.

### Gating live inference

`--enforce` turns the validator into a gate: a `/predict` whose output fails
validation returns **HTTP 422** instead of 200, so the same contract that gated
training and evaluation also gates production. Without `--enforce`, `?validate=1`
annotates the response but never blocks it.

```bash
maatml serve examples/support-ticket-triage/ --enforce
```

`--max-retries N` softens that: on a validation failure the server feeds the
error back to the model and re-asks, up to N times, before giving up. Every
response reports `attempts` and `retries`, and a request still failing after the
budget returns 422 with the retry count, so retries are always visible, never
silent. (Retry-with-feedback works for any predictor. Constrained decoding that
enforces the schema *during* generation is a planned follow-up: it needs a serve
extra and only applies to generative architectures.)

### Auth

`--auth-token TOKEN` (or `MAATML_SERVE_TOKEN`) requires `Authorization: Bearer
TOKEN` on `/predict`. It is compared in constant time, mandatory for `--capture`
(below), and strongly recommended for any non-loopback bind, serving on
`0.0.0.0` without it now prints a warning, because anyone who can reach the port
can query the model.

```bash
maatml serve examples/support-ticket-triage/ --host 0.0.0.0 --auth-token "$TOKEN"
```

### Capture and the reviewed flywheel

`--capture PATH` appends served predictions to a JSONL for later review. A
captured row is **not** gold: it carries `approved: false` / `needs_review:
true`, holds only the sanitized request and the model's own output, and the file
is row/byte capped so an unattended server cannot fill the disk. Capture
requires `--auth-token`.

The retrain loop is deliberate at every step:

```bash
maatml serve <model> --auth-token "$TOKEN" --capture captures.jsonl
# ... traffic accumulates in captures.jsonl ...
# review: fix the target and set "approved": true on the rows worth keeping
maatml ingest <model> --input captures.jsonl   # refuses any row not approved
maatml run <model>                              # the new seeds make prepare stale
```

`maatml ingest` refuses a `serve_capture` row unless a reviewer set `approved:
true` (dropping the flag does not sneak it through), so a raw model prediction
can never become training data without a human or teacher approving it.

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
the sha256 of each file listed in the manifest and checks it against the
recorded value, so you can catch truncated, corrupted, or partially copied
artifacts before you ship. This detects accidental corruption, not tampering:
anyone who can rewrite a file can also recompute its hash in `manifest.json`, so
treat `verify` as an integrity check, not a signature.

```bash
maatml export examples/support-ticket-triage/
maatml verify examples/support-ticket-triage/output/export/<run_id>
```
