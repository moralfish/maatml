#!/usr/bin/env bash
# Serve a vision-vlm export directory with vLLM (Linux / Jetson).
#
# Usage:
#   ./scripts/serve_vllm.sh /path/to/export [--port 8000]
#
# On Jetson Orin, set USE_JETSON_CONTAINER=1 to wrap the command in the
# nvidia-ai-iot vLLM container.
set -euo pipefail

EXPORT_DIR="${1:-}"
shift || true
PORT=8000
HOST=0.0.0.0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --port) PORT="$2"; shift 2 ;;
    --host) HOST="$2"; shift 2 ;;
    *) echo "Unknown arg: $1" >&2; exit 2 ;;
  esac
done

if [[ -z "$EXPORT_DIR" || ! -d "$EXPORT_DIR" ]]; then
  echo "Usage: $0 <export-dir> [--port N] [--host HOST]" >&2
  exit 2
fi
EXPORT_DIR="$(cd "$EXPORT_DIR" && pwd)"

if [[ "${USE_JETSON_CONTAINER:-0}" == "1" ]]; then
  exec docker run --rm -it --runtime nvidia --network host \
    --shm-size=8g \
    -v "$EXPORT_DIR":/model:ro \
    -v "${HOME}/.cache/huggingface:/root/.cache/huggingface" \
    ghcr.io/nvidia-ai-iot/vllm:latest-jetson-orin \
    vllm serve /model --host "$HOST" --port "$PORT" --trust-remote-code
fi

if ! command -v vllm >/dev/null 2>&1; then
  echo "vllm not found. On Linux: pip install 'maatml[vllm]'" >&2
  echo "On Jetson: USE_JETSON_CONTAINER=1 $0 $EXPORT_DIR" >&2
  exit 1
fi

exec vllm serve "$EXPORT_DIR" --host "$HOST" --port "$PORT" --trust-remote-code
