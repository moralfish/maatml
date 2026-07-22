#!/usr/bin/env python3
"""Stdlib client for vision-vlm: posts an image to vLLM or maatml serve.

Examples:
  # maatml serve (JSON /predict)
  python scripts/client_openai.py image.png --maatml http://127.0.0.1:8080/predict

  # vLLM OpenAI-compatible chat completions
  python scripts/client_openai.py image.png --vllm http://127.0.0.1:8000 \\
      --model vision-vlm
"""
from __future__ import annotations

import argparse
import base64
import json
import sys
import urllib.request
from pathlib import Path


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("image", type=Path)
    p.add_argument("--maatml", help="maatml serve /predict URL")
    p.add_argument("--vllm", help="vLLM base URL (e.g. http://127.0.0.1:8000)")
    p.add_argument("--model", default="vision-vlm")
    p.add_argument("--validate", action="store_true")
    p.add_argument(
        "--prompt",
        default=(
            "Describe this synthetic scene in one short factual sentence covering "
            "the background style, any colored shapes, and the stick figure's pose."
        ),
    )
    args = p.parse_args()
    if not args.image.is_file():
        print(f"image not found: {args.image}", file=sys.stderr)
        return 1
    if not args.maatml and not args.vllm:
        print("pass --maatml URL or --vllm URL", file=sys.stderr)
        return 2

    b64 = base64.b64encode(args.image.read_bytes()).decode("ascii")
    data_url = f"data:image/png;base64,{b64}"

    if args.maatml:
        url = args.maatml + ("?validate=1" if args.validate else "")
        body = {"image": data_url}
    else:
        url = args.vllm.rstrip("/") + "/v1/chat/completions"
        body = {
            "model": args.model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": args.prompt},
                        {"type": "image_url", "image_url": {"url": data_url}},
                    ],
                }
            ],
            "max_tokens": 64,
            "temperature": 0,
        }

    req = urllib.request.Request(
        url,
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        result = json.loads(resp.read().decode("utf-8"))
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
