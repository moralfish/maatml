#!/usr/bin/env python3
"""Chain the vision server and the vision-describer server.

Requires two running ``maatml serve`` processes:

    maatml serve examples/vision --port 8080
    maatml serve examples/vision-describer --port 8081

Usage:
    python examples/vision-describer/scripts/compose_client.py path/to/image.png
    python examples/vision-describer/scripts/compose_client.py path/to/image.png \\
        --vision-url http://127.0.0.1:8080 --describer-url http://127.0.0.1:8081
"""
from __future__ import annotations

import argparse
import base64
import json
import mimetypes
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

EXAMPLE_ROOT = Path(__file__).resolve().parents[1]
REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(EXAMPLE_ROOT))

from vision_describer_plugin.linearize import linearize_vision_result  # noqa: E402


def _post_json(url: str, payload: dict[str, Any], timeout: float = 60.0) -> dict[str, Any]:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code} from {url}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Failed to reach {url}: {exc}") from exc
    parsed = json.loads(body)
    if not isinstance(parsed, dict):
        raise RuntimeError(f"Expected JSON object from {url}, got {type(parsed)}")
    return parsed


def _image_to_data_uri(path: Path) -> str:
    mime, _ = mimetypes.guess_type(str(path))
    mime = mime or "image/png"
    b64 = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{b64}"


def compose(
    image: str | Path,
    *,
    vision_url: str = "http://127.0.0.1:8080",
    describer_url: str = "http://127.0.0.1:8081",
    validate: bool = False,
) -> dict[str, Any]:
    """Run vision → linearize → describer and return a combined payload."""
    image_path = Path(image)
    if image_path.is_file():
        # Prefer a model-relative path when the image lives under examples/vision;
        # otherwise send a data-URI so the vision server can decode it.
        try:
            rel = image_path.resolve().relative_to(
                (REPO / "examples" / "vision").resolve()
            )
            vision_row: dict[str, Any] = {"image": str(rel)}
        except ValueError:
            vision_row = {"image": _image_to_data_uri(image_path)}
    else:
        # Already a path/URI string the vision server understands.
        vision_row = {"image": str(image)}

    q = "?validate=1" if validate else ""
    vision_resp = _post_json(f"{vision_url.rstrip('/')}/predict{q}", vision_row)
    vision_out = vision_resp.get("output")
    if vision_out is None and isinstance(vision_resp.get("raw"), str):
        vision_out = json.loads(vision_resp["raw"])
    if not isinstance(vision_out, dict):
        raise RuntimeError(f"Vision server did not return a JSON object: {vision_resp}")

    request = linearize_vision_result(vision_out)
    describer_resp = _post_json(
        f"{describer_url.rstrip('/')}/predict{q}",
        {"request": request},
    )
    desc_out = describer_resp.get("output")
    if isinstance(desc_out, dict) and "description" in desc_out:
        description = desc_out["description"]
    elif isinstance(describer_resp.get("raw"), str):
        try:
            description = json.loads(describer_resp["raw"]).get("description")
        except json.JSONDecodeError:
            description = describer_resp["raw"]
    else:
        description = desc_out

    return {
        "vision": vision_out,
        "description": description,
        "request": request,
        "latency_ms": {
            "vision": vision_resp.get("latency_ms"),
            "describer": describer_resp.get("latency_ms"),
        },
        "vision_valid": vision_resp.get("valid"),
        "describer_valid": describer_resp.get("valid"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", help="Image path (or vision-server image field value)")
    parser.add_argument("--vision-url", default="http://127.0.0.1:8080")
    parser.add_argument("--describer-url", default="http://127.0.0.1:8081")
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Ask both servers to run their configured validators",
    )
    args = parser.parse_args()
    result = compose(
        args.image,
        vision_url=args.vision_url,
        describer_url=args.describer_url,
        validate=args.validate,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
