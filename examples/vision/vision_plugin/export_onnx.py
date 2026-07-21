"""ONNX exporter + Jetson deploy kit for ``vision_multitask``."""
from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any, Optional

from maatml.config import ModelDefinition, get_dataset_cfg
from maatml.export.manifest import build_manifest, write_manifest
from maatml.registry import register_exporter

from .model import load_checkpoint

_CLIENT_PY = '''\
#!/usr/bin/env python3
"""POST an image to ``maatml serve`` and pretty-print the multitask result.

Stdlib only — works on Mac and Jetson without extra deps.
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
    p.add_argument("image", type=Path, help="Path to an image file")
    p.add_argument("--url", default="http://127.0.0.1:8080/predict")
    p.add_argument("--validate", action="store_true")
    args = p.parse_args()
    if not args.image.is_file():
        print(f"image not found: {args.image}", file=sys.stderr)
        return 1
    b64 = base64.b64encode(args.image.read_bytes()).decode("ascii")
    payload = {"image": f"data:image/png;base64,{b64}"}
    url = args.url + ("?validate=1" if args.validate else "")
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        result = json.loads(resp.read().decode("utf-8"))
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
'''

_BUILD_ENGINE_SH = '''\
#!/usr/bin/env bash
# Optional power-user path: build a TensorRT engine with trtexec.
# Prefer ``maatml serve`` with onnxruntime TensorRT EP for most deployments.
set -euo pipefail
DIR="$(cd "$(dirname "$0")/.." && pwd)"
ONNX="$DIR/model.onnx"
OUT="$DIR/deploy/model.fp16.engine"
mkdir -p "$DIR/deploy"
if ! command -v trtexec >/dev/null 2>&1; then
  echo "trtexec not found. Install TensorRT / JetPack tools, or just run:"
  echo "  maatml serve <model-dir> --checkpoint $DIR"
  exit 1
fi
trtexec --onnx="$ONNX" --fp16 --saveEngine="$OUT"
echo "wrote $OUT"
'''

_DEPLOY_README = '''\
# Deploy kit

## Serve (recommended)

On the training host (CPU/MPS) or a Jetson with onnxruntime GPU/TensorRT EP:

```bash
maatml serve /path/to/model --checkpoint /path/to/this/export --host 0.0.0.0 --port 8080
python deploy/client.py path/to/image.png --url http://HOST:8080/predict
```

## Jetson notes

1. Install NVIDIA's onnxruntime-gpu wheel matching your JetPack.
2. `maatml serve` will prefer TensorRT → CUDA → CPU execution providers.
3. Optional: `./deploy/build_engine.sh` builds a standalone fp16 engine via `trtexec`
   for offline pipelines (not required for serve).

Int8 calibration is out of scope for this example.
'''


def _copy_sidecars(model_def: ModelDefinition, out_dir: Path) -> list[Path]:
    cfg = get_dataset_cfg(model_def)
    copied: list[Path] = []
    for key in ("schema", "contracts", "prompt_spec"):
        rel = cfg.get(key)
        if not isinstance(rel, str):
            continue
        src = model_def.resolve(rel)
        if src.is_file():
            dest = out_dir / src.name
            shutil.copy2(src, dest)
            copied.append(dest)
    return copied


@register_exporter("onnx")
def export_onnx(
    model_def: ModelDefinition,
    checkpoint_dir: Path,
    out_dir: Path,
    *,
    run_id: Optional[str] = None,
) -> Path:
    """Export MultitaskNet to ``model.onnx`` + deploy kit + manifest.json."""
    import torch

    checkpoint_dir = Path(checkpoint_dir).resolve()
    out_dir = Path(out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    model, cfg = load_checkpoint(
        checkpoint_dir, device="cpu", pretrained_backbone=False
    )
    model.eval()

    # Persist config alongside ONNX for predictor reload.
    (out_dir / "config.json").write_text(
        json.dumps(cfg.to_dict(), indent=2) + "\n", encoding="utf-8"
    )
    # Also copy safetensors so safetensors consumers / parity torch path work.
    src_weights = checkpoint_dir / "model.safetensors"
    if src_weights.is_file():
        shutil.copy2(src_weights, out_dir / "model.safetensors")

    class _Wrapper(torch.nn.Module):
        def __init__(self, net: Any) -> None:
            super().__init__()
            self.net = net

        def forward(self, x: torch.Tensor):
            out = self.net(x)
            return (
                out["scene_logits"],
                out["heatmaps"],
                out["sizes"],
                out["offsets"],
                out["pose_coords"],
            )

    wrapper = _Wrapper(model)
    dummy = torch.zeros(1, 3, cfg.image_size, cfg.image_size)
    onnx_path = out_dir / "model.onnx"
    torch.onnx.export(
        wrapper,
        dummy,
        str(onnx_path),
        input_names=["image"],
        output_names=[
            "scene_logits",
            "heatmaps",
            "sizes",
            "offsets",
            "pose_coords",
        ],
        opset_version=17,
        dynamo=False,
    )

    # Smoke-check with onnxruntime when available.
    try:
        import onnxruntime as ort
        import numpy as np

        sess = ort.InferenceSession(
            str(onnx_path), providers=["CPUExecutionProvider"]
        )
        sess.run(None, {"image": dummy.numpy().astype(np.float32)})
    except ImportError:
        pass

    files = [onnx_path, out_dir / "config.json"]
    if (out_dir / "model.safetensors").is_file():
        files.append(out_dir / "model.safetensors")
    files.extend(_copy_sidecars(model_def, out_dir))

    deploy = out_dir / "deploy"
    deploy.mkdir(parents=True, exist_ok=True)
    client = deploy / "client.py"
    client.write_text(_CLIENT_PY, encoding="utf-8")
    client.chmod(client.stat().st_mode | 0o111)
    engine = deploy / "build_engine.sh"
    engine.write_text(_BUILD_ENGINE_SH, encoding="utf-8")
    engine.chmod(engine.stat().st_mode | 0o111)
    (deploy / "README.md").write_text(_DEPLOY_README, encoding="utf-8")
    files.extend([client, engine, deploy / "README.md"])

    # Also copy decode.py for reference (optional offline tooling).
    decode_src = Path(__file__).resolve().parent / "decode.py"
    if decode_src.is_file():
        dest = deploy / "decode.py"
        shutil.copy2(decode_src, dest)
        files.append(dest)

    unique: list[Path] = []
    seen: set[str] = set()
    for f in files:
        key = f.resolve().as_posix()
        if key in seen or not f.is_file():
            continue
        seen.add(key)
        unique.append(f)

    manifest = build_manifest(
        model_def=model_def,
        export_dir=out_dir,
        files=unique,
        formats=["onnx"],
        source_checkpoint=checkpoint_dir,
        run_id=run_id,
    )
    write_manifest(out_dir, manifest)
    return out_dir
