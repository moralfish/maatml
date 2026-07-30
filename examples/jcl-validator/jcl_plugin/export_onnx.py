"""ONNX serving-bundle exporter for the multi-head JCL classifier.

Produces ``<name>.onnx/`` with model.onnx (+ external weights), tokenizer.json,
and a serving.json contract, so Flow's ONNX engine can serve the model without
torch. Heads are baked into the graph: pooled linear heads on the CLS position
and a per-token linear head for the line pointer.
"""
from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any, Optional

from maatml.config import ModelDefinition, get_dataset_cfg
from maatml.evaluation.predictors import _load_head_specs
from maatml.export.bundle import export_safetensors_bundle
from maatml.export.manifest import build_manifest, write_manifest
from maatml.registry import register_exporter
from maatml.utils.io import write_json

from .tokenizer import SPECIAL_TOKENS

SERVING_CONTRACT = "maatml.serving/1"

# Line-start marker emitted by pre_tokenize_jcl; clients map per-token line
# head hits back to deck lines by counting these.
_LINE_POINTER_MARKER = "<COL1>"

_ONNX_OPSET = 18
_ONNX_INPUTS = ["input_ids", "attention_mask"]


def build_serving_config(
    model_def: ModelDefinition, head_specs: list[dict[str, Any]]
) -> dict[str, Any]:
    """serving.json body for the ``multi_head_classifier`` serving contract.

    ``max_input_tokens`` is packaging.max_input_tokens, the same budget
    evaluate and serve enforce.
    """
    if not head_specs:
        raise ValueError(
            "No head specs found (run_metadata.json heads or training.heads)"
        )
    heads: list[dict[str, Any]] = []
    for spec in head_specs:
        head: dict[str, Any] = {
            "name": spec["name"],
            "kind": spec.get("kind", "classification"),
        }
        labels = list(spec.get("labels") or [])
        if labels:
            head["labels"] = labels
        heads.append(head)

    specials = {key: f"<{key.upper()}>" for key in ("pad", "unk", "cls", "sep")}
    missing = [tok for tok in specials.values() if tok not in SPECIAL_TOKENS]
    if missing:
        raise ValueError(f"Tokenizer SPECIAL_TOKENS missing {missing}")

    return {
        "contract": SERVING_CONTRACT,
        "kind": "multi_head_classifier",
        "max_input_tokens": model_def.packaging.max_input_tokens,
        "text_transform": get_dataset_cfg(model_def).get("text_transform"),
        "line_pointer_marker": _LINE_POINTER_MARKER,
        "special_tokens": specials,
        "heads": heads,
        "onnx": {
            "file": "model.onnx",
            "inputs": list(_ONNX_INPUTS),
            "outputs": [spec["name"] for spec in head_specs],
        },
    }


@register_exporter("onnx")
def export_onnx(
    model_def: ModelDefinition,
    checkpoint_dir: Path,
    out_dir: Path,
    *,
    run_id: Optional[str] = None,
) -> Path:
    """Export the encoder + classifier heads to ``<name>.onnx/`` + manifest."""
    import torch
    from safetensors.torch import load_file
    from transformers import AutoModel

    checkpoint_dir = Path(checkpoint_dir).resolve()
    out_dir = Path(out_dir).resolve()
    export_safetensors_bundle(model_def, checkpoint_dir, out_dir, run_id=run_id)

    head_specs = _load_head_specs(checkpoint_dir, model_def)
    serving = build_serving_config(model_def, head_specs)

    tokenizer_path = checkpoint_dir / "tokenizer.json"
    if not tokenizer_path.is_file():
        raise FileNotFoundError(f"tokenizer.json not found in {checkpoint_dir}")

    encoder = AutoModel.from_pretrained(checkpoint_dir, attn_implementation="eager")
    encoder.eval()
    head_state = load_file(checkpoint_dir / "classifier_heads.safetensors")

    class _Wrapper(torch.nn.Module):
        """Encoder + per-head linear layers, one output tensor per head."""

        def __init__(self, net: Any, specs: list[dict[str, Any]]) -> None:
            super().__init__()
            self.net = net
            self.specs = specs
            self.heads = torch.nn.ModuleDict()
            for spec in specs:
                name = spec["name"]
                weight = head_state[f"heads.{name}.weight"]
                linear = torch.nn.Linear(weight.shape[1], weight.shape[0])
                with torch.no_grad():
                    linear.weight.copy_(weight)
                    linear.bias.copy_(head_state[f"heads.{name}.bias"])
                self.heads[name] = linear

        def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
            out = self.net(input_ids=input_ids, attention_mask=attention_mask)
            seq = out.last_hidden_state
            pooled = seq[:, 0, :]
            outputs = []
            for spec in self.specs:
                head = self.heads[spec["name"]]
                # line_pointer scores every token; classification heads pool CLS.
                if spec.get("kind") == "line_pointer":
                    outputs.append(head(seq))
                else:
                    outputs.append(head(pooled))
            return tuple(outputs)

    wrapper = _Wrapper(encoder, head_specs)
    wrapper.eval()

    bundle_dir = out_dir / f"{model_def.name}.onnx"
    bundle_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = bundle_dir / "model.onnx"

    output_names = [spec["name"] for spec in head_specs]
    dynamic_axes: dict[str, dict[int, str]] = {
        name: {0: "batch", 1: "sequence"} for name in _ONNX_INPUTS
    }
    for spec in head_specs:
        axes = {0: "batch"}
        if spec.get("kind") == "line_pointer":
            axes[1] = "sequence"
        dynamic_axes[spec["name"]] = axes

    # Start clean so a re-export cannot leave a stale weights file behind.
    data_path = bundle_dir / "model.onnx.data"
    if data_path.exists():
        data_path.unlink()

    # dynamo=True: the TorchScript tracer cannot follow the transformers mask
    # construction; the torch.export path can, and it writes the weights as
    # model.onnx.data alongside the graph.
    dummy_ids = torch.zeros(2, 16, dtype=torch.long)
    dummy_mask = torch.ones(2, 16, dtype=torch.long)
    torch.onnx.export(
        wrapper,
        (dummy_ids, dummy_mask),
        str(onnx_path),
        input_names=list(_ONNX_INPUTS),
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        opset_version=_ONNX_OPSET,
        dynamo=True,
        external_data=True,
    )

    shutil.copy2(tokenizer_path, bundle_dir / "tokenizer.json")
    write_json(bundle_dir / "serving.json", serving, sort_keys=False)

    # Smoke-check with onnxruntime when available.
    try:
        import numpy as np
        import onnxruntime as ort

        sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        sess.run(
            None,
            {
                "input_ids": np.zeros((1, 8), dtype=np.int64),
                "attention_mask": np.ones((1, 8), dtype=np.int64),
            },
        )
    except ImportError:
        pass

    files = [p for p in out_dir.rglob("*") if p.is_file() and p.name != "manifest.json"]
    manifest = build_manifest(
        model_def=model_def,
        export_dir=out_dir,
        files=files,
        formats=["safetensors", "onnx"],
        source_checkpoint=checkpoint_dir,
        run_id=run_id,
    )
    write_manifest(out_dir, manifest)
    return out_dir
