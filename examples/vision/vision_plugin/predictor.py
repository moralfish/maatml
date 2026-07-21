"""Dual-backend predictor: torch checkpoint or ``model.onnx`` via onnxruntime."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

from .dataset import image_bytes_to_tensor, resolve_image_bytes_or_path
from .decode import decode_multitask_outputs
from .model import MultitaskConfig, load_checkpoint


class VisionMultitaskPredictor:
    """Predict multitask JSON for a dataset-shaped row with an ``image`` field."""

    def __init__(self) -> None:
        self.model = None
        self.cfg: MultitaskConfig | None = None
        self.session = None
        self.device = "cpu"
        self.model_dir: Path | None = None
        self.checkpoint_dir: Path | None = None
        self.backend: str = "none"

    def setup(
        self,
        checkpoint_dir: Path,
        *,
        model_def: Any = None,
        device: Any = "cpu",
        max_input_tokens: Optional[int] = None,
        schema_path: Optional[Path] = None,
        contracts_path: Optional[Path] = None,
        prompt_spec_path: Optional[Path] = None,
    ) -> None:
        del max_input_tokens, schema_path, contracts_path, prompt_spec_path
        self.checkpoint_dir = Path(checkpoint_dir)
        self.model_dir = Path(model_def.model_dir) if model_def is not None else None
        self.device = str(device)

        onnx_path = self.checkpoint_dir / "model.onnx"
        if onnx_path.is_file():
            self._setup_onnx(onnx_path)
            return
        self._setup_torch()

    def _setup_torch(self) -> None:
        assert self.checkpoint_dir is not None
        self.model, self.cfg = load_checkpoint(
            self.checkpoint_dir, device=self.device, pretrained_backbone=False
        )
        self.backend = "torch"

    def _setup_onnx(self, onnx_path: Path) -> None:
        import onnxruntime as ort

        cfg_path = onnx_path.parent / "config.json"
        if cfg_path.is_file():
            self.cfg = MultitaskConfig.from_dict(
                json.loads(cfg_path.read_text(encoding="utf-8"))
            )
        else:
            self.cfg = MultitaskConfig()

        providers: list[str] = []
        available = ort.get_available_providers()
        for p in ("TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"):
            if p in available:
                providers.append(p)
        if not providers:
            providers = ["CPUExecutionProvider"]
        self.session = ort.InferenceSession(str(onnx_path), providers=providers)
        self.backend = "onnx"

    def predict(self, row: dict[str, Any]) -> str:
        if self.cfg is None:
            raise RuntimeError("Predictor.setup() was not called")
        image_val = row.get("image")
        if image_val is None:
            raise KeyError("row missing 'image' field")
        data = resolve_image_bytes_or_path(image_val, model_dir=self.model_dir)
        tensor = image_bytes_to_tensor(data, self.cfg.image_size)

        if self.backend == "onnx":
            outputs = self._infer_onnx(tensor)
        elif self.backend == "torch":
            outputs = self._infer_torch(tensor)
        else:
            raise RuntimeError(f"Unknown backend {self.backend!r}")

        decoded = decode_multitask_outputs(
            scene_logits=outputs["scene_logits"],
            heatmaps=outputs["heatmaps"],
            sizes=outputs["sizes"],
            offsets=outputs["offsets"],
            pose_coords=outputs["pose_coords"],
            scene_labels=self.cfg.scene_labels,
            shape_labels=self.cfg.shape_labels,
            keypoint_names=self.cfg.keypoint_names,
            score_thresh=self.cfg.score_thresh,
        )
        return json.dumps(decoded, ensure_ascii=False)

    def _infer_torch(self, tensor: Any) -> dict[str, Any]:
        import torch

        assert self.model is not None
        x = tensor.unsqueeze(0).to(self.device)
        with torch.no_grad():
            out = self.model(x)
        return {
            "scene_logits": out["scene_logits"][0].detach().cpu().numpy(),
            "heatmaps": out["heatmaps"][0].detach().cpu().numpy(),
            "sizes": out["sizes"][0].detach().cpu().numpy(),
            "offsets": out["offsets"][0].detach().cpu().numpy(),
            "pose_coords": out["pose_coords"][0].detach().cpu().numpy(),
        }

    def _infer_onnx(self, tensor: Any) -> dict[str, Any]:
        import numpy as np

        assert self.session is not None
        x = tensor.unsqueeze(0).numpy().astype(np.float32)
        input_name = self.session.get_inputs()[0].name
        outs = self.session.run(None, {input_name: x})
        names = [o.name for o in self.session.get_outputs()]
        by_name = {n: v[0] for n, v in zip(names, outs)}
        # Fallback positional if names differ.
        if "scene_logits" not in by_name and len(outs) >= 5:
            by_name = {
                "scene_logits": outs[0][0],
                "heatmaps": outs[1][0],
                "sizes": outs[2][0],
                "offsets": outs[3][0],
                "pose_coords": outs[4][0],
            }
        return by_name
