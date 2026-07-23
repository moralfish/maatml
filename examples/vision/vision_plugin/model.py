"""MultitaskNet: MobileNetV3-Large backbone + scene / CenterNet / pose heads."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .constants import (
    DEFAULT_IMAGE_SIZE,
    DEFAULT_OUTPUT_STRIDE,
    KEYPOINT_NAMES,
    SCENE_LABELS,
    SHAPE_LABELS,
)


@dataclass
class MultitaskConfig:
    image_size: int = DEFAULT_IMAGE_SIZE
    backbone: str = "mobilenet_v3_large"
    pretrained: bool = True
    scene_labels: list[str] = field(default_factory=lambda: list(SCENE_LABELS))
    shape_labels: list[str] = field(default_factory=lambda: list(SHAPE_LABELS))
    keypoint_names: list[str] = field(default_factory=lambda: list(KEYPOINT_NAMES))
    output_stride: int = DEFAULT_OUTPUT_STRIDE
    loss_weights: dict[str, float] = field(
        default_factory=lambda: {"scene": 1.0, "detect": 1.0, "pose": 1.0}
    )
    score_thresh: float = 0.25

    @property
    def heatmap_size(self) -> int:
        return max(1, self.image_size // self.output_stride)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "MultitaskConfig":
        known = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in raw.items() if k in known})

    @classmethod
    def from_model_def(cls, model_def: Any) -> "MultitaskConfig":
        train = dict(model_def.training or {})
        heads = train.get("heads") or {}
        return cls(
            image_size=int(train.get("image_size") or DEFAULT_IMAGE_SIZE),
            backbone=str(train.get("backbone") or "mobilenet_v3_large"),
            pretrained=bool(train.get("pretrained", True)),
            scene_labels=list(heads.get("scene_labels") or SCENE_LABELS),
            shape_labels=list(heads.get("shape_labels") or SHAPE_LABELS),
            keypoint_names=list(heads.get("keypoint_names") or KEYPOINT_NAMES),
            loss_weights=dict(
                train.get("loss_weights")
                or {"scene": 1.0, "detect": 1.0, "pose": 1.0}
            ),
            score_thresh=float(train.get("score_thresh") or 0.25),
        )


def _build_backbone(name: str, pretrained: bool):
    import torchvision.models as models

    weights = None
    if pretrained:
        if name == "mobilenet_v3_large":
            weights = models.MobileNet_V3_Large_Weights.DEFAULT
        elif name == "mobilenet_v3_small":
            weights = models.MobileNet_V3_Small_Weights.DEFAULT
    if name == "mobilenet_v3_large":
        net = models.mobilenet_v3_large(weights=weights)
        feat_dim = 960
    elif name == "mobilenet_v3_small":
        net = models.mobilenet_v3_small(weights=weights)
        feat_dim = 576
    else:
        raise ValueError(f"Unsupported backbone {name!r}")
    backbone = net.features
    # Freeze BN running stats friendliness, leave trainable by default.
    return backbone, feat_dim


class MultitaskNet:
    """Thin wrapper so the module imports without torch at collection time.

    Construct via ``MultitaskNet.build(cfg)`` which returns an ``nn.Module``.
    """

    @staticmethod
    def build(cfg: MultitaskConfig):
        import torch
        import torch.nn as nn
        import torch.nn.functional as F

        class _MultitaskNet(nn.Module):
            def __init__(self, config: MultitaskConfig) -> None:
                super().__init__()
                self.config = config
                backbone, feat_dim = _build_backbone(config.backbone, config.pretrained)
                self.backbone = backbone
                self.feat_dim = feat_dim
                n_scene = len(config.scene_labels)
                n_shape = len(config.shape_labels)
                n_kpts = len(config.keypoint_names)

                # 2× upsample → finer heatmaps (stride 16 instead of 32).
                self.det_upsample = nn.Sequential(
                    nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                    nn.Conv2d(feat_dim, feat_dim, 3, padding=1),
                    nn.ReLU(inplace=True),
                )
                self.scene_head = nn.Sequential(
                    nn.AdaptiveAvgPool2d(1),
                    nn.Flatten(),
                    nn.Linear(feat_dim, n_scene),
                )
                self.det_heatmap = nn.Sequential(
                    nn.Conv2d(feat_dim, 128, 3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(128, n_shape, 1),
                )
                self.det_size = nn.Sequential(
                    nn.Conv2d(feat_dim, 128, 3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(128, 2, 1),
                )
                self.det_offset = nn.Sequential(
                    nn.Conv2d(feat_dim, 128, 3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(128, 2, 1),
                )
                self.pose_head = nn.Sequential(
                    nn.AdaptiveAvgPool2d(1),
                    nn.Flatten(),
                    nn.Linear(feat_dim, 256),
                    nn.ReLU(inplace=True),
                    nn.Linear(256, n_kpts * 2),
                )

            def forward(self, x: "torch.Tensor") -> dict[str, "torch.Tensor"]:
                feats = self.backbone(x)
                if feats.ndim != 4:
                    raise RuntimeError(f"Expected 4D features, got {tuple(feats.shape)}")
                scene_logits = self.scene_head(feats)
                det_feats = self.det_upsample(feats)
                heatmaps = self.det_heatmap(det_feats)
                sizes = self.det_size(det_feats)
                offsets = self.det_offset(det_feats)
                pose_coords = self.pose_head(feats)
                return {
                    "scene_logits": scene_logits,
                    "heatmaps": heatmaps,
                    "sizes": sizes,
                    "offsets": offsets,
                    "pose_coords": pose_coords,
                }

            def compute_loss(
                self,
                outputs: dict[str, "torch.Tensor"],
                targets: dict[str, "torch.Tensor"],
            ) -> dict[str, "torch.Tensor"]:
                w = self.config.loss_weights
                scene_loss = F.cross_entropy(outputs["scene_logits"], targets["scene_idx"])
                # Focal loss on heatmaps
                hm_loss = _focal_loss(
                    outputs["heatmaps"], targets["heatmaps"]
                )
                # Size / offset L1 only at positive centers
                mask = targets["center_mask"]  # (B,1,H,W)
                n_pos = mask.sum().clamp(min=1.0)
                size_loss = (
                    F.l1_loss(outputs["sizes"] * mask, targets["sizes"] * mask, reduction="sum")
                    / n_pos
                )
                offset_loss = (
                    F.l1_loss(
                        outputs["offsets"] * mask,
                        targets["offsets"] * mask,
                        reduction="sum",
                    )
                    / n_pos
                )
                pose_loss = F.smooth_l1_loss(outputs["pose_coords"], targets["pose_coords"])
                detect_loss = hm_loss + 0.1 * size_loss + offset_loss
                total = (
                    float(w.get("scene", 1.0)) * scene_loss
                    + float(w.get("detect", 1.0)) * detect_loss
                    + float(w.get("pose", 1.0)) * pose_loss
                )
                return {
                    "loss": total,
                    "scene_loss": scene_loss.detach(),
                    "detect_loss": detect_loss.detach(),
                    "pose_loss": pose_loss.detach(),
                }

        return _MultitaskNet(cfg)


def _focal_loss(pred: Any, gt: Any, alpha: float = 2.0, beta: float = 4.0) -> Any:
    """CornerNet / CenterNet focal loss (pred = logits)."""
    import torch

    pred_sig = torch.sigmoid(pred)
    pos = gt.eq(1).float()
    neg = gt.lt(1).float()
    neg_weights = torch.pow(1 - gt, beta)
    pos_loss = -torch.log(pred_sig.clamp(min=1e-6)) * torch.pow(1 - pred_sig, alpha) * pos
    neg_loss = (
        -torch.log((1 - pred_sig).clamp(min=1e-6))
        * torch.pow(pred_sig, alpha)
        * neg_weights
        * neg
    )
    n_pos = pos.sum().clamp(min=1.0)
    return (pos_loss.sum() + neg_loss.sum()) / n_pos


def save_checkpoint(model: Any, cfg: MultitaskConfig, out_dir: Path) -> None:
    from safetensors.torch import save_file

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    save_file(state, out_dir / "model.safetensors")
    (out_dir / "config.json").write_text(
        json.dumps(cfg.to_dict(), indent=2) + "\n", encoding="utf-8"
    )


def load_checkpoint(
    checkpoint_dir: Path,
    *,
    device: str = "cpu",
    pretrained_backbone: bool = False,
) -> tuple[Any, MultitaskConfig]:
    from safetensors.torch import load_file

    checkpoint_dir = Path(checkpoint_dir)
    cfg_path = checkpoint_dir / "config.json"
    if not cfg_path.is_file():
        raise FileNotFoundError(f"Missing config.json in {checkpoint_dir}")
    cfg = MultitaskConfig.from_dict(json.loads(cfg_path.read_text(encoding="utf-8")))
    # Don't re-download ImageNet weights when loading a fine-tuned checkpoint.
    cfg.pretrained = pretrained_backbone
    model = MultitaskNet.build(cfg)
    weights = load_file(str(checkpoint_dir / "model.safetensors"))
    model.load_state_dict(weights, strict=True)
    model.to(device)
    model.eval()
    return model, cfg
