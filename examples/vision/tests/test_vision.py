"""Vision example tests — pure-python by default; torch/PIL optional."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


@pytest.fixture(scope="module")
def plugin():
    from vision_plugin import (  # noqa: F401
        compute_vision_metrics,
        validate_vision_scene,
    )
    return True


def test_validate_model_dir() -> None:
    from maatml.scaffold import validate_model_dir

    errors = validate_model_dir(ROOT)
    assert errors == [], errors


def test_validator_accepts_expected_payload(plugin) -> None:
    from vision_plugin.validator import validate_vision_scene

    seed = ROOT / "datasets" / "samples" / "seed_samples.jsonl"
    row = json.loads(seed.read_text(encoding="utf-8").splitlines()[0])
    result = validate_vision_scene(
        json.dumps(row["expected"]),
        schema_path=ROOT / "datasets" / "schema.json",
    )
    assert result.ok, result.errors


def test_validator_rejects_bad_json(plugin) -> None:
    from vision_plugin.validator import validate_vision_scene

    result = validate_vision_scene("not-json")
    assert not result.ok
    assert any(e.code == "invalid_json" for e in result.errors)


def test_metrics_perfect_match(plugin) -> None:
    from vision_plugin.metrics import compute_vision_metrics
    from maatml.validation.base import ValidationResult

    seed = ROOT / "datasets" / "samples" / "seed_samples.jsonl"
    row = json.loads(seed.read_text(encoding="utf-8").splitlines()[0])
    expected = row["expected"]

    class _Item:
        def __init__(self):
            self.row = row
            self.gen_text = json.dumps(expected)
            self.result = ValidationResult(
                raw_output=self.gen_text, n_layers=4, required_layers={1, 2, 3, 4}
            )
            self.result.passed_layers = {1, 2, 3, 4}

    metrics = compute_vision_metrics([_Item()])
    assert metrics["scene_accuracy"] == 1.0
    assert metrics["pck_0_2"] == 1.0
    assert metrics["map_50"] == 1.0


def test_decode_scene_argmax(plugin) -> None:
    from vision_plugin.decode import decode_scene
    from vision_plugin.constants import SCENE_LABELS

    out = decode_scene([0.1, 5.0, 0.2, 0.0, -1.0], SCENE_LABELS)
    assert out["label"] == SCENE_LABELS[1]
    assert out["confidence"] > 0.5


def test_synth_deterministic(plugin) -> None:
    pytest.importorskip("PIL")
    from vision_plugin.synth import make_scene_spec, render_scene

    a = make_scene_spec(3, base_seed=42, size=64)
    b = make_scene_spec(3, base_seed=42, size=64)
    assert a == b
    img1, exp1 = render_scene(a, size=64)
    img2, exp2 = render_scene(b, size=64)
    assert exp1 == exp2
    assert list(img1.getdata()) == list(img2.getdata())


def test_plugin_registers_trainer_and_onnx(plugin) -> None:
    from maatml.registry import TRAINERS, EXPORTERS, PREDICTORS, GENERATORS

    assert "vision_multitask" in TRAINERS.names()
    assert "vision_multitask" in PREDICTORS.names()
    assert "synthetic_scenes" in GENERATORS.names()
    assert "onnx" in EXPORTERS.names()


@pytest.mark.skipif(
    __import__("importlib").util.find_spec("torch") is None,
    reason="torch required",
)
def test_tiny_train_predict_onnx_roundtrip(tmp_path: Path, plugin) -> None:
    pytest.importorskip("torch")
    pytest.importorskip("torchvision")
    pytest.importorskip("PIL")
    pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")
    pytest.importorskip("safetensors")

    from vision_plugin.model import MultitaskConfig, MultitaskNet, save_checkpoint
    from vision_plugin.synth import build_sample_row
    from vision_plugin.dataset import VisionSceneDataset, collate_vision
    from vision_plugin.predictor import VisionMultitaskPredictor
    from vision_plugin.export_onnx import export_onnx
    from maatml.config import ModelDefinition
    import torch
    from torch.utils.data import DataLoader

    # Tiny synthetic corpus
    images = tmp_path / "images"
    images.mkdir()
    rows = [
        build_sample_row(
            i,
            base_seed=0,
            size=64,
            image_rel="images/{id}.png",
            images_dir=images,
        )
        for i in range(8)
    ]
    for r in rows:
        r["image"] = str(Path("images") / Path(r["image"]).name)

    cfg = MultitaskConfig(
        image_size=64,
        backbone="mobilenet_v3_small",
        pretrained=False,
        output_stride=16,
    )
    model = MultitaskNet.build(cfg)
    ds = VisionSceneDataset.build(rows, model_dir=tmp_path, cfg=cfg)
    loader = DataLoader(ds, batch_size=2, collate_fn=collate_vision)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    model.train()
    for step, batch in enumerate(loader):
        if step >= 2:
            break
        out = model(batch["image"])
        losses = model.compute_loss(
            out,
            {
                "scene_idx": batch["scene_idx"],
                "heatmaps": batch["heatmaps"],
                "sizes": batch["sizes"],
                "offsets": batch["offsets"],
                "center_mask": batch["center_mask"],
                "pose_coords": batch["pose_coords"],
            },
        )
        losses["loss"].backward()
        opt.step()
        opt.zero_grad()

    ckpt = tmp_path / "ckpt"
    save_checkpoint(model, cfg, ckpt)

    pred = VisionMultitaskPredictor()

    class _MD:
        model_dir = tmp_path

    pred.setup(ckpt, model_def=_MD(), device="cpu")
    raw = pred.predict(rows[0])
    parsed = json.loads(raw)
    assert "scene" in parsed and "detections" in parsed and "pose" in parsed
    assert len(parsed["pose"]["keypoints"]) == 12

    md = ModelDefinition(
        name="vision",
        model_id="vision",
        version="0.1.0",
        architecture="vision_multitask",
        dataset={},
    )
    object.__setattr__(md, "model_dir", tmp_path)
    export_dir = tmp_path / "export"
    export_onnx(md, ckpt, export_dir)
    assert (export_dir / "model.onnx").is_file()
    assert (export_dir / "deploy" / "client.py").is_file()

    pred2 = VisionMultitaskPredictor()
    pred2.setup(export_dir, model_def=_MD(), device="cpu")
    assert pred2.backend == "onnx"
    raw2 = pred2.predict(rows[0])
    parsed2 = json.loads(raw2)
    assert parsed2["scene"]["label"] in cfg.scene_labels
