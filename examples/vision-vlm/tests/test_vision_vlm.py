"""Vision-VLM tests — pure-python by default; torch optional."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


@pytest.fixture(scope="module")
def plugin():
    from vlm_plugin import (  # noqa: F401
        compute_vision_vlm_metrics,
        validate_vision_vlm,
    )
    return True


def test_validate_model_dir() -> None:
    from maatml.scaffold import validate_model_dir

    errors = validate_model_dir(ROOT)
    assert errors == [], errors


def test_describe_deterministic(plugin) -> None:
    from vlm_plugin.describe import describe, extract_gt

    expected = {
        "scene": {"label": "checker", "confidence": 1.0},
        "detections": [
            {"label": "circle", "box": [0.1, 0.1, 0.2, 0.2], "confidence": 1.0},
            {"label": "circle", "box": [0.3, 0.1, 0.4, 0.2], "confidence": 1.0},
            {"label": "star", "box": [0.5, 0.1, 0.6, 0.2], "confidence": 1.0},
        ],
        "pose": {
            "keypoints": [
                {"name": "head", "x": 0.5, "y": 0.2, "confidence": 1.0},
                {"name": "neck", "x": 0.5, "y": 0.3, "confidence": 1.0},
                {"name": "l_shoulder", "x": 0.4, "y": 0.32, "confidence": 1.0},
                {"name": "r_shoulder", "x": 0.6, "y": 0.32, "confidence": 1.0},
                {"name": "l_elbow", "x": 0.35, "y": 0.25, "confidence": 1.0},
                {"name": "r_elbow", "x": 0.65, "y": 0.25, "confidence": 1.0},
                {"name": "l_wrist", "x": 0.3, "y": 0.18, "confidence": 1.0},
                {"name": "r_wrist", "x": 0.7, "y": 0.18, "confidence": 1.0},
                {"name": "hip", "x": 0.5, "y": 0.55, "confidence": 1.0},
                {"name": "l_knee", "x": 0.35, "y": 0.75, "confidence": 1.0},
                {"name": "r_knee", "x": 0.65, "y": 0.75, "confidence": 1.0},
                {"name": "feet", "x": 0.5, "y": 0.9, "confidence": 1.0},
            ]
        },
    }
    gt = extract_gt(expected)
    a = describe(gt)
    b = describe(gt)
    assert a == b
    assert "checker" in a.lower() or "checkerboard" in a.lower()
    assert "circle" in a.lower()
    assert "star" in a.lower()
    assert "raised" in a.lower()


def test_validator_accepts_description(plugin) -> None:
    from vlm_plugin.validator import validate_vision_vlm

    raw = json.dumps(
        {
            "description": (
                "A plain background with one square; the figure stands with "
                "arms relaxed and a neutral stance."
            )
        }
    )
    result = validate_vision_vlm(raw, schema_path=ROOT / "datasets" / "schema.json")
    assert result.ok, result.errors


def test_validator_rejects_too_long(plugin) -> None:
    from vlm_plugin.validator import validate_vision_vlm

    long = "word " * 50
    raw = json.dumps({"description": long.strip()})
    result = validate_vision_vlm(raw, schema_path=ROOT / "datasets" / "schema.json")
    assert not result.ok
    assert any(e.code == "too_long" for e in result.errors)


def test_metrics_perfect_match(plugin) -> None:
    from maatml.validation.base import ValidationResult
    from vlm_plugin.metrics import compute_vision_vlm_metrics

    desc = (
        "A striped background with two circles and a star; the figure stands "
        "with arms lowered and a wide stance."
    )
    row = {
        "gt": {
            "scene": "striped",
            "shape_counts": {"circle": 2, "star": 1},
            "arms": "lowered",
            "stance": "wide",
        },
        "expected_output": {"description": desc},
    }

    class _Item:
        def __init__(self) -> None:
            self.row = row
            self.gen_text = json.dumps({"description": desc})
            self.result = ValidationResult(
                raw_output=self.gen_text, n_layers=4, required_layers={1, 2, 3, 4}
            )
            self.result.passed_layers = {1, 2, 3, 4}

    metrics = compute_vision_vlm_metrics([_Item()])
    assert metrics["scene_mention_rate"] == 1.0
    assert metrics["shape_mention_f1"] == 1.0
    assert metrics["pose_phrase_rate"] == 1.0
    assert metrics["brevity_rate"] == 1.0


def test_plugin_registers(plugin) -> None:
    from maatml.registry import GENERATORS, METRICS, PREDICTORS, TRAINERS, VALIDATORS

    assert "vlm_sft" in TRAINERS.names()
    assert "vision_vlm" in PREDICTORS.names()
    assert "vision_vlm" in VALIDATORS.names()
    assert "vision_vlm" in METRICS.names()
    assert "described_scenes" in GENERATORS.names()


def test_normalize_description_wraps_plain_text(plugin) -> None:
    from vlm_plugin.predictor import _normalize_description

    out = json.loads(_normalize_description("A plain background with no shapes."))
    assert out["description"].startswith("A plain")


@pytest.mark.skipif(
    __import__("importlib").util.find_spec("torch") is None,
    reason="torch required",
)
def test_synth_render_and_describe(plugin) -> None:
    pytest.importorskip("PIL")
    from vlm_plugin.datagen import build_described_row

    tmp = ROOT / "output" / "_test_images"
    tmp.mkdir(parents=True, exist_ok=True)
    row = build_described_row(
        0,
        base_seed=0,
        size=64,
        image_rel="output/_test_images/{id}.png",
        images_dir=tmp,
    )
    assert "description" in row["expected_output"]
    assert "gt" in row
    assert Path(ROOT / row["image"]).is_file()


@pytest.mark.skipif(
    __import__("importlib").util.find_spec("torch") is None,
    reason="torch required",
)
def test_label_mask_collate_masks_pad(plugin) -> None:
    """Collate should mask pad (+ image tokens when known) to -100."""
    torch = pytest.importorskip("torch")
    from vlm_plugin.trainer import _build_messages

    pad_id = 0
    image_token_id = 99
    # Simulate a short padded batch: [tok, image, tok, pad]
    input_ids = torch.tensor([[1, image_token_id, 2, pad_id], [3, 4, pad_id, pad_id]])
    labels = input_ids.clone()
    labels[labels == pad_id] = -100
    labels[labels == image_token_id] = -100
    assert labels.tolist() == [[1, -100, 2, -100], [3, 4, -100, -100]]
    messages = _build_messages("Describe.", "A plain background.")
    assert messages[0]["role"] == "user"
    assert any(c.get("type") == "image" for c in messages[0]["content"])


@pytest.mark.skipif(
    __import__("importlib").util.find_spec("torch") is None,
    reason="torch required",
)
def test_predictor_wraps_generate_output(plugin, monkeypatch) -> None:
    """Predictor normalizes free-text generate() output to description JSON."""
    from vlm_plugin.predictor import VisionVlmPredictor, _normalize_description

    assert json.loads(_normalize_description("Hello scene."))["description"] == "Hello scene."

    pred = VisionVlmPredictor()
    pred.backend = "transformers"
    pred.model = object()
    pred.processor = object()
    pred.device = "cpu"
    pred.max_new_tokens = 8
    pred.user_prompt = "Describe."

    class _Tok:
        def decode(self, *_a, **_k):
            return "A striped background with one circle."

    class _Proc:
        tokenizer = _Tok()

        def apply_chat_template(self, *_a, **_k):
            return "<prompt>"

        def __call__(self, **_k):
            class _T:
                def to(self, _device):
                    return {"input_ids": __import__("torch").tensor([[1, 2, 3]])}

            return _T()

    class _Model:
        def generate(self, **_k):
            return __import__("torch").tensor([[1, 2, 3, 4, 5]])

    pred.processor = _Proc()
    pred.model = _Model()
    monkeypatch.setattr(
        "vlm_plugin.predictor._resolve_image_bytes",
        lambda *_a, **_k: b"fake",
    )
    monkeypatch.setattr(
        "vlm_plugin.predictor._to_pil",
        lambda _b: object(),
    )
    out = json.loads(pred.predict({"image": "ignored.png"}))
    assert "background" in out["description"].lower() or "striped" in out["description"].lower()
