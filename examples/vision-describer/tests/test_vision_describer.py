"""Vision-describer tests, dependency-free (no torch required)."""
from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def plugin():
    from vision_describer_plugin import (  # noqa: F401
        compute_vision_describer_metrics,
        validate_vision_describer,
    )

    return True


def test_validate_model_dir() -> None:
    from maatml.scaffold import validate_model_dir

    errors = validate_model_dir(ROOT)
    assert errors == [], errors


def test_linearize_filters_and_rounds(plugin) -> None:
    from vision_describer_plugin.linearize import (
        clean_vision_result,
        linearize_vision_result,
    )

    raw = {
        "scene": {"label": "striped", "confidence": 0.94123},
        "detections": [
            {
                "label": "circle",
                "box": [0.1111, 0.2222, 0.3333, 0.4444],
                "confidence": 0.88,
            },
            {
                "label": "square",
                "box": [0.5, 0.5, 0.6, 0.6],
                "confidence": 0.1,  # below thresh
            },
        ],
        "pose": {
            "keypoints": [
                {"name": "head", "x": 0.50123, "y": 0.20123, "confidence": 1.0},
            ]
            + [
                {"name": n, "x": 0.0, "y": 0.0, "confidence": 0.0}
                for n in [
                    "neck",
                    "l_shoulder",
                    "r_shoulder",
                    "l_elbow",
                    "r_elbow",
                    "l_wrist",
                    "r_wrist",
                    "hip",
                    "l_knee",
                    "r_knee",
                    "feet",
                ]
            ]
        },
    }
    cleaned = clean_vision_result(raw)
    assert cleaned["scene"]["confidence"] == 0.94
    assert len(cleaned["detections"]) == 1
    assert cleaned["detections"][0]["box"] == [0.11, 0.22, 0.33, 0.44]
    text = linearize_vision_result(raw)
    assert "0.1111" not in text
    assert '"label":"circle"' in text
    assert text == linearize_vision_result(text)  # idempotent


def test_describe_mentions_scene_and_objects(plugin) -> None:
    from vision_describer_plugin.describe import describe_vision_result

    payload = {
        "scene": {"label": "checker", "confidence": 1.0},
        "detections": [
            {"label": "circle", "box": [0.1, 0.1, 0.2, 0.2], "confidence": 0.9},
            {"label": "circle", "box": [0.3, 0.3, 0.4, 0.4], "confidence": 0.9},
            {"label": "square", "box": [0.5, 0.5, 0.6, 0.6], "confidence": 0.9},
        ],
        "pose": {
            "keypoints": [
                {"name": "head", "x": 0.5, "y": 0.2, "confidence": 1.0},
                {"name": "neck", "x": 0.5, "y": 0.3, "confidence": 1.0},
                {"name": "l_shoulder", "x": 0.4, "y": 0.32, "confidence": 1.0},
                {"name": "r_shoulder", "x": 0.6, "y": 0.32, "confidence": 1.0},
                {"name": "l_elbow", "x": 0.35, "y": 0.25, "confidence": 1.0},
                {"name": "r_elbow", "x": 0.65, "y": 0.25, "confidence": 1.0},
                {"name": "l_wrist", "x": 0.33, "y": 0.18, "confidence": 1.0},
                {"name": "r_wrist", "x": 0.67, "y": 0.18, "confidence": 1.0},
                {"name": "hip", "x": 0.5, "y": 0.55, "confidence": 1.0},
                {"name": "l_knee", "x": 0.45, "y": 0.7, "confidence": 1.0},
                {"name": "r_knee", "x": 0.55, "y": 0.7, "confidence": 1.0},
                {"name": "feet", "x": 0.5, "y": 0.85, "confidence": 1.0},
            ]
        },
    }
    text = describe_vision_result(payload)
    assert "checker" in text
    assert "circle" in text
    assert "square" in text
    assert "raising both arms" in text
    assert len(text.split()) <= 30


def test_validator_accepts_seed_row(plugin) -> None:
    from vision_describer_plugin.validator import validate_vision_describer

    row = json.loads(
        (ROOT / "datasets/samples/seed_samples.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()[0]
    )
    result = validate_vision_describer(
        json.dumps(row["expected_description"]),
        schema_path=ROOT / "datasets/schema.json",
        contracts_path=ROOT / "datasets/node_contracts.json",
        user_prompt=row["request"],
    )
    assert result.ok, result.errors


def test_validator_rejects_bad_json(plugin) -> None:
    from vision_describer_plugin.validator import validate_vision_describer

    result = validate_vision_describer(
        "not-json",
        schema_path=ROOT / "datasets/schema.json",
    )
    assert not result.ok
    assert any(e.code == "invalid_json" for e in result.errors)


def test_validator_rejects_ungrounded_scene(plugin) -> None:
    from vision_describer_plugin.linearize import linearize_vision_result
    from vision_describer_plugin.validator import validate_vision_describer

    request = linearize_vision_result(
        {
            "scene": {"label": "noisy", "confidence": 1.0},
            "detections": [],
            "pose": {"keypoints": []},
        }
    )
    result = validate_vision_describer(
        json.dumps({"description": "A plain scene contains no shapes."}),
        schema_path=ROOT / "datasets/schema.json",
        user_prompt=request,
    )
    assert not result.ok
    assert any(e.code == "scene_ungrounded" for e in result.errors)


def test_schemas_round_trip(plugin) -> None:
    from vision_describer_plugin.schemas import (
        DescriptionResult,
        VisionDescriberSample,
    )

    sample = VisionDescriberSample(
        sample_id="t1",
        source="test",
        category="plain",
        request='{"scene":{"label":"plain"}}',
        expected_description=DescriptionResult(
            description="A plain scene contains no shapes."
        ),
    )
    assert sample.within_word_limit()
    dumped = sample.model_dump()
    assert dumped["expected_description"]["description"].startswith("A plain")


def test_metrics_perfect_match(plugin) -> None:
    from maatml.validation.base import ValidationResult
    from vision_describer_plugin.metrics import compute_vision_describer_metrics

    row = json.loads(
        (ROOT / "datasets/samples/seed_samples.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()[0]
    )
    gen = json.dumps(row["expected_description"])

    class _Item:
        def __init__(self) -> None:
            self.row = row
            self.gen_text = gen
            self.result = ValidationResult(
                raw_output=gen, n_layers=6, required_layers={1, 2, 3, 4, 5, 6}
            )
            self.result.parsed = row["expected_description"]
            self.result.passed_layers = {1, 2, 3, 4, 5, 6}

    metrics = compute_vision_describer_metrics([_Item()])
    assert metrics["exact_match_rate"] == 1.0
    assert metrics["json_parse_rate"] == 1.0
    assert metrics["scene_grounding_rate"] == 1.0


def test_plugin_registration(plugin) -> None:
    from maatml.registry import GENERATORS, METRICS, VALIDATORS, discover_plugins
    from maatml.registry import load_model_plugins

    discover_plugins(force=True)
    load_model_plugins(ROOT, ["./vision_describer_plugin"])
    assert "vision_describer" in VALIDATORS.names()
    assert "vision_describer" in METRICS.names()
    assert "vision_describer" in GENERATORS.names()


def test_compose_client_with_fake_servers(plugin) -> None:
    import importlib.util

    script = ROOT / "scripts" / "compose_client.py"
    spec = importlib.util.spec_from_file_location(
        "maatml_test_compose_client", script
    )
    assert spec is not None and spec.loader is not None
    compose_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(compose_mod)

    vision_payload = {
        "scene": {"label": "striped", "confidence": 0.9},
        "detections": [
            {"label": "star", "box": [0.1, 0.1, 0.2, 0.2], "confidence": 0.95}
        ],
        "pose": {
            "keypoints": [
                {"name": n, "x": 0.5, "y": 0.5, "confidence": 1.0}
                for n in [
                    "head",
                    "neck",
                    "l_shoulder",
                    "r_shoulder",
                    "l_elbow",
                    "r_elbow",
                    "l_wrist",
                    "r_wrist",
                    "hip",
                    "l_knee",
                    "r_knee",
                    "feet",
                ]
            ]
        },
    }

    class VisionHandler(BaseHTTPRequestHandler):
        def log_message(self, fmt: str, *args) -> None:  # noqa: A003
            return

        def do_POST(self) -> None:  # noqa: N802
            length = int(self.headers.get("Content-Length") or 0)
            self.rfile.read(length)
            body = json.dumps(
                {"output": vision_payload, "raw": json.dumps(vision_payload), "latency_ms": 1.0}
            ).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    class DescriberHandler(BaseHTTPRequestHandler):
        def log_message(self, fmt: str, *args) -> None:  # noqa: A003
            return

        def do_POST(self) -> None:  # noqa: N802
            length = int(self.headers.get("Content-Length") or 0)
            raw = self.rfile.read(length)
            payload = json.loads(raw.decode())
            assert "request" in payload
            out = {"description": "A striped scene contains a star."}
            body = json.dumps(
                {"output": out, "raw": json.dumps(out), "latency_ms": 2.0}
            ).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    vision_srv = ThreadingHTTPServer(("127.0.0.1", 0), VisionHandler)
    desc_srv = ThreadingHTTPServer(("127.0.0.1", 0), DescriberHandler)
    threading.Thread(target=vision_srv.serve_forever, daemon=True).start()
    threading.Thread(target=desc_srv.serve_forever, daemon=True).start()
    try:
        v_port = vision_srv.server_address[1]
        d_port = desc_srv.server_address[1]
        # Use a data-URI so we don't need a real file on disk.
        import base64

        png = base64.b64encode(
            b"\x89PNG\r\n\x1a\n" + b"\x00" * 16
        ).decode()
        result = compose_mod.compose(
            f"data:image/png;base64,{png}",
            vision_url=f"http://127.0.0.1:{v_port}",
            describer_url=f"http://127.0.0.1:{d_port}",
        )
        assert result["description"] == "A striped scene contains a star."
        assert result["vision"]["scene"]["label"] == "striped"
        assert result["latency_ms"]["vision"] == 1.0
    finally:
        vision_srv.shutdown()
        desc_srv.shutdown()
