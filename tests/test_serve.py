"""Tests for ``maatml serve`` — fake predictor, no torch required."""
from __future__ import annotations

import json
import threading
import urllib.error
import urllib.request
from pathlib import Path

import pytest

from maatml.config import ModelDefinition, PackagingSpec
from maatml.registry import PREDICTORS, VALIDATORS, discover_plugins
from maatml.serve import build_serve_context, serve_model
from maatml.validation.base import ValidationError, ValidationResult


class _FakePredictor:
    def __init__(self) -> None:
        self.setup_called = False
        self.last_row: dict | None = None

    def setup(self, checkpoint_dir, **kwargs) -> None:  # noqa: ANN001
        del kwargs
        self.setup_called = True
        self.checkpoint_dir = Path(checkpoint_dir)

    def predict(self, row: dict) -> str:
        self.last_row = row
        text = row.get("request") or row.get("image") or ""
        return json.dumps({"echo": text, "ok": True})


def _fake_validator(raw_output: str, **kwargs) -> ValidationResult:  # noqa: ANN003
    del kwargs
    result = ValidationResult(raw_output=raw_output, n_layers=1, required_layers={1})
    try:
        parsed = json.loads(raw_output)
        result.parsed = parsed
        if parsed.get("ok") is True:
            result.passed_layers.add(1)
        else:
            result.errors.append(
                ValidationError(layer=1, code="not_ok", message="ok flag missing")
            )
    except json.JSONDecodeError as exc:
        result.errors.append(
            ValidationError(layer=1, code="invalid_json", message=str(exc))
        )
    return result


@pytest.fixture()
def serve_model_dir(tmp_path: Path) -> tuple[ModelDefinition, Path]:
    discover_plugins(force=True)
    PREDICTORS.register("fake_echo", _FakePredictor, source="test")
    VALIDATORS.register("fake_echo", _fake_validator, source="test")

    model_dir = tmp_path / "toy"
    model_dir.mkdir()
    ckpt = model_dir / "output" / "checkpoints" / "smoke"
    ckpt.mkdir(parents=True)
    (ckpt / "model.safetensors").write_bytes(b"x")

    md = ModelDefinition(
        name="toy-serve",
        model_id="toy-serve",
        version="0.1.0",
        architecture="causal_sft",
        evaluation={"predictor": "fake_echo", "validator": "fake_echo"},
        packaging=PackagingSpec(expected_latency_ms=100),
    )
    object.__setattr__(md, "model_dir", model_dir)
    return md, ckpt


def _http_json(method: str, url: str, body: dict | None = None) -> tuple[int, dict]:
    data = None
    headers = {"Accept": "application/json"}
    if body is not None:
        data = json.dumps(body).encode("utf-8")
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=5) as resp:
            return resp.status, json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        payload = json.loads(exc.read().decode("utf-8"))
        return exc.code, payload


def test_build_serve_context_loads_predictor(
    serve_model_dir: tuple[ModelDefinition, Path],
) -> None:
    md, ckpt = serve_model_dir
    ctx = build_serve_context(md, checkpoint=ckpt, device="cpu")
    assert ctx.predictor.setup_called
    assert ctx.checkpoint_dir == ckpt.resolve()
    assert ctx.validator is not None


def test_serve_health_info_predict_validate(
    serve_model_dir: tuple[ModelDefinition, Path],
) -> None:
    md, ckpt = serve_model_dir
    ctx = build_serve_context(md, checkpoint=ckpt, device="cpu")
    server = serve_model(md, host="127.0.0.1", port=0, context=ctx)
    host, port = server.server_address[:2]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://{host}:{port}"
    try:
        status, health = _http_json("GET", f"{base}/health")
        assert status == 200
        assert health["status"] == "ok"
        assert health["identity"] == "toy-serve@0.1.0"
        assert health["architecture"] == "causal_sft"

        status, info = _http_json("GET", f"{base}/info")
        assert status == 200
        assert info["packaging"]["expected_latency_ms"] == 100
        assert info["predictor"] == "fake_echo"

        status, pred = _http_json(
            "POST", f"{base}/predict", {"request": "hello serve"}
        )
        assert status == 200
        assert pred["output"]["echo"] == "hello serve"
        assert pred["output"]["ok"] is True
        assert "latency_ms" in pred
        assert "valid" not in pred

        status, pred_v = _http_json(
            "POST",
            f"{base}/predict?validate=1",
            {"request": "hello serve"},
        )
        assert status == 200
        assert pred_v["valid"] is True
        assert pred_v["errors"] == []

        status, missing = _http_json("GET", f"{base}/nope")
        assert status == 404
        assert "error" in missing
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_serve_predict_bad_body(
    serve_model_dir: tuple[ModelDefinition, Path],
) -> None:
    md, ckpt = serve_model_dir
    ctx = build_serve_context(md, checkpoint=ckpt, device="cpu")
    server = serve_model(md, host="127.0.0.1", port=0, context=ctx)
    host, port = server.server_address[:2]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://{host}:{port}"
    try:
        req = urllib.request.Request(
            f"{base}/predict",
            data=b"not-json",
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with pytest.raises(urllib.error.HTTPError) as exc_info:
            urllib.request.urlopen(req, timeout=5)
        assert exc_info.value.code == 400
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
