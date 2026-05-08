from __future__ import annotations

from pathlib import Path

from flow_ml.models.manifest import ConfidenceThresholds, ModelManifest


def test_manifest_defaults() -> None:
    manifest = ModelManifest(
        model_id="jcl-validator:v1",
        task="jcl_validation",
        max_input_tokens=2048,
        expected_latency_ms=500,
    )
    assert manifest.runtime == "candle"
    assert manifest.confidence_thresholds == ConfidenceThresholds()
    assert manifest.confidence_thresholds.high == 0.9
    assert manifest.confidence_thresholds.low == 0.6


def test_manifest_round_trip(tmp_path: Path) -> None:
    manifest = ModelManifest(
        model_id="spool-interpreter:v1",
        task="spool_interpretation",
        max_input_tokens=2048,
        expected_latency_ms=750,
        labels_file="labels.json",
        prompt_spec_file="prompt_spec.json",
        base_checkpoint="HuggingFaceTB/SmolLM2-360M-Instruct",
        sha256={"model.safetensors": "deadbeef" * 8},
        confidence_thresholds=ConfidenceThresholds(high=0.85, low=0.55),
    )
    path = manifest.write(tmp_path / "manifest.json")
    loaded = ModelManifest.read(path)
    assert loaded == manifest
    assert loaded.sha256["model.safetensors"] == "deadbeef" * 8


def test_manifest_weights_dtype_defaults_to_f32() -> None:
    # Backwards compatibility: existing JCL/spool packages do not set
    # `weights_dtype`, so the field must default to `"f32"` to preserve
    # their on-disk shape and the runtime's mmap path.
    manifest = ModelManifest(
        model_id="jcl-validator:v1",
        task="jcl_validation",
        max_input_tokens=2048,
        expected_latency_ms=500,
    )
    assert manifest.weights_dtype == "f32"


def test_manifest_weights_dtype_round_trips(tmp_path: Path) -> None:
    # The 7B dsl-generator ships with `weights_dtype: "f16"`; it must
    # survive write/read intact so the runtime resolves the matching
    # candle DType.
    manifest = ModelManifest(
        model_id="dsl-generator:v1",
        task="dsl_generation",
        max_input_tokens=1024,
        expected_latency_ms=2000,
        weights_dtype="f16",
        prompt_spec_file="prompt_spec.json",
    )
    path = manifest.write(tmp_path / "manifest.json")
    loaded = ModelManifest.read(path)
    assert loaded.weights_dtype == "f16"
