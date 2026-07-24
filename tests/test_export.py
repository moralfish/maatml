"""Safetensors export bundle (fake checkpoint, no real weights required)."""
from __future__ import annotations

from pathlib import Path

import pytest

from maatml.config import ModelDefinition, PackagingSpec
from maatml.export.bundle import export_safetensors_bundle, resolve_export_format
from maatml.export.manifest import verify_manifest
from maatml.registry import discover_plugins
from maatml.utils.io import read_json


@pytest.fixture(autouse=True)
def _boot_exporters():
    discover_plugins(force=True)


def test_resolve_export_format_constraints() -> None:
    assert resolve_export_format("causal_sft") == "safetensors"
    assert resolve_export_format("causal_sft", "gguf") == "gguf"
    with pytest.raises(ValueError, match="only supported"):
        resolve_export_format("seq2seq", "gguf")
    with pytest.raises(ValueError, match="only supported"):
        resolve_export_format("classifier", "mlx")
    with pytest.raises(ValueError, match="Unknown export format"):
        resolve_export_format("causal_sft", "not-a-real-format")


def test_resolve_export_format_accepts_registered_plugin_format() -> None:
    from maatml.registry import EXPORTERS, register_exporter

    @register_exporter("toy_fmt")
    def _toy_export(model_def, checkpoint_dir, out_dir, *, run_id=None):  # noqa: ANN001
        del model_def, checkpoint_dir, run_id
        return Path(out_dir)

    try:
        assert resolve_export_format("vision_multitask", "toy_fmt") == "toy_fmt"
    finally:
        EXPORTERS.unregister("toy_fmt")


def test_export_safetensors_bundle(tmp_path: Path) -> None:
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    schema = model_dir / "schema.json"
    schema.write_text('{"type":"object"}', encoding="utf-8")

    ckpt = tmp_path / "ckpt"
    ckpt.mkdir()
    (ckpt / "model.safetensors").write_bytes(b"not-real-weights")
    (ckpt / "config.json").write_text('{"architectures":["Toy"]}', encoding="utf-8")
    (ckpt / "tokenizer.json").write_text('{"model":{}}', encoding="utf-8")
    (ckpt / "tokenizer_config.json").write_text("{}", encoding="utf-8")

    md = ModelDefinition(
        name="toy-export",
        model_id="toy-export",
        version="0.1.0",
        architecture="causal_sft",
        base_model="toy/base",
        dataset={"schema": "schema.json"},
        packaging=PackagingSpec(weights_dtype="f16"),
    )
    object.__setattr__(md, "model_dir", model_dir)

    out = tmp_path / "export"
    export_safetensors_bundle(md, ckpt, out, run_id="smoke-run")

    assert (out / "model.safetensors").is_file()
    assert (out / "tokenizer.json").is_file()
    assert (out / "schema.json").is_file()
    manifest = read_json(out / "manifest.json")
    assert manifest["identity"] == "toy-export@0.1.0"
    assert manifest["architecture"] == "causal_sft"
    assert "safetensors" in manifest["runtime_hints"]["formats"]
    paths = {e["path"] for e in manifest["files"]}
    assert "model.safetensors" in paths
    assert "schema.json" in paths
    assert verify_manifest(out) == []


def test_gguf_missing_tools_raises(tmp_path: Path) -> None:
    from maatml.export.gguf import export_gguf

    model_dir = tmp_path / "model"
    model_dir.mkdir()
    ckpt = tmp_path / "ckpt"
    ckpt.mkdir()
    (ckpt / "model.safetensors").write_bytes(b"x")
    (ckpt / "config.json").write_text("{}", encoding="utf-8")
    (ckpt / "tokenizer.json").write_text("{}", encoding="utf-8")

    md = ModelDefinition(
        name="toy",
        model_id="toy",
        version="0.1.0",
        architecture="causal_sft",
    )
    object.__setattr__(md, "model_dir", model_dir)

    with pytest.raises(ImportError, match="llama.cpp"):
        export_gguf(md, ckpt, tmp_path / "out")


def test_parity_skipped_without_benchmark(tmp_path: Path) -> None:
    from maatml.export.bundle import run_parity_check

    md = ModelDefinition(
        name="toy",
        model_id="toy",
        version="0.1.0",
        architecture="causal_sft",
    )
    object.__setattr__(md, "model_dir", tmp_path)
    out = run_parity_check(md, tmp_path / "export")
    assert out["skipped"] is True
    assert out["passed"] is True


def test_parity_gates_with_mocked_eval(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from maatml.evaluation.harness import Report
    import maatml.evaluation.harness as harness_mod

    model_dir = tmp_path / "model"
    model_dir.mkdir()
    bench = model_dir / "bench.jsonl"
    bench.write_text('{"request":"x","target":{}}\n', encoding="utf-8")
    export_dir = tmp_path / "export"
    export_dir.mkdir()

    md = ModelDefinition(
        name="toy",
        model_id="toy",
        version="0.1.0",
        architecture="causal_sft",
        dataset={"benchmark_samples": "bench.jsonl"},
        evaluation={
            "predictor": "causal_sft",
            "gates": {"json_parse_rate": 0.9},
        },
    )
    object.__setattr__(md, "model_dir", model_dir)

    def _fake_eval(**kwargs):
        del kwargs
        return Report(model_id="toy", metrics={"json_parse_rate": 0.95})

    monkeypatch.setattr(harness_mod, "run_evaluation", _fake_eval)

    from maatml.export.bundle import run_parity_check

    result = run_parity_check(md, export_dir)
    assert result["skipped"] is False
    assert result["passed"] is True
