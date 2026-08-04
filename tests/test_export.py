"""Safetensors export bundle (fake checkpoint, no real weights required)."""

from __future__ import annotations

import json
import struct
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
    assert resolve_export_format("seq2seq", "mlx") == "mlx"
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


def _write_safetensors(path: Path, names: list[str]) -> None:
    """Write a minimal valid .safetensors file with the given tensor names."""
    header: dict[str, object] = {}
    offset = 0
    for name in names:
        header[name] = {"dtype": "F32", "shape": [1], "data_offsets": [offset, offset + 4]}
        offset += 4
    hb = json.dumps(header).encode("utf-8")
    path.write_bytes(struct.pack("<Q", len(hb)) + hb + b"\x00" * offset)


def _seq2seq_checkpoint(tmp_path: Path, *, with_spiece: bool = True) -> Path:
    ckpt = tmp_path / "ckpt"
    ckpt.mkdir()
    _write_safetensors(ckpt / "model.safetensors", ["shared.weight", "lm_head.weight"])
    (ckpt / "config.json").write_text(
        json.dumps(
            {
                "architectures": ["T5ForConditionalGeneration"],
                "tie_word_embeddings": True,
                "decoder_start_token_id": 0,
                "eos_token_id": 1,
            }
        ),
        encoding="utf-8",
    )
    (ckpt / "generation_config.json").write_text(
        '{"decoder_start_token_id": 0, "max_new_tokens": 256}', encoding="utf-8"
    )
    (ckpt / "tokenizer.json").write_text('{"model":{}}', encoding="utf-8")
    (ckpt / "tokenizer_config.json").write_text("{}", encoding="utf-8")
    if with_spiece:
        (ckpt / "spiece.model").write_bytes(b"spm")
    return ckpt


def _seq2seq_model_def(tmp_path: Path) -> ModelDefinition:
    model_dir = tmp_path / "model"
    model_dir.mkdir(exist_ok=True)
    md = ModelDefinition(
        name="toy-seq2seq",
        model_id="toy-seq2seq",
        version="0.1.0",
        architecture="seq2seq",
        base_model="toy/t5",
        dataset={"source_prefix": "interpret spool: "},
        training={"source_max_len": 1024, "generation": {"max_new_tokens": 512}},
        evaluation={"repair_braces": True},
    )
    object.__setattr__(md, "model_dir", model_dir)
    return md


def test_export_mlx_seq2seq_bundle(tmp_path: Path) -> None:
    from maatml.export.mlx_export import export_mlx

    ckpt = _seq2seq_checkpoint(tmp_path)
    md = _seq2seq_model_def(tmp_path)
    out = tmp_path / "export"

    # No mlx_lm required: the seq2seq path assembles the bundle directly.
    export_mlx(md, ckpt, out, run_id="run-1")

    bundle = out / "toy-seq2seq.mlx"
    for name in (
        "config.json",
        "model.safetensors",
        "tokenizer.json",
        "tokenizer_config.json",
        "generation_config.json",
        "spiece.model",
        "serving.json",
    ):
        assert (bundle / name).is_file(), name

    serving = read_json(bundle / "serving.json")
    assert serving == {
        "contract": "maatml.serving/1",
        "kind": "seq2seq_lm",
        "max_input_tokens": 1024,
        "max_new_tokens": 512,
        "decoder_start_token_id": 0,
        "eos_token_id": 1,
        "input_prefix": "interpret spool: ",
        "output_note": (
            "emits the JSON object body without outer braces; T5 vocab maps "
            "{ } to unk. Clients re-add the braces before parsing."
        ),
    }

    # The fine-tune untied the head, so the bundle config must say so.
    assert read_json(bundle / "config.json")["tie_word_embeddings"] is False
    # The top-level safetensors bundle keeps the checkpoint config untouched.
    assert read_json(out / "config.json")["tie_word_embeddings"] is True

    manifest = read_json(out / "manifest.json")
    assert manifest["runtime_hints"]["formats"] == ["safetensors", "mlx"]
    paths = {e["path"] for e in manifest["files"]}
    assert "toy-seq2seq.mlx/serving.json" in paths
    assert "toy-seq2seq.mlx/model.safetensors" in paths
    assert verify_manifest(out) == []


def test_export_mlx_seq2seq_no_repair_no_note(tmp_path: Path) -> None:
    from maatml.export.mlx_export import build_seq2seq_serving

    ckpt = _seq2seq_checkpoint(tmp_path)
    md = _seq2seq_model_def(tmp_path)
    md.evaluation = {}
    serving = build_seq2seq_serving(md, ckpt)
    assert "output_note" not in serving


def test_export_mlx_seq2seq_missing_spiece_warns(tmp_path: Path) -> None:
    from maatml.export.mlx_export import export_mlx

    ckpt = _seq2seq_checkpoint(tmp_path, with_spiece=False)
    # Tied checkpoint without an lm_head tensor keeps its config untouched.
    _write_safetensors(ckpt / "model.safetensors", ["shared.weight"])
    md = _seq2seq_model_def(tmp_path)
    out = tmp_path / "export"

    with pytest.warns(RuntimeWarning, match="spiece.model"):
        export_mlx(md, ckpt, out)

    bundle = out / "toy-seq2seq.mlx"
    assert not (bundle / "spiece.model").exists()
    assert read_json(bundle / "config.json")["tie_word_embeddings"] is True
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
    import maatml.evaluation.harness as harness_mod
    from maatml.evaluation.harness import Report

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
