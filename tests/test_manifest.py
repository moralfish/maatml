"""Export manifest build / verify checksum roundtrip."""
from __future__ import annotations

from pathlib import Path

from maatml.config import ModelDefinition, PackagingSpec
from maatml.export.manifest import build_manifest, verify_manifest, write_manifest
from maatml.utils.io import sha256_file, write_json


def test_manifest_verify_roundtrip(tmp_path: Path) -> None:
    export_dir = tmp_path / "export"
    export_dir.mkdir()
    a = export_dir / "weights.bin"
    b = export_dir / "tokenizer.json"
    a.write_bytes(b"fake-weights")
    b.write_text('{"version":1}', encoding="utf-8")

    md = ModelDefinition(
        name="toy",
        model_id="toy",
        version="0.1.0",
        architecture="causal_sft",
        packaging=PackagingSpec(max_input_tokens=512, expected_latency_ms=100),
    )
    object.__setattr__(md, "model_dir", tmp_path)

    manifest = build_manifest(
        model_def=md,
        export_dir=export_dir,
        files=[a, b],
        formats=["safetensors"],
        source_checkpoint=tmp_path / "ckpt",
        run_id="run-1",
    )
    write_manifest(export_dir, manifest)

    assert (export_dir / "manifest.json").is_file()
    assert manifest["identity"] == "toy@0.1.0"
    assert manifest["run_id"] == "run-1"
    assert len(manifest["files"]) == 2
    assert manifest["files"][0]["sha256"] == sha256_file(a)

    assert verify_manifest(export_dir) == []
    assert verify_manifest(export_dir / "manifest.json") == []


def test_verify_detects_tamper(tmp_path: Path) -> None:
    export_dir = tmp_path / "export"
    export_dir.mkdir()
    f = export_dir / "model.safetensors"
    f.write_bytes(b"v1")
    write_json(
        export_dir / "manifest.json",
        {
            "files": [{"path": "model.safetensors", "sha256": sha256_file(f)}],
        },
    )
    assert verify_manifest(export_dir) == []
    f.write_bytes(b"v2-tampered")
    errors = verify_manifest(export_dir)
    assert len(errors) == 1
    assert "checksum mismatch" in errors[0]
