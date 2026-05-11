"""Packaging tests.

Two halves:
  1. .fm archive safety primitives — path traversal, oversize, manifest-only
     extraction. These don't depend on any specific model.
  2. End-to-end package_jcl + verify_package integration using a fake
     ModernBERT-class encoder + classifier-heads sidecar (tiny, on the fly).
     Confirms the manifest contract for the v2 classifier path:
     `architecture: candle_bert_classifier`.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")
safetensors_torch = pytest.importorskip("safetensors.torch")

from transformers import AutoTokenizer  # noqa: E402

from flow_ml.packaging.package_model import (  # noqa: E402
    package_jcl,
    verify_package,
)


# ---------------------------------------------------------------------------
# .fm safety primitives — fake-manifest hand-built archives
# ---------------------------------------------------------------------------


def _build_minimal_fm(fm_path: Path, *, manifest: dict, files: dict[str, bytes]) -> Path:
    import zipfile
    fm_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(fm_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("manifest.json", json.dumps(manifest, sort_keys=True))
        for name, data in files.items():
            zf.writestr(name, data)
    return fm_path


def _hash_bytes(data: bytes) -> str:
    import hashlib
    return hashlib.sha256(data).hexdigest()


def _valid_fake_manifest(weights: bytes) -> dict:
    return {
        "model_id": "fake:v1",
        "task": "spool_interpretation",
        "weights": "model.safetensors",
        "tokenizer": "tokenizer.json",
        "config": "config.json",
        "max_input_tokens": 64,
        "expected_latency_ms": 10,
        "version": "v1",
        "sha256": {"model.safetensors": _hash_bytes(weights)},
    }


def test_read_manifest_from_fm_does_direct_read(tmp_path: Path) -> None:
    """read_manifest_from_fm should return a ModelManifest without extracting."""
    from flow_ml.packaging.package_model import read_manifest_from_fm

    weights = b"x" * 16
    fm = _build_minimal_fm(
        tmp_path / "ok.fm",
        manifest=_valid_fake_manifest(weights),
        files={"model.safetensors": weights},
    )
    before = set(tmp_path.iterdir())
    md = read_manifest_from_fm(fm)
    after = set(tmp_path.iterdir())
    assert md.model_id == "fake:v1"
    assert md.task == "spool_interpretation"
    assert "model.safetensors" in md.sha256
    assert before == after, "read_manifest_from_fm must not write to disk"


def test_verify_rejects_path_traversal(tmp_path: Path) -> None:
    weights = b"x" * 16
    fm = _build_minimal_fm(
        tmp_path / "evil.fm",
        manifest=_valid_fake_manifest(weights),
        files={"model.safetensors": weights, "../escape.txt": b"evil"},
    )
    result = verify_package(fm)
    assert not result.ok
    assert any(
        "'..'" in i or "traversal" in i.lower() or "archive validation" in i.lower()
        for i in result.issues
    ), result.issues


def test_verify_rejects_too_many_entries(tmp_path: Path) -> None:
    from flow_ml.packaging.package_model import MAX_ENTRIES
    weights = b"x" * 16
    extra = {f"junk_{i}.txt": b"" for i in range(MAX_ENTRIES + 5)}
    extra["model.safetensors"] = weights
    fm = _build_minimal_fm(
        tmp_path / "fat.fm",
        manifest=_valid_fake_manifest(weights),
        files=extra,
    )
    result = verify_package(fm)
    assert not result.ok
    assert any("entries" in i.lower() and "max" in i.lower() for i in result.issues), result.issues


# ---------------------------------------------------------------------------
# End-to-end: package_jcl with a fake ModernBERT classifier checkpoint
# ---------------------------------------------------------------------------


def _make_tiny_bert_classifier_checkpoint(tmp_path: Path) -> Path:
    """Write a tiny BERT encoder + 4 classifier heads sidecar to disk so
    package_jcl has a complete v2 classifier checkpoint to package."""
    from transformers import BertConfig, BertModel

    cfg = BertConfig(
        vocab_size=128,
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        intermediate_size=64,
        max_position_embeddings=128,
    )
    model = BertModel(cfg)
    ckpt = tmp_path / "ckpt"
    ckpt.mkdir()
    model.save_pretrained(ckpt, safe_serialization=True)

    # 4-head classifier sidecar: each head is a tiny nn.Linear off
    # the [CLS] pooled representation. Mirrors what jcl_classifier.py
    # writes during training.
    head_state: dict[str, torch.Tensor] = {}
    for name, dim in (
        ("validity", 2),
        ("error_code", 8),
        ("severity", 3),
        ("line", 64),
    ):
        head_state[f"heads.{name}.weight"] = torch.randn(dim, cfg.hidden_size)
        head_state[f"heads.{name}.bias"] = torch.zeros(dim)
    safetensors_torch.save_file(head_state, ckpt / "classifier_heads.safetensors")

    # Stand in for the custom JCL tokenizer the trainer writes — a real
    # tokenizer.json keeps the manifest sha256 check happy.
    tok = AutoTokenizer.from_pretrained("bert-base-uncased")
    tok.save_pretrained(ckpt)
    return ckpt


def test_package_jcl_emits_jcl_validation_task(tmp_path: Path) -> None:
    """Smoke-test package_jcl with a tiny checkpoint. Confirms the manifest
    contract + that the JCL-specific schema/contracts files end up in the
    .fm archive."""
    ckpt = _make_tiny_bert_classifier_checkpoint(tmp_path)
    out = tmp_path / "models" / "jcl-test"
    repo_dataset = (
        Path(__file__).resolve().parents[1] / "models" / "jcl-validator" / "datasets"
    )
    result = package_jcl(
        ckpt,
        out,
        prompt_spec_path=repo_dataset / "prompt_spec.json",
        schema_path=repo_dataset / "jcl_validation_schema.json",
        contracts_path=repo_dataset / "node_contracts.json",
        model_id="jcl-test:v1",
        max_input_tokens=64,
        expected_latency_ms=10,
        weights_dtype="f32",
    )
    manifest_path = result.pkg_dir / "manifest.json"
    assert manifest_path.exists()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["task"] == "jcl_validation"
    assert manifest["model_id"] == "jcl-test:v1"
    assert manifest["architecture"] == "candle_bert_classifier"
    for required in (
        "model.safetensors",
        "classifier_heads.safetensors",
        "config.json",
        "tokenizer.json",
        "prompt_spec.json",
        "jcl_validation_schema.json",
        "node_contracts.json",
    ):
        assert (result.pkg_dir / required).exists(), f"missing {required} in package"
