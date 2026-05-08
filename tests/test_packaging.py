from __future__ import annotations

import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from transformers import BertConfig, BertModel, BertTokenizerFast  # noqa: E402

from flow_ml.packaging.package_model import (  # noqa: E402
    package_jcl,
    package_spool,
    verify_package,
)
from flow_ml.training.jcl_validator import JclMultiHeadModel  # noqa: E402


def _vocab_file(tmp_path: Path) -> Path:
    """Write a minimal WordPiece vocab file for BertTokenizerFast."""
    vocab = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"] + [f"tok{i}" for i in range(50)]
    p = tmp_path / "vocab.txt"
    p.write_text("\n".join(vocab) + "\n", encoding="utf-8")
    return p


def _make_jcl_checkpoint(tmp_path: Path) -> Path:
    cfg = BertConfig(
        vocab_size=55,
        hidden_size=8,
        num_hidden_layers=1,
        num_attention_heads=1,
        intermediate_size=16,
        max_position_embeddings=64,
    )
    encoder = BertModel(cfg)
    model = JclMultiHeadModel(encoder)
    ckpt = tmp_path / "ckpt"
    saved = model.save(ckpt, base_model_id="dummy/tiny-bert")

    vocab_path = _vocab_file(tmp_path)
    tok = BertTokenizerFast(vocab_file=str(vocab_path), do_lower_case=False)
    tok.save_pretrained(saved)

    (saved / "labels.json").write_text(
        json.dumps(
            {"sequence": ["valid", "invalid"], "category": ["none", "missing_dd"], "line": ["no_error", "error"]},
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return saved


def test_package_jcl_creates_manifest_and_passes_verify(tmp_path: Path) -> None:
    ckpt = _make_jcl_checkpoint(tmp_path)
    pkg = tmp_path / "models" / "jcl-test"
    result = package_jcl(
        ckpt,
        pkg,
        model_id="jcl-test:v1",
        max_input_tokens=64,
        expected_latency_ms=10,
    )
    assert (result.pkg_dir / "manifest.json").exists()
    assert result.manifest.task == "jcl_validation"
    assert "model.safetensors" in result.manifest.sha256
    assert "flow_heads.safetensors" in result.manifest.sha256
    assert result.manifest.labels_file == "labels.json"

    verify = verify_package(result.pkg_dir)
    assert verify.ok, f"verify failed: issues={verify.issues}"
    assert verify.forward_ok


def test_package_jcl_missing_file_fails(tmp_path: Path) -> None:
    ckpt = _make_jcl_checkpoint(tmp_path)
    (ckpt / "labels.json").unlink()
    with pytest.raises(FileNotFoundError):
        package_jcl(ckpt, tmp_path / "models" / "broken")


def test_package_emits_fm_archive_and_verify_accepts_it(tmp_path: Path) -> None:
    """package_jcl writes an unpacked dir AND a `.fm` zip; verify_package handles both."""
    import zipfile

    ckpt = _make_jcl_checkpoint(tmp_path)
    pkg = tmp_path / "models" / "jcl-fm"
    result = package_jcl(
        ckpt, pkg,
        model_id="jcl-test:v1",
        max_input_tokens=64,
        expected_latency_ms=10,
    )
    assert result.fm_path is not None
    assert result.fm_path.exists()
    assert result.fm_path.suffix == ".fm"
    # The .fm contains exactly the manifest + sha256 keys
    with zipfile.ZipFile(result.fm_path) as zf:
        files = set(zf.namelist())
    assert "manifest.json" in files
    for required in result.manifest.sha256:
        assert required in files, f"{required} missing from .fm"
    # verify_package also accepts the .fm directly
    verify = verify_package(result.fm_path)
    assert verify.ok, f"verify failed on .fm: issues={verify.issues}"


def test_verify_detects_sha_mismatch(tmp_path: Path) -> None:
    ckpt = _make_jcl_checkpoint(tmp_path)
    pkg = tmp_path / "models" / "jcl-tamper"
    package_jcl(ckpt, pkg, model_id="jcl-test:v1", max_input_tokens=64, expected_latency_ms=10)
    (pkg / "labels.json").write_text("{}", encoding="utf-8")  # tamper after manifest write
    result = verify_package(pkg)
    assert not result.ok
    assert any("sha256 mismatch" in s for s in result.issues)


# ---------------------------------------------------------------------------
# .fm archive safety tests
#
# These cover the validate-before-extract guarantees in
# flow_ml.packaging.package_model: read manifest directly from a .fm without
# extraction, reject path traversal / oversize / symlink archives, and only
# extract the files listed in manifest.sha256 (stray entries are ignored).
# ---------------------------------------------------------------------------


def _build_minimal_fm(fm_path: Path, *, manifest: dict, files: dict[str, bytes]) -> Path:
    """Hand-build a .fm zip with a given manifest dict and named entries."""
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
    """Minimal manifest schema that satisfies ModelManifest validators."""
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
    """read_manifest_from_fm() should return a ModelManifest without extracting."""
    from flow_ml.packaging.package_model import read_manifest_from_fm

    weights = b"x" * 16
    fm = _build_minimal_fm(
        tmp_path / "ok.fm",
        manifest=_valid_fake_manifest(weights),
        files={"model.safetensors": weights},
    )
    # Snapshot tempdir state before / after to confirm nothing was written
    before = set((tmp_path).iterdir())
    md = read_manifest_from_fm(fm)
    after = set((tmp_path).iterdir())
    assert md.model_id == "fake:v1"
    assert md.task == "spool_interpretation"
    assert "model.safetensors" in md.sha256
    assert before == after, "read_manifest_from_fm should not write to disk"


def test_verify_rejects_path_traversal(tmp_path: Path) -> None:
    """A .fm whose entry name contains '..' must be rejected before any write."""
    weights = b"x" * 16
    fm = _build_minimal_fm(
        tmp_path / "evil.fm",
        manifest=_valid_fake_manifest(weights),
        files={"model.safetensors": weights, "../escape.txt": b"evil"},
    )
    result = verify_package(fm)
    assert not result.ok
    assert any("'..'" in i or "traversal" in i.lower() or "archive validation" in i.lower()
               for i in result.issues), result.issues


def test_verify_rejects_too_many_entries(tmp_path: Path) -> None:
    """A .fm with more than MAX_ENTRIES files must be rejected."""
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


def test_verify_only_extracts_manifest_listed_files(tmp_path: Path) -> None:
    """A stray entry not listed in manifest.sha256 must not appear in the tempdir."""
    import tempfile
    import zipfile

    from flow_ml.packaging import package_model as pm

    # Real packaging flow + a hand-injected stray file in the .fm
    ckpt = _make_jcl_checkpoint(tmp_path)
    pkg = tmp_path / "models" / "jcl-fm-stray"
    result = package_jcl(
        ckpt, pkg,
        model_id="jcl-test:v1",
        max_input_tokens=64,
        expected_latency_ms=10,
    )
    assert result.fm_path is not None

    # Repack the .fm with an extra "intruder.txt" alongside the legit files.
    intruded = tmp_path / "intruded.fm"
    with zipfile.ZipFile(result.fm_path, "r") as src, \
         zipfile.ZipFile(intruded, "w", compression=zipfile.ZIP_DEFLATED) as dst:
        for name in src.namelist():
            dst.writestr(name, src.read(name))
        dst.writestr("intruder.txt", b"i should not be extracted")

    # Spy on extracted contents by capturing the tempdir name before cleanup.
    captured: dict[str, list[str]] = {}
    real_TD = tempfile.TemporaryDirectory

    class _Spy(real_TD):  # type: ignore[misc]
        def __init__(self, *a, **kw):
            super().__init__(*a, **kw)
            captured["names"] = []
            captured["dir"] = self.name
        def cleanup(self):
            try:
                captured["names"] = [
                    str(p.relative_to(self.name))
                    for p in Path(self.name).rglob("*")
                    if p.is_file()
                ]
            finally:
                super().cleanup()

    monkey_target = pm.tempfile
    monkey_target.TemporaryDirectory = _Spy  # type: ignore[assignment]
    try:
        verify = verify_package(intruded)
    finally:
        monkey_target.TemporaryDirectory = real_TD  # type: ignore[assignment]

    extracted = set(captured.get("names", []))
    assert "intruder.txt" not in extracted, (
        f"stray entry should be ignored; tempdir contained {extracted}"
    )
    # The manifest-listed files must be present
    for name in result.manifest.sha256:
        assert name in extracted, f"{name} should have been extracted; got {extracted}"
    # And the verify path itself should still pass on the legit content
    assert verify.ok, f"verify failed: {verify.issues}"


def test_package_spool_requires_prompt_spec(tmp_path: Path) -> None:
    fake_ckpt = tmp_path / "spool-ckpt"
    fake_ckpt.mkdir()
    (fake_ckpt / "model.safetensors").write_bytes(b"\x00")
    (fake_ckpt / "config.json").write_text("{}", encoding="utf-8")
    (fake_ckpt / "tokenizer.json").write_text("{}", encoding="utf-8")
    with pytest.raises(FileNotFoundError):
        package_spool(fake_ckpt, tmp_path / "models" / "spool-test")
