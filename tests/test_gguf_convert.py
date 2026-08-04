"""S4: gguf convert script must be explicit-only (no PATH/cwd search)."""

from __future__ import annotations

from pathlib import Path

import pytest

from maatml.config import ModelDefinition
from maatml.export.gguf import _find_convert_script


def _md(tmp_path: Path, *, extensions=None) -> ModelDefinition:
    md = ModelDefinition(name="g", model_id="g", version="0.1.0", extensions=extensions or {})
    object.__setattr__(md, "model_dir", tmp_path)
    return md


def test_no_config_returns_none(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("MAATML_LLAMA_CONVERT", raising=False)
    assert _find_convert_script(_md(tmp_path)) is None


def test_env_var_existing_file(tmp_path: Path, monkeypatch) -> None:
    script = tmp_path / "conv.py"
    script.write_text("# x", encoding="utf-8")
    monkeypatch.setenv("MAATML_LLAMA_CONVERT", str(script))
    assert _find_convert_script(_md(tmp_path)) == script.resolve()


def test_env_var_missing_file_raises(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("MAATML_LLAMA_CONVERT", str(tmp_path / "nope.py"))
    with pytest.raises(FileNotFoundError):
        _find_convert_script(_md(tmp_path))


def test_extensions_config_resolves_against_model_dir(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("MAATML_LLAMA_CONVERT", raising=False)
    (tmp_path / "conv.py").write_text("# x", encoding="utf-8")
    md = _md(tmp_path, extensions={"gguf": {"convert_script": "conv.py"}})
    assert _find_convert_script(md) == (tmp_path / "conv.py").resolve()


def test_path_and_cwd_not_searched(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("MAATML_LLAMA_CONVERT", raising=False)
    (tmp_path / "convert.py").write_text("# x", encoding="utf-8")
    monkeypatch.setenv("PATH", str(tmp_path))
    monkeypatch.chdir(tmp_path)
    assert _find_convert_script(_md(tmp_path)) is None


def test_quantize_config_absent_means_no_quantization(tmp_path: Path, monkeypatch) -> None:
    from maatml.export.gguf import _quantize_config

    monkeypatch.delenv("MAATML_LLAMA_QUANTIZE", raising=False)
    binary, levels = _quantize_config(_md(tmp_path))
    assert binary is None and levels == []


def test_quantize_produces_one_file_per_level(tmp_path: Path, monkeypatch) -> None:
    from maatml.export.gguf import _quantize

    monkeypatch.delenv("MAATML_LLAMA_QUANTIZE", raising=False)
    binary = tmp_path / "llama-quantize"
    # A stand-in that copies input to output, recording the level.
    binary.write_text('#!/bin/sh\ncp "$1" "$2"\necho "$3" >> "$2"\n', encoding="utf-8")
    binary.chmod(0o755)
    md = _md(
        tmp_path,
        extensions={"gguf": {"quantize_binary": str(binary), "quant_levels": ["Q4_K_M", "Q5_K_M"]}},
    )
    source = tmp_path / "g.gguf"
    source.write_bytes(b"gguf")
    produced = _quantize(md, source, tmp_path)
    assert [p.name for p in produced] == ["g-Q4_K_M.gguf", "g-Q5_K_M.gguf"]
    assert all(p.is_file() for p in produced)


def test_quantize_failure_is_loud(tmp_path: Path, monkeypatch) -> None:
    from maatml.export.gguf import _quantize

    monkeypatch.delenv("MAATML_LLAMA_QUANTIZE", raising=False)
    binary = tmp_path / "llama-quantize"
    binary.write_text('#!/bin/sh\necho "boom" >&2\nexit 1\n', encoding="utf-8")
    binary.chmod(0o755)
    md = _md(tmp_path, extensions={"gguf": {"quantize_binary": str(binary)}})
    source = tmp_path / "g.gguf"
    source.write_bytes(b"gguf")
    with pytest.raises(RuntimeError, match="boom"):
        _quantize(md, source, tmp_path)


def test_quantize_missing_binary_raises(tmp_path: Path, monkeypatch) -> None:
    from maatml.export.gguf import _quantize_config

    monkeypatch.delenv("MAATML_LLAMA_QUANTIZE", raising=False)
    md = _md(tmp_path, extensions={"gguf": {"quantize_binary": "nope"}})
    with pytest.raises(FileNotFoundError):
        _quantize_config(md)
