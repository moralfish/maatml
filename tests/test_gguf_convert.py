"""S4: gguf convert script must be explicit-only (no PATH/cwd search)."""
from __future__ import annotations

from pathlib import Path

import pytest

from maatml.config import ModelDefinition
from maatml.export.gguf import _find_convert_script


def _md(tmp_path: Path, *, extensions=None) -> ModelDefinition:
    md = ModelDefinition(
        name="g", model_id="g", version="0.1.0", extensions=extensions or {}
    )
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
