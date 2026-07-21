"""Tests for training guards (tokenizer contract + run metadata)."""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from maatml.config import ModelDefinition
from maatml.training.guards import ensure_tokenizer_model_contract, write_run_metadata


class _FakeEmb:
    def __init__(self, n: int) -> None:
        self.weight = SimpleNamespace(shape=(n, 8))


class _FakeModel:
    def __init__(self, vocab: int) -> None:
        self._vocab = vocab
        self.resized_to: int | None = None
        self.config = SimpleNamespace(vocab_size=vocab)

    def get_input_embeddings(self):
        return _FakeEmb(self._vocab)

    def resize_token_embeddings(self, n: int) -> None:
        self.resized_to = n
        self._vocab = n


def test_tokenizer_contract_matching_sizes_noop() -> None:
    model = _FakeModel(100)
    tok = list(range(100))
    ensure_tokenizer_model_contract(model, tok, embedding_strategy=None)
    assert model.resized_to is None


def test_tokenizer_contract_requires_strategy_on_mismatch() -> None:
    model = _FakeModel(100)
    tok = list(range(80))
    with pytest.raises(ValueError, match="embedding_strategy"):
        ensure_tokenizer_model_contract(model, tok, embedding_strategy=None)


def test_tokenizer_contract_resize() -> None:
    model = _FakeModel(100)
    tok = list(range(120))
    ensure_tokenizer_model_contract(model, tok, embedding_strategy="resize")
    assert model.resized_to == 120


def test_tokenizer_contract_reuse_rejects_larger_tokenizer() -> None:
    model = _FakeModel(100)
    tok = list(range(120))
    with pytest.raises(ValueError, match="reuse"):
        ensure_tokenizer_model_contract(model, tok, embedding_strategy="reuse")


def test_write_run_metadata(tmp_path: Path) -> None:
    seed = tmp_path / "train.jsonl"
    seed.write_text('{"sample_id":"a"}\n', encoding="utf-8")
    md = ModelDefinition(
        name="demo",
        model_id="demo",
        architecture="causal_sft",
        version="0.1.0",
    )
    object.__setattr__(md, "model_dir", tmp_path)
    out = tmp_path / "ckpt"
    path = write_run_metadata(out, md, {"train": seed})
    assert path.is_file()
    text = path.read_text(encoding="utf-8")
    assert "demo@0.1.0" in text
    assert "spec_hash" in text
