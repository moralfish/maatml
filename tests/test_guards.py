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


class _FakeTokenizer(list):
    """A tokenizer that knows its own special-token ids."""

    def __init__(self, n: int, **ids: int | None) -> None:
        super().__init__(range(n))
        for key, value in ids.items():
            setattr(self, key, value)


def test_resize_realigns_special_token_ids_into_the_new_vocab() -> None:
    """Shrinking a vocab must not leave token ids pointing past the embeddings.

    ModernBERT plus a 343-token custom JCL tokenizer produced a checkpoint with
    vocab_size=343 and pad_token_id=50283. Training passed, and every reload
    (evaluate / export / serve) died on `Padding_idx must be within
    num_embeddings`.
    """
    model = _FakeModel(50368)
    model.config = SimpleNamespace(
        vocab_size=50368, pad_token_id=50283, cls_token_id=50281, sep_token_id=50282
    )
    tok = _FakeTokenizer(343, pad_token_id=1, cls_token_id=2, sep_token_id=3)

    ensure_tokenizer_model_contract(model, tok, embedding_strategy="resize")

    assert model.resized_to == 343
    for field in ("pad_token_id", "cls_token_id", "sep_token_id"):
        value = getattr(model.config, field)
        assert value is not None and value < 343, f"{field} still out of range: {value}"


def test_resize_clears_ids_the_tokenizer_cannot_supply() -> None:
    model = _FakeModel(50368)
    model.config = SimpleNamespace(vocab_size=50368, bos_token_id=50281, pad_token_id=50283)
    tok = _FakeTokenizer(343, pad_token_id=1)  # no bos

    ensure_tokenizer_model_contract(model, tok, embedding_strategy="resize")

    assert model.config.pad_token_id == 1
    assert model.config.bos_token_id is None


def test_resize_clears_pad_when_the_tokenizer_has_none() -> None:
    """No pad token is survivable; an out-of-range one is not.

    ``nn.Embedding`` accepts ``padding_idx=None``, so clearing the id keeps the
    checkpoint reloadable. If padding is genuinely needed, the failure then
    comes from the collator naming the missing pad token, which is a far more
    actionable message than an embedding assertion.
    """
    model = _FakeModel(50368)
    model.config = SimpleNamespace(vocab_size=50368, pad_token_id=50283)
    tok = _FakeTokenizer(343)  # tokenizer defines nothing

    ensure_tokenizer_model_contract(model, tok, embedding_strategy="resize")

    assert model.config.pad_token_id is None


class _FakeTensor:
    """Stand-in for an embedding weight: a shape plus an identity."""

    def __init__(self, ptr: int, rows: int) -> None:
        self._ptr = ptr
        self.shape = (rows, 8)

    def data_ptr(self) -> int:
        return self._ptr


class _ModelWithUntiedHead(_FakeModel):
    """A model whose output projection is trained separately (flan-t5 shape)."""

    def __init__(self, vocab: int, *, tied: bool) -> None:
        super().__init__(vocab)
        self._in = SimpleNamespace(weight=_FakeTensor(1, vocab))
        self._out = SimpleNamespace(weight=_FakeTensor(1 if tied else 2, vocab))

    def get_input_embeddings(self):
        return self._in

    def get_output_embeddings(self):
        return self._out

    def resize_token_embeddings(self, n: int) -> None:
        super().resize_token_embeddings(n)
        self._in = SimpleNamespace(weight=_FakeTensor(1, n))
        self._out = SimpleNamespace(weight=_FakeTensor(1, n))  # resize ties them


def test_resize_refuses_to_discard_an_untied_output_head() -> None:
    """Shrinking flan-t5 re-ties lm_head to the embeddings and destroys it.

    Initial loss on flan-t5-base goes from 3.5 to 355 while training still
    reports success, so the damage only surfaces as degenerate generation at
    eval. `reuse` is always available here, since the embedding matrix is
    already larger than the tokenizer needs.
    """
    model = _ModelWithUntiedHead(32128, tied=False)
    tok = _FakeTokenizer(32100, pad_token_id=0)

    with pytest.raises(ValueError, match="reuse"):
        ensure_tokenizer_model_contract(model, tok, embedding_strategy="resize")

    assert model.resized_to is None, "the resize must not have happened"


def test_resize_allowed_when_the_head_is_already_tied() -> None:
    """A tied head has nothing separate to lose, so the shrink is safe."""
    model = _ModelWithUntiedHead(50368, tied=True)
    tok = _FakeTokenizer(343, pad_token_id=1)

    ensure_tokenizer_model_contract(model, tok, embedding_strategy="resize")

    assert model.resized_to == 343


def test_resize_allowed_when_growing_the_vocabulary() -> None:
    """Growing adds rows; it never discards a trained head."""
    model = _ModelWithUntiedHead(32000, tied=False)
    tok = _FakeTokenizer(32100, pad_token_id=0)

    ensure_tokenizer_model_contract(model, tok, embedding_strategy="resize")

    assert model.resized_to == 32100


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
