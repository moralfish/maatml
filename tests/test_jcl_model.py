from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from transformers import BertConfig, BertModel  # noqa: E402

from flow_ml.training.jcl_validator import JclMultiHeadModel, NUM_CATEGORIES  # noqa: E402


def _tiny_encoder() -> BertModel:
    cfg = BertConfig(
        vocab_size=64,
        hidden_size=8,
        num_hidden_layers=1,
        num_attention_heads=1,
        intermediate_size=16,
        max_position_embeddings=64,
    )
    return BertModel(cfg)


def test_multi_head_forward_returns_logits() -> None:
    model = JclMultiHeadModel(_tiny_encoder(), num_categories=NUM_CATEGORIES)
    input_ids = torch.randint(0, 64, (2, 16))
    attention_mask = torch.ones_like(input_ids)
    out = model(input_ids=input_ids, attention_mask=attention_mask)
    assert out["seq_logits"].shape == (2, 2)
    assert out["cat_logits"].shape == (2, NUM_CATEGORIES)
    assert out["line_logits"].shape == (2, 16, 2)
    assert "loss" not in out


def test_multi_head_forward_with_labels_returns_scalar_loss() -> None:
    model = JclMultiHeadModel(_tiny_encoder(), num_categories=NUM_CATEGORIES)
    input_ids = torch.randint(0, 64, (2, 16))
    attention_mask = torch.ones_like(input_ids)
    seq_label = torch.tensor([0, 1])
    cat_label = torch.tensor([0, 3])
    line_labels = torch.zeros((2, 16), dtype=torch.long)
    line_labels[1, 5] = 1
    out = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        seq_label=seq_label,
        cat_label=cat_label,
        line_labels=line_labels,
    )
    assert out["loss"].dim() == 0
    assert out["loss"].requires_grad
    out["loss"].backward()


def test_save_and_load_round_trip(tmp_path) -> None:
    model = JclMultiHeadModel(_tiny_encoder(), num_categories=NUM_CATEGORIES)
    saved = model.save(tmp_path, base_model_id="dummy/tiny")
    assert (saved / "flow_heads.safetensors").exists()
    assert (saved / "flow_metadata.json").exists()
    loaded = JclMultiHeadModel.load(saved)
    assert loaded.num_categories == NUM_CATEGORIES
    for k in model.seq_head.state_dict().keys():
        assert torch.equal(loaded.seq_head.state_dict()[k], model.seq_head.state_dict()[k])
    for k in model.line_head.state_dict().keys():
        assert torch.equal(loaded.line_head.state_dict()[k], model.line_head.state_dict()[k])
