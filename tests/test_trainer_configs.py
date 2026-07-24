"""Config parsing and gold-label mapping for the trainers (no torch needed).

seq2seq / multi_head keep their torch imports inside functions, so their
config surface is unit-testable on the torch-free matrix. The tokenization
and label-masking tests live in test_trainers_torch.py.
"""
from __future__ import annotations

import pytest

from maatml.data.preference import as_completion_text, normalize_preference
from maatml.training.multi_head import (
    HeadSpec,
    MultiHeadConfig,
    UnknownLabelError,
    _label_index,
    parse_heads,
    scan_label_coverage,
)
from maatml.training.seq2seq import (
    Seq2SeqConfig,
    _drop_targetless,
    _serialise_target,
    has_target,
)
from maatml.training.sft_config import SFTTrainConfig, validate_precision


# --- precision -------------------------------------------------------------


def test_precision_validated_at_parse_time() -> None:
    assert validate_precision("bf16") == "bf16"
    for cfg_cls, kwargs in (
        (Seq2SeqConfig.from_dict, {"precision": "bfloat16"}),
        (MultiHeadConfig.from_dict, {"precision": "float16"}),
    ):
        with pytest.raises(ValueError, match="training.precision must be one of"):
            cfg_cls(kwargs)
    with pytest.raises(ValueError, match="training.precision must be one of"):
        SFTTrainConfig(precision="fp8")


# --- fractional epochs -----------------------------------------------------


def test_fractional_epochs_survive_config_parse() -> None:
    assert Seq2SeqConfig.from_dict({"epochs": 0.5}).epochs == 0.5
    assert MultiHeadConfig.from_dict(
        {"epochs": 0.25, "heads": [{"name": "h", "labels": ["a", "b"]}]}
    ).epochs == 0.25
    # Parity with the SFT config, which already modelled epochs as a float.
    assert SFTTrainConfig(epochs=0.5).epochs == 0.5


# --- multi_head heads ------------------------------------------------------


def test_absent_or_malformed_heads_is_an_error() -> None:
    for training in ({}, {"heads": []}, {"heads": "validity"}, {"heads": {}}):
        with pytest.raises(ValueError, match="training.heads must be"):
            parse_heads(training)


def test_legacy_jcl_head_shape_still_parses_when_its_keys_are_present() -> None:
    heads = parse_heads({"heads": {"error_codes": ["missing_dd", "none"]}})
    assert [h.name for h in heads] == ["validity", "error_code", "severity", "line"]
    assert parse_heads({"head_loss_weights": {"validity": 2.0}})[0].loss_weight == 2.0


def test_bool_gold_honours_declared_label_order() -> None:
    assert _label_index(True, ["invalid", "valid"]) == 1
    assert _label_index(False, ["invalid", "valid"]) == 0
    # Reversed declaration: True must still mean "valid", not index 1.
    assert _label_index(True, ["valid", "invalid"]) == 0
    assert _label_index(False, ["valid", "invalid"]) == 1
    # Two labels with no boolean-ish names fall back to False→0, True→1.
    assert _label_index(True, ["off", "on"]) == 1


def test_unknown_gold_label_raises_instead_of_mapping_to_none() -> None:
    with pytest.raises(UnknownLabelError):
        _label_index("not_a_declared_code", ["missing_dd", "none"])
    with pytest.raises(UnknownLabelError):
        _label_index(None, ["a", "b"])
    assert _label_index(None, ["a", "none"]) == 1


def test_scan_label_coverage_counts_unknown_gold() -> None:
    heads = [
        HeadSpec(name="code", kind="classification", labels=["a", "none"], target_path="code"),
        HeadSpec(name="line", kind="line_pointer", target_path="line"),
    ]
    rows = [
        {"target": {"code": "a"}},
        {"target": {"code": "zzz"}},
        {"target": {"code": "zzz"}},
    ]
    assert scan_label_coverage(rows, heads, target_field="target") == {
        "code": {"'zzz'": 2}
    }
    assert scan_label_coverage(rows[:1], heads, target_field="target") == {}


# --- seq2seq targets -------------------------------------------------------


def test_falsy_targets_are_dropped_not_serialised_to_braces() -> None:
    rows = [
        {"target": {"a": 1}},
        {"target": {}},
        {"target": None},
        {"target": ""},
        {},
    ]
    kept, dropped = _drop_targetless(rows, "target")
    assert kept == [{"target": {"a": 1}}]
    assert dropped == 4
    assert has_target({"target": "text"}, "target") is True
    with pytest.raises(ValueError, match="target is empty"):
        _serialise_target({})


def test_serialise_target_honours_key_order() -> None:
    assert _serialise_target({"b": 2, "a": 1}, key_order=["a", "b"]) == '{"a":1,"b":2}'


# --- preference rows -------------------------------------------------------


def test_structured_completions_serialise_as_json_not_repr() -> None:
    assert as_completion_text({"a": 1, "b": None}) == '{"a":1,"b":null}'
    assert as_completion_text(["x"]) == '["x"]'
    assert as_completion_text("already text") == "already text"
    row = normalize_preference(
        {"prompt": "p", "chosen": {"ok": True}, "rejected": {"ok": False}}
    )
    assert row["chosen"] == '{"ok":true}'
    assert "'" not in row["chosen"]


def test_identical_chosen_and_rejected_warns() -> None:
    with pytest.warns(UserWarning, match="identical chosen and rejected"):
        normalize_preference({"prompt": "p", "chosen": "same", "rejected": "same"})
