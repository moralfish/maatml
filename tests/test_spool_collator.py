from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from flow_ml.training.spool_interpreter import (  # noqa: E402
    SpoolDataCollator,
    build_chat_example,
    render_target,
)


class _FakeTok:
    pad_token_id = 0
    eos_token_id = 999

    def apply_chat_template(self, messages, *, add_generation_prompt, tokenize, return_tensors=None):
        out: list[int] = []
        for m in messages:
            out.append({"system": 1, "user": 2, "assistant": 3}[m["role"]])
            for ch in m["content"]:
                out.append(ord(ch) % 800 + 4)
        if add_generation_prompt:
            out.append(3)
        return out

    def __call__(self, text, *, add_special_tokens):
        return {"input_ids": [ord(c) % 800 + 4 for c in text]}


def _spec() -> dict:
    return {
        "system": "SYS",
        "user_template": "<<SANITIZED_SPOOL>>",
        "json_keys_order": ["summary", "status", "returnCode", "rootCause", "suggestedFix", "confidence"],
    }


def _sample() -> dict:
    return {
        "sanitized_spool": "JOB ENDED",
        "status": "failed",
        "return_code": "0008",
        "root_cause": "missing dataset",
        "suggested_fix": "check catalog",
    }


def test_render_target_uses_keys_order() -> None:
    s = render_target(_sample(), _spec())
    assert s.startswith('{"summary":')
    assert '"confidence": 1.0' in s


def test_chat_example_masks_prompt_loss() -> None:
    tok = _FakeTok()
    ex = build_chat_example(_sample(), _spec(), tok, max_length=4096)
    assert len(ex["input_ids"]) == len(ex["labels"])
    # Prompt portion is -100; suffix matches input_ids.
    assert ex["labels"][0] == -100
    last = ex["labels"][-1]
    assert last != -100
    assert last == ex["input_ids"][-1] == tok.eos_token_id


def test_collator_pads_and_aligns_labels() -> None:
    tok = _FakeTok()
    coll = SpoolDataCollator(tok, _spec(), max_length=512)
    batch = [_sample(), _sample() | {"sanitized_spool": "JOB ENDED LONGER MESSAGE FOR PADDING"}]
    out = coll(batch)
    B, T = out["input_ids"].shape
    assert B == 2
    assert out["attention_mask"].shape == (B, T)
    assert out["labels"].shape == (B, T)
    # Padding positions must be masked out.
    for i in range(B):
        attn = out["attention_mask"][i].tolist()
        labels = out["labels"][i].tolist()
        for t, m in enumerate(attn):
            if m == 0:
                assert labels[t] == -100
