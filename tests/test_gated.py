"""Shared gated corpus builder."""
from __future__ import annotations

from maatml.data.gated import build_gated_corpus


def test_gated_corpus_accept_reject() -> None:
    counter = {"n": 0}

    def generate():
        counter["n"] += 1
        return {"id": counter["n"], "ok": counter["n"] % 2 == 0}

    def validate(row):
        return bool(row.get("ok"))

    accepted, rejected = build_gated_corpus(
        generate_fn=generate,
        validate_fn=validate,
        target_n=3,
        max_attempts=20,
    )
    assert len(accepted) == 3
    assert all(r["ok"] for r in accepted)
    assert rejected
    assert all(not r["ok"] for r in rejected)


def test_gated_corpus_respects_max_attempts() -> None:
    def generate():
        return {"ok": False}

    accepted, rejected = build_gated_corpus(
        generate_fn=generate,
        validate_fn=lambda r: False,
        target_n=10,
        max_attempts=5,
    )
    assert accepted == []
    assert len(rejected) == 5


def test_gated_corpus_skips_none() -> None:
    state = {"n": 0}

    def generate():
        state["n"] += 1
        if state["n"] < 3:
            return None
        return {"v": state["n"]}

    accepted, rejected = build_gated_corpus(
        generate_fn=generate,
        validate_fn=lambda r: True,
        target_n=1,
        max_attempts=10,
    )
    assert len(accepted) == 1
    assert rejected == []
