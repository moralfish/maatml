"""serving.json contract for the seq2seq MLX serving bundle (no torch required)."""

from __future__ import annotations

import json
from pathlib import Path

from maatml.config import load_model_def
from maatml.export.mlx_export import build_seq2seq_serving

EXAMPLE_ROOT = Path(__file__).resolve().parents[1]


def test_serving_config_matches_contract(tmp_path: Path) -> None:
    md = load_model_def(EXAMPLE_ROOT, load_plugins=False)
    ckpt = tmp_path / "ckpt"
    ckpt.mkdir()
    (ckpt / "config.json").write_text(
        json.dumps({"decoder_start_token_id": 0, "eos_token_id": 1}),
        encoding="utf-8",
    )
    serving = build_seq2seq_serving(md, ckpt)
    assert serving == {
        "contract": "maatml.serving/1",
        "kind": "seq2seq_lm",
        "max_input_tokens": 1024,
        "max_new_tokens": 512,
        "decoder_start_token_id": 0,
        "eos_token_id": 1,
        "input_prefix": "interpret spool: ",
        "output_note": (
            "emits the JSON object body without outer braces; T5 vocab maps "
            "{ } to unk. Clients re-add the braces before parsing."
        ),
    }
