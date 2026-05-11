"""Custom JCL tokenizer for the v2 ModernBERT classifier.

Two stages — see `COLUMN_RULES.md` for the normative spec both this
module and flow-studio's Rust `BertClassifierBackend` implement:

1. `pre_tokenize_jcl(text)` — column-aware pre-tokenizer that strips
   columns 73+, emits `<COL1>` line-start markers, preserves continuation
   markers, drops blank lines.
2. `train_jcl_tokenizer(corpus_path, out_path)` — one-shot BPE training
   using the `tokenizers` library. Pass a JSONL corpus with `request`
   fields; outputs a HuggingFace `tokenizer.json` for the classifier.

The trained tokenizer can be loaded back via standard
`transformers.AutoTokenizer.from_pretrained(...)` for the training-time
`JclDataset`.

CLI:

    python -m flow_ml.tokenization.jcl_tokenizer train \
        --corpus models/jcl-validator/datasets/samples/tokenizer_corpus.jsonl \
        --out models/jcl-validator/datasets/tokenizer.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterator

SPECIAL_TOKENS = ["<PAD>", "<UNK>", "<CLS>", "<SEP>", "<MASK>", "<COL1>", "<CONT>"]


def pre_tokenize_jcl(text: str) -> str:
    """Apply the seven column rules from `COLUMN_RULES.md` to raw JCL.

    Returns a string where each non-empty line is prefixed with `<COL1> `
    and suffixed with ` <CONT>` if column 72 was non-blank. Columns 73+
    are stripped. Tabs are expanded to 4 spaces. Blank lines are dropped.
    """
    lines = text.replace("\r\n", "\n").split("\n")
    out_lines: list[str] = []
    for raw_line in lines:
        line = raw_line.expandtabs(4)
        if not line.strip():
            continue  # Rule 5: drop blank lines
        # Rule 3: detect column-72 continuation marker (1-based col 72 = index 71).
        cont = len(line) > 71 and line[71] != " "
        # Rule 2: strip cols 73+.
        if len(line) > 72:
            line = line[:72]
        # Rule 4: prepend <COL1>; Rule 3: append <CONT> if needed.
        if cont:
            out_lines.append(f"<COL1> {line} <CONT>")
        else:
            out_lines.append(f"<COL1> {line}")
    return "\n".join(out_lines)


def _iter_corpus_requests(corpus_path: Path) -> Iterator[str]:
    """Yield the `request` field of each JSONL row, pre-tokenized."""
    with corpus_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            req = row.get("request")
            if isinstance(req, str) and req:
                yield pre_tokenize_jcl(req)


def train_jcl_tokenizer(
    corpus_path: Path,
    out_path: Path,
    vocab_size: int = 30_000,
) -> None:
    """Train a BPE tokenizer on the corpus (already-pre-tokenized text)
    and save to `out_path` in the HuggingFace `tokenizer.json` format.

    The tokenizer's `pre_tokenizer` is `Whitespace` because we've already
    applied the column-aware pre-tokenization upstream. The BPE then
    learns subword units across JCL keywords (DD, EXEC, PGM, …) and the
    bounded set of dataset-name fragments in the corpus.
    """
    from tokenizers import Tokenizer, models, pre_tokenizers, trainers
    from tokenizers.processors import TemplateProcessing

    tokenizer = Tokenizer(models.BPE(unk_token="<UNK>"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()

    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=SPECIAL_TOKENS,
        show_progress=True,
    )
    tokenizer.train_from_iterator(_iter_corpus_requests(corpus_path), trainer=trainer)

    # BERT-style template so the tokenizer plays well with
    # `transformers.AutoTokenizer.from_pretrained(...)` downstream.
    cls_id = tokenizer.token_to_id("<CLS>")
    sep_id = tokenizer.token_to_id("<SEP>")
    if cls_id is None or sep_id is None:
        raise RuntimeError("special tokens missing after training")
    tokenizer.post_processor = TemplateProcessing(
        single="<CLS> $A <SEP>",
        pair="<CLS> $A <SEP> $B:1 <SEP>:1",
        special_tokens=[("<CLS>", cls_id), ("<SEP>", sep_id)],
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    tokenizer.save(str(out_path))
    print(f"saved tokenizer to {out_path} (vocab={tokenizer.get_vocab_size()})")


def _cli(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="JCL pre-tokenizer + BPE trainer."
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_train = sub.add_parser("train", help="Train a JCL BPE tokenizer.")
    p_train.add_argument(
        "--corpus", required=True, help="Path to a JSONL corpus with `request` fields."
    )
    p_train.add_argument(
        "--out", required=True, help="Output path for the HuggingFace tokenizer.json."
    )
    p_train.add_argument(
        "--vocab-size", type=int, default=30_000, help="BPE vocab size (default 30000)."
    )

    p_pre = sub.add_parser(
        "pre-tokenize",
        help="Apply the column rules to a JCL string read from stdin.",
    )

    args = parser.parse_args(argv)

    if args.cmd == "train":
        train_jcl_tokenizer(Path(args.corpus), Path(args.out), args.vocab_size)
        return 0
    if args.cmd == "pre-tokenize":
        text = sys.stdin.read()
        print(pre_tokenize_jcl(text))
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(_cli())
