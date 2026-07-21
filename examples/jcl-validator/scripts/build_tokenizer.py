#!/usr/bin/env python3
"""Train the custom JCL BPE tokenizer for this example.

Wraps ``jcl_plugin.tokenizer`` train CLI with example-relative defaults.

Usage:
    python examples/jcl-validator/scripts/build_tokenizer.py
    python examples/jcl-validator/scripts/build_tokenizer.py \\
        --corpus datasets/samples/tokenizer_corpus.jsonl \\
        --out datasets/tokenizer.json
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

EXAMPLE_ROOT = Path(__file__).resolve().parents[1]
REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(EXAMPLE_ROOT))

from jcl_plugin.tokenizer import train_jcl_tokenizer  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train the JCL BPE tokenizer.")
    parser.add_argument(
        "--corpus",
        type=Path,
        default=EXAMPLE_ROOT / "datasets" / "samples" / "tokenizer_corpus.jsonl",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=EXAMPLE_ROOT / "datasets" / "tokenizer.json",
    )
    parser.add_argument("--vocab-size", type=int, default=30_000)
    args = parser.parse_args(argv)
    corpus = args.corpus if args.corpus.is_absolute() else EXAMPLE_ROOT / args.corpus
    out = args.out if args.out.is_absolute() else EXAMPLE_ROOT / args.out
    train_jcl_tokenizer(corpus, out, args.vocab_size)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
