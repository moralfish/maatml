"""Special-token resolution shared by the trainer and the predictor.

Both sides must bind the same special tokens or training and evaluation
tokenize differently. They previously kept separate hard-coded copies that had
drifted: the trainer rebound ``pad_token`` to whatever the tokenizer file
declared, while the predictor always kept the built-in ``<PAD>``. A model whose
custom tokenizer names its pad token ``[PAD]`` therefore trained with one pad
token and was evaluated with another.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

# Built-in names, used when a tokenizer file declares nothing better.
DEFAULT_SPECIAL_TOKENS: dict[str, Any] = {
    "pad_token": "<PAD>",
    "unk_token": "<UNK>",
    "cls_token": "<CLS>",
    "sep_token": "<SEP>",
    "mask_token": "<MASK>",
    "additional_special_tokens": ["<COL1>", "<CONT>"],
}


def resolve_special_tokens(tokenizer_path: Optional[Path | str]) -> dict[str, Any]:
    """Special tokens for ``PreTrainedTokenizerFast``, read from tokenizer.json.

    Falls back to :data:`DEFAULT_SPECIAL_TOKENS` for any slot the file does not
    name. Unreadable or malformed files fall back entirely rather than raising:
    a tokenizer that cannot be introspected is not a reason to fail a run.
    """
    defaults: dict[str, Any] = {
        key: (list(val) if isinstance(val, list) else val)
        for key, val in DEFAULT_SPECIAL_TOKENS.items()
    }
    if tokenizer_path is None:
        return defaults
    path = Path(tokenizer_path)
    if not path.exists():
        return defaults
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        added = data.get("added_tokens") or []
        specials = [
            t["content"] for t in added if isinstance(t, dict) and t.get("special")
        ]
        if not specials:
            return defaults
        known = set(specials)
        for key, val in list(defaults.items()):
            if key == "additional_special_tokens":
                defaults[key] = [t for t in val if t in known] or val
            elif val in known:
                continue
            else:
                # The file names this slot differently (``[PAD]`` for
                # ``pad_token``): match on the slot name inside the token.
                needle = key.replace("_token", "").upper()
                for candidate in specials:
                    if needle in candidate.upper().strip("<>[]"):
                        defaults[key] = candidate
                        break
        return defaults
    except Exception:  # noqa: BLE001  an unreadable tokenizer falls back
        return defaults
