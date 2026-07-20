"""Lazy trainer registration — importable without torch / transformers.

Heavy trainer modules pull ML deps at import time (notably ``sft_base``).
CLI commands like ``scaffold`` / ``plugins`` / ``validate`` only need the
architecture names registered, so discovery loads this shim instead.
"""
from __future__ import annotations

from typing import Any

from ..registry import register_trainer


@register_trainer("causal_sft")
def train_causal_sft(model_def: Any, **kwargs: Any) -> Any:
    from .sft_base import train_sft

    return train_sft(model_def, **kwargs)


@register_trainer("multi_head_classifier")
@register_trainer("classifier")
def train_classifier(model_def: Any, **kwargs: Any) -> Any:
    from .jcl_classifier import train_jcl_classifier

    return train_jcl_classifier(model_def, **kwargs)


@register_trainer("seq2seq")
def train_seq2seq(model_def: Any, **kwargs: Any) -> Any:
    from .spool_seq2seq import train_spool_seq2seq

    return train_spool_seq2seq(model_def, **kwargs)
