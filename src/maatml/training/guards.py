"""Training-time guards: NaN abort, run provenance, tokenizer/model contract."""
from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from ..config import ModelDefinition
from ..device import is_main_process
from ..utils.io import sha256_file, write_json


class NanGuardCallback:
    """Abort training when loss or grad_norm becomes non-finite.

    Instantiated lazily so importing this module does not require torch /
    transformers (keeps unit tests light). Use :func:`make_nan_guard_callback`
    or construct after ``from transformers import TrainerCallback``.
    """

    @staticmethod
    def create():
        from transformers import TrainerCallback, TrainerControl, TrainerState, TrainingArguments

        class _NanGuardCallback(TrainerCallback):
            def on_log(
                self,
                args: TrainingArguments,
                state: TrainerState,
                control: TrainerControl,
                logs: Optional[dict[str, float]] = None,
                **kwargs,
            ):
                if not logs:
                    return
                for key in ("loss", "grad_norm", "train_loss", "eval_loss"):
                    val = logs.get(key)
                    if val is None:
                        continue
                    try:
                        fval = float(val)
                    except (TypeError, ValueError):
                        continue
                    if fval != fval or fval in (float("inf"), float("-inf")):  # noqa: PLR0124
                        raise RuntimeError(
                            f"Non-finite {key}={val!r} at step {state.global_step}; aborting training"
                        )

        return _NanGuardCallback()


def make_nan_guard_callback():
    """Return a transformers ``TrainerCallback`` that aborts on non-finite loss."""
    return NanGuardCallback.create()


def _git_sha(cwd: Optional[Path] = None) -> Optional[str]:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=str(cwd) if cwd else None,
            stderr=subprocess.DEVNULL,
            text=True,
        )
        return out.strip() or None
    except Exception:  # noqa: BLE001
        return None


def _pkg_version(name: str) -> Optional[str]:
    try:
        from importlib.metadata import version

        return version(name)
    except Exception:  # noqa: BLE001
        return None


def write_run_metadata(
    out_dir: str | Path,
    model_def: ModelDefinition,
    dataset_paths: dict[str, str | Path],
    extra: Optional[dict[str, Any]] = None,
) -> Optional[Path]:
    """Write ``run_metadata.json`` with spec snapshot, hashes, and env provenance.

    Rank-0 only under multi-GPU; non-main ranks return ``None``.
    """
    if not is_main_process():
        return None
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    spec = model_def.model_dump(mode="json", exclude={"model_dir"})
    spec_json = json.dumps(spec, sort_keys=True, separators=(",", ":"))
    import hashlib

    spec_hash = hashlib.sha256(spec_json.encode("utf-8")).hexdigest()

    ds_hashes: dict[str, str] = {}
    for key, path in dataset_paths.items():
        p = Path(path)
        if p.is_file():
            ds_hashes[key] = sha256_file(p)

    payload: dict[str, Any] = {
        "identity": model_def.identity,
        "model_id": model_def.model_id,
        "architecture": model_def.architecture,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git_sha": _git_sha(model_def.model_dir),
        "spec_hash": spec_hash,
        "spec": spec,
        "dataset_hashes": ds_hashes,
        "package_versions": {
            "maatml": _pkg_version("maatml"),
            "torch": _pkg_version("torch"),
            "transformers": _pkg_version("transformers"),
            "peft": _pkg_version("peft"),
            "safetensors": _pkg_version("safetensors"),
        },
    }
    if extra:
        payload["extra"] = extra

    return write_json(out_dir / "run_metadata.json", payload)


def ensure_tokenizer_model_contract(
    model: Any,
    tokenizer: Any,
    *,
    embedding_strategy: Optional[str] = None,
) -> None:
    """Align model embedding size with tokenizer vocab when they diverge.

    Strategies:
      - ``resize`` / ``reinit``: call ``model.resize_token_embeddings(len(tokenizer))``
      - ``reuse``: allow only when tokenizer vocab ≤ model vocab (no resize)
    """
    try:
        tok_size = len(tokenizer)
    except TypeError:
        tok_size = tokenizer.vocab_size

    model_vocab = None
    if hasattr(model, "get_input_embeddings"):
        emb = model.get_input_embeddings()
        if emb is not None and hasattr(emb, "weight"):
            model_vocab = int(emb.weight.shape[0])
    if model_vocab is None and hasattr(model, "config"):
        model_vocab = getattr(model.config, "vocab_size", None)
    if model_vocab is None:
        return
    model_vocab = int(model_vocab)

    if tok_size == model_vocab:
        return

    allowed = {"resize", "reinit", "reuse"}
    if embedding_strategy not in allowed:
        raise ValueError(
            f"Tokenizer vocab ({tok_size}) != model vocab ({model_vocab}). "
            f"Set training.embedding_strategy to one of {sorted(allowed)}."
        )

    if embedding_strategy == "reuse":
        if tok_size > model_vocab:
            raise ValueError(
                f"embedding_strategy=reuse requires tokenizer vocab ({tok_size}) "
                f"<= model vocab ({model_vocab})"
            )
        return

    # resize / reinit
    if hasattr(model, "resize_token_embeddings"):
        model.resize_token_embeddings(tok_size)
