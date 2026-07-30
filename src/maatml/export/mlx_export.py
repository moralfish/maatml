"""MLX export: seq2seq serving bundles directly, decoder-only via ``mlx_lm.convert``."""

from __future__ import annotations

import json
import shutil
import warnings
from pathlib import Path
from typing import Any, Optional

from ..config import ModelDefinition, get_dataset_cfg
from ..registry import register_exporter
from ..scaffold import normalize_architecture
from ..utils.io import write_json
from .bundle import export_safetensors_bundle
from .manifest import build_manifest, read_safetensors_tensor_names, write_manifest

_INSTALL_HINT = (
    "MLX export requires mlx_lm. Install with `pip install mlx-lm` "
    "(Apple Silicon / macOS) and retry."
)

SERVING_CONTRACT = "maatml.serving/1"

# T5 SentencePiece maps `{`/`}` to <unk>; clients restore them (see
# Seq2SeqPredictor's repair_braces handling).
_BRACE_OUTPUT_NOTE = (
    "emits the JSON object body without outer braces; T5 vocab maps { } to "
    "unk. Clients re-add the braces before parsing."
)

# Files a seq2seq MLX serving bundle carries besides serving.json and
# spiece.model. MLX loads the T5 safetensors directly; the engine does the
# weight-name mapping, so no mlx_lm conversion is involved.
_SEQ2SEQ_BUNDLE_FILES = (
    "config.json",
    "model.safetensors",
    "tokenizer.json",
    "tokenizer_config.json",
    "generation_config.json",
)


def build_seq2seq_serving(model_def: ModelDefinition, checkpoint_dir: Path) -> dict[str, Any]:
    """serving.json body for the ``seq2seq_lm`` serving contract.

    Token budgets use the same resolution as training and evaluation:
    ``training.source_max_len`` for the input and the training generation
    config for ``max_new_tokens`` (Seq2SeqPredictor's defaults otherwise).
    """
    config_path = Path(checkpoint_dir) / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"config.json not found in checkpoint: {checkpoint_dir}")
    ckpt_cfg = json.loads(config_path.read_text(encoding="utf-8"))

    training = model_def.training or {}
    generation = training.get("generation") or {}
    serving: dict[str, Any] = {
        "contract": SERVING_CONTRACT,
        "kind": "seq2seq_lm",
        "max_input_tokens": int(training.get("source_max_len", 1024)),
        "max_new_tokens": int(generation.get("max_new_tokens", 512)),
        "decoder_start_token_id": ckpt_cfg.get("decoder_start_token_id"),
        "eos_token_id": ckpt_cfg.get("eos_token_id"),
        "input_prefix": get_dataset_cfg(model_def).get("source_prefix") or "",
    }
    if (model_def.evaluation or {}).get("repair_braces"):
        serving["output_note"] = _BRACE_OUTPUT_NOTE
    return serving


def _untie_config_if_head_present(config_path: Path, weights_path: Path) -> None:
    """Set ``tie_word_embeddings`` false when the checkpoint has an lm_head.

    The fine-tune untied the head; a tied config makes downstream loaders
    project logits through the wrong matrix.
    """
    if "lm_head.weight" not in read_safetensors_tensor_names(weights_path):
        return
    cfg = json.loads(config_path.read_text(encoding="utf-8"))
    if cfg.get("tie_word_embeddings") is False:
        return
    cfg["tie_word_embeddings"] = False
    config_path.write_text(json.dumps(cfg, indent=2) + "\n", encoding="utf-8")


def _export_seq2seq_mlx(
    model_def: ModelDefinition,
    checkpoint_dir: Path,
    out_dir: Path,
    *,
    run_id: Optional[str] = None,
) -> Path:
    """Assemble ``<name>.mlx/`` from the safetensors bundle + serving.json."""
    export_safetensors_bundle(model_def, checkpoint_dir, out_dir, run_id=run_id)

    serving = build_seq2seq_serving(model_def, checkpoint_dir)

    bundle_dir = out_dir / f"{model_def.name}.mlx"
    bundle_dir.mkdir(parents=True, exist_ok=True)
    for name in _SEQ2SEQ_BUNDLE_FILES:
        src = out_dir / name
        if src.is_file():
            shutil.copy2(src, bundle_dir / name)

    _untie_config_if_head_present(bundle_dir / "config.json", checkpoint_dir / "model.safetensors")

    spiece = checkpoint_dir / "spiece.model"
    if spiece.is_file():
        shutil.copy2(spiece, bundle_dir / "spiece.model")
    else:
        warnings.warn(
            f"spiece.model not found in checkpoint {checkpoint_dir}; the MLX "
            "bundle ships without the SentencePiece model",
            RuntimeWarning,
            stacklevel=2,
        )

    write_json(bundle_dir / "serving.json", serving, sort_keys=False)

    files = [p for p in out_dir.rglob("*") if p.is_file() and p.name != "manifest.json"]
    manifest = build_manifest(
        model_def=model_def,
        export_dir=out_dir,
        files=files,
        formats=["safetensors", "mlx"],
        source_checkpoint=checkpoint_dir,
        run_id=run_id,
    )
    write_manifest(out_dir, manifest)
    return out_dir


@register_exporter("mlx")
def export_mlx(
    model_def: ModelDefinition,
    checkpoint_dir: Path,
    out_dir: Path,
    *,
    run_id: Optional[str] = None,
) -> Path:
    """Export an MLX bundle.

    seq2seq checkpoints are assembled directly (``mlx_lm.convert`` only
    supports decoder-only models); everything else converts via mlx_lm.
    """
    out_dir = Path(out_dir).resolve()
    checkpoint_dir = Path(checkpoint_dir).resolve()
    if normalize_architecture(model_def.architecture) == "seq2seq":
        return _export_seq2seq_mlx(model_def, checkpoint_dir, out_dir, run_id=run_id)

    export_safetensors_bundle(model_def, checkpoint_dir, out_dir, run_id=run_id)

    try:
        from mlx_lm import convert as mlx_convert  # type: ignore[import-untyped]
    except ImportError as exc:
        raise ImportError(_INSTALL_HINT) from exc

    mlx_dir = out_dir / "mlx"
    mlx_dir.mkdir(parents=True, exist_ok=True)
    try:
        # mlx_lm.convert API: convert(hf_path, mlx_path=..., quantize=False)
        mlx_convert(str(out_dir), mlx_path=str(mlx_dir))
    except TypeError:
        try:
            mlx_convert(hf_path=str(out_dir), mlx_path=str(mlx_dir))
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(f"mlx_lm.convert failed: {exc}\n{_INSTALL_HINT}") from exc
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"mlx_lm.convert failed: {exc}\n{_INSTALL_HINT}") from exc

    files = [p for p in out_dir.rglob("*") if p.is_file() and p.name != "manifest.json"]
    # Prefer paths relative to out_dir for the manifest helper.
    manifest = build_manifest(
        model_def=model_def,
        export_dir=out_dir,
        files=files,
        formats=["safetensors", "mlx"],
        source_checkpoint=checkpoint_dir,
        run_id=run_id,
    )
    write_manifest(out_dir, manifest)
    return out_dir
