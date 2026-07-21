"""Optional MLX export via ``mlx_lm.convert`` when installed."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from ..config import ModelDefinition
from ..registry import register_exporter
from .bundle import export_safetensors_bundle
from .manifest import build_manifest, write_manifest

_INSTALL_HINT = (
    "MLX export requires mlx_lm. Install with `pip install mlx-lm` "
    "(Apple Silicon / macOS) and retry."
)


@register_exporter("mlx")
def export_mlx(
    model_def: ModelDefinition,
    checkpoint_dir: Path,
    out_dir: Path,
    *,
    run_id: Optional[str] = None,
) -> Path:
    """Export safetensors bundle, then convert with ``mlx_lm.convert`` if available."""
    out_dir = Path(out_dir).resolve()
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
