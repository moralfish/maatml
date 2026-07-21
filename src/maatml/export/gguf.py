"""Optional GGUF export via llama.cpp conversion scripts (not vendored)."""
from __future__ import annotations

import importlib
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Optional

from ..config import ModelDefinition
from ..registry import register_exporter
from .bundle import export_safetensors_bundle
from .manifest import build_manifest, write_manifest

_INSTALL_HINT = (
    "GGUF export requires llama.cpp convert tooling. "
    "Install llama.cpp and ensure `convert_hf_to_gguf.py` (or the "
    "`llama_cpp.convert` module) is on PATH / PYTHONPATH, then retry."
)


def _find_convert_script() -> Optional[Path]:
    """Locate a known HF→GGUF conversion entry point if present."""
    candidates = (
        "convert_hf_to_gguf.py",
        "convert-hf-to-gguf.py",
        "convert.py",
    )
    for name in candidates:
        found = shutil.which(name)
        if found:
            return Path(found)
    # Also check common local clone locations relative to cwd.
    for rel in (
        Path("llama.cpp") / "convert_hf_to_gguf.py",
        Path("llama.cpp") / "convert-hf-to-gguf.py",
    ):
        if rel.is_file():
            return rel.resolve()
    return None


def _try_module_convert(model_dir: Path, out_path: Path) -> bool:
    """Attempt ``llama_cpp.convert``-style module if importable."""
    for mod_name in ("llama_cpp.convert", "llama_cpp_python.convert"):
        try:
            mod = importlib.import_module(mod_name)
        except ImportError:
            continue
        convert_fn = getattr(mod, "main", None) or getattr(mod, "convert", None)
        if convert_fn is None:
            continue
        try:
            convert_fn([str(model_dir), "--outfile", str(out_path)])
            return out_path.is_file()
        except TypeError:
            try:
                convert_fn(str(model_dir), outfile=str(out_path))
                return out_path.is_file()
            except Exception:  # noqa: BLE001
                continue
        except Exception:  # noqa: BLE001
            continue
    return False


@register_exporter("gguf")
def export_gguf(
    model_def: ModelDefinition,
    checkpoint_dir: Path,
    out_dir: Path,
    *,
    run_id: Optional[str] = None,
) -> Path:
    """Export safetensors bundle first, then attempt GGUF conversion."""
    out_dir = Path(out_dir).resolve()
    # Always materialise a HF-compatible directory first.
    export_safetensors_bundle(model_def, checkpoint_dir, out_dir, run_id=run_id)

    gguf_path = out_dir / f"{model_def.name}.gguf"
    script = _find_convert_script()
    converted = False
    if script is not None:
        cmd = [sys.executable, str(script), str(out_dir), "--outfile", str(gguf_path)]
        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True)
            converted = gguf_path.is_file()
        except (subprocess.CalledProcessError, OSError) as exc:
            raise RuntimeError(
                f"GGUF conversion failed via {script}: {exc}\n{_INSTALL_HINT}"
            ) from exc
    else:
        converted = _try_module_convert(out_dir, gguf_path)

    if not converted:
        raise ImportError(_INSTALL_HINT)

    files = [p for p in out_dir.iterdir() if p.is_file() and p.name != "manifest.json"]
    manifest = build_manifest(
        model_def=model_def,
        export_dir=out_dir,
        files=files,
        formats=["safetensors", "gguf"],
        source_checkpoint=checkpoint_dir,
        run_id=run_id,
    )
    write_manifest(out_dir, manifest)
    return out_dir
