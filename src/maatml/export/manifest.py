"""Export ``manifest.json`` build / verify helpers."""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from ..config import ModelDefinition, PackagingSpec
from ..utils.io import read_json, sha256_file, write_json


def _file_entries(root: Path, files: list[Path]) -> list[dict[str, str]]:
    entries: list[dict[str, str]] = []
    for path in files:
        rel = path.relative_to(root).as_posix()
        entries.append({"path": rel, "sha256": sha256_file(path)})
    return entries


def build_manifest(
    *,
    model_def: ModelDefinition,
    export_dir: Path,
    files: list[Path],
    formats: list[str],
    source_checkpoint: str | Path,
    run_id: Optional[str] = None,
    packaging: Optional[PackagingSpec] = None,
    extra_runtime_hints: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """Assemble an export manifest (inspired by legacy ``.fm`` manifests)."""
    pkg = packaging or model_def.packaging
    hints: dict[str, Any] = {
        "formats": list(formats),
        "max_input_tokens": pkg.max_input_tokens,
        "expected_latency_ms": pkg.expected_latency_ms,
        "weights_dtype": pkg.weights_dtype,
    }
    if extra_runtime_hints:
        hints.update(extra_runtime_hints)

    manifest: dict[str, Any] = {
        "name": model_def.name,
        "version": model_def.version,
        "identity": model_def.identity,
        "architecture": model_def.architecture,
        "base_model": model_def.base_model or model_def.training.get("model_id"),
        "runtime_hints": hints,
        "packaging": pkg.model_dump(mode="json"),
        "files": _file_entries(export_dir, files),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_checkpoint": str(source_checkpoint),
    }
    if run_id:
        manifest["run_id"] = run_id
    return manifest


def write_manifest(export_dir: Path, manifest: dict[str, Any]) -> Path:
    return write_json(Path(export_dir) / "manifest.json", manifest)


def load_manifest(path: str | Path) -> tuple[Path, dict[str, Any]]:
    """Load a manifest from a file path or an export directory."""
    path = Path(path).resolve()
    if path.is_dir():
        manifest_path = path / "manifest.json"
    else:
        manifest_path = path
    if not manifest_path.is_file():
        raise FileNotFoundError(f"manifest.json not found at {manifest_path}")
    data = read_json(manifest_path)
    if not isinstance(data, dict):
        raise ValueError(f"Invalid manifest (expected object): {manifest_path}")
    return manifest_path.parent, data


def verify_manifest(path: str | Path) -> list[str]:
    """Recompute sha256 for listed files; return a list of mismatch messages.

    Empty list means OK. Missing files are reported as mismatches.
    """
    root, data = load_manifest(path)
    files = data.get("files") or []
    errors: list[str] = []
    if not isinstance(files, list):
        return ["manifest.files must be a list"]
    for entry in files:
        if not isinstance(entry, dict):
            errors.append(f"invalid file entry: {entry!r}")
            continue
        rel = entry.get("path")
        expected = entry.get("sha256")
        if not rel or not expected:
            errors.append(f"incomplete file entry: {entry!r}")
            continue
        target = root / rel
        if not target.is_file():
            errors.append(f"missing file: {rel}")
            continue
        actual = sha256_file(target)
        if actual != expected:
            errors.append(f"checksum mismatch: {rel} (expected {expected}, got {actual})")
    return errors
