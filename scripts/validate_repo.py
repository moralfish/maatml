"""Validate all model folders under models/ (and examples/ if present).

Delegates to ``flow_ml.scaffold.validate_model_dir`` — no hardcoded
scaffold file lists or runtime checks.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from flow_ml.scaffold import validate_model_dir  # noqa: E402


def _model_dirs() -> list[Path]:
    dirs: list[Path] = []
    for root_name in ("models", "examples"):
        root = ROOT / root_name
        if not root.is_dir():
            continue
        for child in sorted(root.iterdir()):
            if child.is_dir() and (child / "model.yml").is_file():
                dirs.append(child)
    return dirs


def main() -> int:
    dirs = _model_dirs()
    if not dirs:
        print("validate_repo: no model.yml folders found under models/ or examples/")
        return 1

    all_errors: list[str] = []
    for model_dir in dirs:
        errors = validate_model_dir(model_dir)
        if errors:
            for err in errors:
                all_errors.append(f"{model_dir.relative_to(ROOT).as_posix()}: {err}")

    if all_errors:
        print("validate_repo: failed")
        for error in all_errors:
            print(f"- {error}")
        return 1

    print(f"validate_repo: ok ({len(dirs)} model folders)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
