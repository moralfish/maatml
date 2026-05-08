"""Shim: forwards to `python -m flow_ml.cli train models/spool-interpreter/ ...`."""

import sys
from pathlib import Path

from flow_ml.cli import app

REPO_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = REPO_ROOT / "models" / "spool-interpreter"

if __name__ == "__main__":
    sys.argv = [sys.argv[0], "train", str(MODEL_DIR), *sys.argv[1:]]
    app()
