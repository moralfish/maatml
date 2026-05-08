"""Shim: forwards to `python -m flow_ml.cli evaluate ...`."""

import sys

from flow_ml.cli import app

if __name__ == "__main__":
    sys.argv = [sys.argv[0], "evaluate", *sys.argv[1:]]
    app()
