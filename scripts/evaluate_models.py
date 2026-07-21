"""Shim: forwards to `python -m maatml.cli evaluate ...`."""

import sys

from maatml.cli import app

if __name__ == "__main__":
    sys.argv = [sys.argv[0], "evaluate", *sys.argv[1:]]
    app()
