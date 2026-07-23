"""Doc claims stay truthful (DOCS-a/b/c) and free of em dashes."""
from __future__ import annotations

import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

_TOUCHED = (
    "README.md",
    "docs/index.md",
    "docs/lifecycle.md",
    "docs/serving.md",
    "SECURITY.md",
    "docs/plugins.md",
)


def _read(rel: str) -> str:
    return (REPO / rel).read_text(encoding="utf-8")


def _norm(rel: str) -> str:
    return re.sub(r"\s+", " ", _read(rel))


def test_serve_gating_wording_updated() -> None:
    for doc in ("README.md", "docs/index.md"):
        norm = _norm(doc)
        assert "your **live inference**, so a MaatML model" not in norm
        assert "--enforce" in norm
    life = _read("docs/lifecycle.md")
    assert "annotates" in life
    assert "--enforce" in life


def test_trust_boundary_documented() -> None:
    readme = _read("README.md")
    assert "Trust boundary" in readme
    assert "maatml validate" in readme
    sec = _read("SECURITY.md")
    assert "Trust model" in sec
    assert "untrusted" in sec
    plug = _read("docs/plugins.md")
    assert "Trust boundary" in plug
    assert "arbitrary Python" in plug


def test_verify_described_as_corruption_not_tamper() -> None:
    serving = _read("docs/serving.md")
    assert "unchanged since export" not in serving
    assert "not a signature" in serving or "not tampering" in serving


def test_touched_docs_have_no_em_dash() -> None:
    for doc in _TOUCHED:
        assert "—" not in _read(doc), f"em dash in {doc}"
