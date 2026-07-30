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


def test_serve_validation_is_not_claimed_on_by_default() -> None:
    """serve validates only under --enforce or ?validate=1 (serve.py
    do_validate). The front page must not promise reporting by default, and it
    must name the opt-in query parameter that actually turns annotation on.

    The behaviour itself is pinned by test_serve.py: a plain /predict has no
    ``valid`` key, ?validate=1 sets it, and --enforce returns 422 on a failing
    output. This test only keeps the docs honest about it."""
    for doc in ("README.md", "docs/index.md"):
        norm = _norm(doc)
        assert "reporting the result by default" not in norm
        assert "validate=1" in norm
        assert "--enforce" in norm


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


def test_documented_validator_example_runs() -> None:
    """The validator snippet in docs/lifecycle.md is the first thing a plugin
    author copies. It previously called ValidationResult(ok=True, errors=[]),
    which raises TypeError: `ok` is a derived property and raw_output is
    required. Execute the documented code so it cannot rot again."""
    import re as _re

    from maatml.registry import VALIDATORS, snapshot_registries, restore_registries

    doc = _read("docs/lifecycle.md")
    section = doc.split("## Registering a validator", 1)[1]
    match = _re.search(r"```python\n(.*?)```", section, _re.S)
    assert match, "no python snippet under 'Registering a validator'"
    snippet = match.group(1)

    snapshot = snapshot_registries()
    try:
        namespace: dict = {}
        exec(compile(snippet, "docs/lifecycle.md", "exec"), namespace)  # noqa: S102
        validate = VALIDATORS.require("my_task")
        good = validate('{"answer": "42"}')
        assert good.ok is True, good.errors
        bad_json = validate("not json")
        assert bad_json.ok is False
        assert [e.code for e in bad_json.errors] == ["invalid_json"]
        missing = validate('{"other": 1}')
        assert missing.ok is False
        assert [e.code for e in missing.errors] == ["missing_answer"]
    finally:
        restore_registries(snapshot)


def test_quickstart_corpus_size_matches_the_example() -> None:
    """docs/getting-started.md quotes the triage corpus size. It described an
    8-row corpus long after the example grew, which made the whole 'expect it
    to fail' narrative wrong. Pin the number to the shipped file."""
    import re as _re

    doc = _read("docs/getting-started.md")
    match = _re.search(r"ships ([\d,]+) seed rows", doc)
    assert match, "getting-started no longer states the seed corpus size"
    claimed = int(match.group(1).replace(",", ""))
    seeds = REPO / "examples/support-ticket-triage/datasets/samples/seed_samples.jsonl"
    actual = sum(1 for line in seeds.read_text(encoding="utf-8").splitlines() if line.strip())
    assert claimed == actual, f"doc says {claimed} seed rows, file has {actual}"
