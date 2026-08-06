from __future__ import annotations

from pathlib import Path

from maatml.data.sanitizer import make_tag_sanitizer


def test_tag_sanitizer_applies_only_matching_rules(tmp_path: Path) -> None:
    rules_path = tmp_path / "rules.yaml"
    rules_path.write_text(
        "rules:\n"
        "  - name: email\n"
        "    pattern: '[A-Za-z]+@[A-Za-z.]+'\n"
        "    replacement: 'REDACTED'\n"
        "    applies_to: [ticket]\n"
        "    length_preserving: false\n"
        "  - name: internal_id\n"
        "    pattern: 'ID=[0-9]+'\n"
        "    replacement: 'ID=REDACTED'\n"
        "    applies_to: [audit]\n"
        "    length_preserving: false\n",
        encoding="utf-8",
    )
    sanitize_ticket = make_tag_sanitizer(rules_path, tag="ticket")
    out = sanitize_ticket("Contact alice@example.com about ID=123")
    assert out == "Contact REDACTED about ID=123"


def test_length_preserving_truncation_warns_once_per_rule(tmp_path: Path) -> None:
    import warnings

    import pytest

    from maatml.data import sanitizer as sanitizer_mod

    rules_path = tmp_path / "rules.yaml"
    rules_path.write_text(
        "rules:\n"
        "  - name: token_assignment\n"
        "    pattern: '(TOKEN=)[A-Z]{1,8}'\n"
        "    replacement: '\\g<1>REDACTED'\n"
        "    applies_to: [record]\n"
        "    length_preserving: true\n",
        encoding="utf-8",
    )
    sanitize_record = make_tag_sanitizer(rules_path, tag="record", length_preserving_only=True)
    sanitizer_mod._warned_truncating_rules.discard("token_assignment")
    with pytest.warns(RuntimeWarning, match="marker is abbreviated"):
        out = sanitize_record("TOKEN=BOB\n")
    # The warning is about legibility, not leakage: the original value is gone.
    assert "BOB" not in out
    # Second hit stays quiet so a large corpus does not emit one warning per row.
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        sanitize_record("TOKEN=AMY\n")


def test_fixed_replacement_that_cannot_fit_is_rejected_at_load(tmp_path: Path) -> None:
    import pytest

    from maatml.data.sanitizer import load_rules

    rules_path = tmp_path / "rules.yaml"
    rules_path.write_text(
        "rules:\n"
        "  - name: too_long\n"
        "    pattern: 'ID=[A-Z]{1,4}'\n"
        "    replacement: 'ID=REDACTED-VALUE'\n"
        "    applies_to: [x]\n"
        "    length_preserving: true\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="cannot fit a match as short as"):
        load_rules(rules_path)


def test_fixed_replacement_that_fits_loads(tmp_path: Path) -> None:
    from maatml.data.sanitizer import apply_rules, load_rules

    rules_path = tmp_path / "rules.yaml"
    rules_path.write_text(
        "rules:\n"
        "  - name: ok\n"
        "    pattern: 'ID=[A-Z]{4}'\n"
        "    replacement: 'ID=XXXX'\n"
        "    applies_to: [x]\n"
        "    length_preserving: true\n",
        encoding="utf-8",
    )
    rules = load_rules(rules_path)
    assert apply_rules("ID=ABCD end", rules) == "ID=XXXX end"
