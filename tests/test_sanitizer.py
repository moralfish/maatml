from __future__ import annotations

from pathlib import Path

from maatml.data.sanitizer import make_tag_sanitizer

REPO = Path(__file__).resolve().parents[1]
JCL_RULES = REPO / "examples" / "jcl-validator" / "jcl_plugin" / "sanitization.yaml"
SPOOL_RULES = (
    REPO / "examples" / "spool-interpreter" / "spool_plugin" / "sanitization.yaml"
)

sanitize_jcl = make_tag_sanitizer(JCL_RULES, tag="jcl", length_preserving_only=True)
sanitize_spool = make_tag_sanitizer(SPOOL_RULES, tag="spool")


def test_jcl_redacts_userid_assignment_length_preserving() -> None:
    src = "//STEP1 EXEC PGM=PGM,USER=ALICE01\n"
    out = sanitize_jcl(src)
    assert "ALICE01" not in out
    assert "USER=REDACT" in out
    assert len(out) == len(src), "length-preserving JCL sanitizer must not change line length"


def test_jcl_redacts_ipv4_length_preserving() -> None:
    src = "//* host 10.20.30.40 reach\n"
    out = sanitize_jcl(src)
    assert "10.20.30.40" not in out
    assert "0.0.0.0" in out
    assert len(out) == len(src)


def test_jcl_leaves_dataset_names_untouched() -> None:
    src = "//IN DD DSN=PROD.DAILY.LOAD,DISP=SHR\n"
    assert sanitize_jcl(src) == src


def test_spool_redacts_credentials() -> None:
    src = "USER=BOB password=hunter2 ended\n"
    out = sanitize_spool(src)
    assert "hunter2" not in out
    assert "password=REDACTED" in out
    assert "BOB" not in out


def test_spool_redacts_hostname_and_ip() -> None:
    src = "connecting to mvs.prod.corp at 192.168.1.10\n"
    out = sanitize_spool(src)
    assert "mvs.prod.corp" not in out
    assert "192.168.1.10" not in out


def test_jcl_clean_input_unchanged() -> None:
    src = "//J JOB (123),CLASS=A,MSGCLASS=H\n//STEP1 EXEC PGM=IEFBR14\n"
    assert sanitize_jcl(src) == src


def test_length_preserving_truncation_warns_once_per_rule() -> None:
    import warnings

    import pytest

    from maatml.data import sanitizer as sanitizer_mod

    sanitizer_mod._warned_truncating_rules.discard("userid_assignment")
    with pytest.warns(RuntimeWarning, match="redaction is incomplete"):
        sanitize_jcl("//STEP1 EXEC PGM=P,TSO=BOB\n")
    # Second hit stays quiet so a large corpus does not emit one warning per row.
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        sanitize_jcl("//STEP2 EXEC PGM=P,TSO=AMY\n")


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
