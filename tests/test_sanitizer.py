from __future__ import annotations

from flow_ml.data.sanitizer import sanitize_jcl, sanitize_spool


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
