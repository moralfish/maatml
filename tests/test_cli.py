"""CLI surface: exit codes, argument parsing, and user-error messages.

Torch-free by construction: every command here fails (or succeeds) before any
model load, which is exactly the behaviour that has to hold in CI.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from maatml.cli import app

runner = CliRunner()


def _write_model(
    tmp_path: Path,
    *,
    name: str = "cli-test",
    evaluation: str = "",
    extra: str = "",
) -> Path:
    mdir = tmp_path / name
    (mdir / "datasets").mkdir(parents=True, exist_ok=True)
    (mdir / "model.yml").write_text(
        f"""name: {name}
model_id: {name}
architecture: causal_sft
version: 0.1.0
dataset:
  seed_samples: datasets/seeds.jsonl
{evaluation}{extra}""",
        encoding="utf-8",
    )
    (mdir / "datasets" / "seeds.jsonl").write_text("", encoding="utf-8")
    return mdir


# --- argument parsing ------------------------------------------------------


def test_help_lists_every_command() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    for command in (
        "prepare",
        "train",
        "sweep",
        "evaluate",
        "export",
        "verify",
        "serve",
        "datagen",
        "ingest",
        "runs",
        "scaffold",
        "validate",
        "plan",
        "plugins",
    ):
        assert command in result.output


def test_unknown_command_exits_two() -> None:
    result = runner.invoke(app, ["definitely-not-a-command"])
    assert result.exit_code == 2


def test_missing_model_dir_is_a_usage_error() -> None:
    result = runner.invoke(app, ["prepare", "/nonexistent/model/dir"])
    assert result.exit_code == 2
    assert "Error" in result.output


def test_ingest_rejects_a_malformed_field_map(tmp_path: Path) -> None:
    mdir = _write_model(tmp_path)
    source = tmp_path / "in.jsonl"
    source.write_text('{"text": "x"}\n', encoding="utf-8")
    result = runner.invoke(
        app, ["ingest", str(mdir), "--input", str(source), "--map", "no-equals-sign"]
    )
    assert result.exit_code == 2
    assert "field=col" in result.output


def test_plugins_lists_registered_trainers() -> None:
    result = runner.invoke(app, ["plugins"])
    assert result.exit_code == 0
    assert "causal_sft" in result.output


# --- evaluate --gate -------------------------------------------------------


def test_evaluate_gate_without_gates_configured_exits_nonzero(tmp_path: Path) -> None:
    mdir = _write_model(tmp_path, evaluation="evaluation:\n  predictor: causal_sft\n")
    result = runner.invoke(app, ["evaluate", str(mdir), "--gate"])
    assert result.exit_code != 0
    assert "no evaluation.gates" in result.output.replace("\n", " ")


def test_evaluate_reports_unregistered_validator_before_loading(tmp_path: Path) -> None:
    mdir = _write_model(
        tmp_path,
        evaluation="evaluation:\n  predictor: causal_sft\n  validator: not_registered\n",
    )
    result = runner.invoke(app, ["evaluate", str(mdir)])
    assert result.exit_code != 0
    assert "not_registered" in result.output


def test_evaluate_reports_unregistered_metrics_before_loading(tmp_path: Path) -> None:
    mdir = _write_model(
        tmp_path,
        evaluation="evaluation:\n  predictor: causal_sft\n  metrics: [nope_metrics]\n",
    )
    result = runner.invoke(app, ["evaluate", str(mdir)])
    assert result.exit_code != 0
    assert "nope_metrics" in result.output


# --- verify ----------------------------------------------------------------


def _export_dir(tmp_path: Path) -> Path:
    """A real export bundle: manifest built the way `maatml export` builds it."""
    from maatml.config import load_model_def
    from maatml.export.manifest import build_manifest, write_manifest

    mdir = _write_model(tmp_path, name="exported")
    export = tmp_path / "export"
    export.mkdir()
    (export / "model.safetensors").write_bytes(b"weights")
    (export / "config.json").write_text("{}", encoding="utf-8")
    manifest = build_manifest(
        model_def=load_model_def(mdir),
        export_dir=export,
        files=[export / "model.safetensors", export / "config.json"],
        formats=["safetensors"],
        source_checkpoint=tmp_path / "ckpt",
    )
    write_manifest(export, manifest)
    return export


def test_verify_passes_on_an_untouched_export(tmp_path: Path) -> None:
    export = _export_dir(tmp_path)
    result = runner.invoke(app, ["verify", str(export)])
    assert result.exit_code == 0, result.output
    assert "OK" in result.output


def test_verify_fails_on_a_tampered_file(tmp_path: Path) -> None:
    export = _export_dir(tmp_path)
    (export / "model.safetensors").write_bytes(b"tampered")
    result = runner.invoke(app, ["verify", str(export)])
    assert result.exit_code == 1
    assert "verify failed" in result.output


def test_verify_fails_when_a_listed_file_is_missing(tmp_path: Path) -> None:
    export = _export_dir(tmp_path)
    (export / "config.json").unlink()
    result = runner.invoke(app, ["verify", str(export)])
    assert result.exit_code == 1
    assert "config.json" in result.output


def test_verify_on_a_missing_manifest_exits_nonzero(tmp_path: Path) -> None:
    result = runner.invoke(app, ["verify", str(tmp_path / "no-such-export")])
    assert result.exit_code != 0


# --- scaffold --------------------------------------------------------------


def test_scaffold_creates_a_model_folder(tmp_path: Path) -> None:
    target = tmp_path / "fresh"
    result = runner.invoke(
        app, ["scaffold", str(target), "--architecture", "causal_sft"]
    )
    assert result.exit_code == 0, result.output
    assert (target / "model.yml").is_file()
    assert (target / "datasets" / "samples" / "seed_samples.jsonl").is_file()


def test_scaffold_refuses_to_overwrite_without_force(tmp_path: Path) -> None:
    target = tmp_path / "existing"
    runner.invoke(app, ["scaffold", str(target), "--architecture", "causal_sft"])
    (target / "datasets" / "samples" / "seed_samples.jsonl").write_text(
        json.dumps({"request": "mine", "expected_output": {"a": 1}}) + "\n",
        encoding="utf-8",
    )

    result = runner.invoke(
        app, ["scaffold", str(target), "--architecture", "causal_sft"]
    )
    assert result.exit_code == 1
    assert "scaffold refused" in result.output
    # The hand-written corpus survives.
    seeds = (target / "datasets" / "samples" / "seed_samples.jsonl").read_text()
    assert "mine" in seeds

    forced = runner.invoke(
        app, ["scaffold", str(target), "--architecture", "causal_sft", "--force"]
    )
    assert forced.exit_code == 0, forced.output


def test_scaffold_rejects_an_unknown_architecture(tmp_path: Path) -> None:
    result = runner.invoke(
        app, ["scaffold", str(tmp_path / "x"), "--architecture", "not_an_arch"]
    )
    assert result.exit_code != 0


# --- validate --------------------------------------------------------------


def test_validate_reports_a_missing_declared_path(tmp_path: Path) -> None:
    mdir = _write_model(tmp_path, name="broken-paths")
    (mdir / "datasets" / "seeds.jsonl").unlink()
    result = runner.invoke(app, ["validate", str(mdir)])
    assert result.exit_code == 1
    assert "validate failed" in result.output


def test_validate_warns_on_unknown_config_keys(tmp_path: Path) -> None:
    mdir = _write_model(tmp_path, name="typos", extra="  tarjet_field: oops\n")
    result = runner.invoke(app, ["validate", str(mdir)])
    assert result.exit_code == 0, result.output
    assert "tarjet_field" in result.output


def test_validate_no_plugins_says_what_it_skipped(tmp_path: Path) -> None:
    mdir = _write_model(tmp_path, name="noplugins")
    result = runner.invoke(app, ["validate", str(mdir), "--no-plugins"])
    assert result.exit_code == 0, result.output
    assert "not verified" in result.output


# --- runs ------------------------------------------------------------------


def test_runs_on_a_model_with_no_runs(tmp_path: Path) -> None:
    mdir = _write_model(tmp_path, name="norun")
    result = runner.invoke(app, ["runs", str(mdir)])
    assert result.exit_code == 0
    assert "no runs yet" in result.output


def test_runs_survives_a_torn_last_line(tmp_path: Path) -> None:
    from maatml.config import load_model_def
    from maatml.runs import runs_path, start_run

    mdir = _write_model(tmp_path, name="torn")
    md = load_model_def(mdir)
    rec = start_run(md)
    with open(runs_path(md), "a", encoding="utf-8") as fh:
        fh.write('{"run_id": "truncated", "identi')

    result = runner.invoke(app, ["runs", str(mdir)])
    assert result.exit_code == 0, result.output
    assert rec.run_id in result.output
    assert (runs_path(md).with_name("runs.jsonl.corrupt")).is_file()


# --- user-error messages ---------------------------------------------------


def test_unparseable_model_yml_prints_one_line(tmp_path: Path) -> None:
    mdir = tmp_path / "bad-yaml"
    mdir.mkdir()
    (mdir / "model.yml").write_text("name: [unclosed\n", encoding="utf-8")
    result = runner.invoke(app, ["validate", str(mdir)])
    assert result.exit_code != 0


def test_missing_model_yml_names_the_file(tmp_path: Path) -> None:
    mdir = tmp_path / "empty-dir"
    mdir.mkdir()
    result = runner.invoke(app, ["plan", str(mdir)], catch_exceptions=True)
    assert result.exit_code != 0
    assert "model.yml" in str(result.output) + str(result.exception)


@pytest.mark.parametrize("debug", [False, True])
def test_main_prints_one_line_for_user_errors(monkeypatch, capsys, debug: bool) -> None:
    """maatml.cli.main turns known user errors into a single actionable line."""
    from maatml import cli

    def _boom(*args, **kwargs):
        raise FileNotFoundError("datasets/samples/seed_samples.jsonl not found")

    monkeypatch.setattr(cli, "app", _boom)
    monkeypatch.setitem(cli._STATE, "debug", debug)

    if debug:
        with pytest.raises(FileNotFoundError):
            cli.main()
        return

    with pytest.raises(SystemExit) as exit_info:
        cli.main()
    assert exit_info.value.code == 1
    out = capsys.readouterr().out
    assert "seed_samples.jsonl not found" in out
    assert "Traceback" not in out
    assert "--debug" in out
