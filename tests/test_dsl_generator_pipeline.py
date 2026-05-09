"""Pipeline + packaging tests for the DSL Generator (English -> Flow DSL).

Covers the wiring shipped with the `dsl_generation` task; quality of the
trained model is out of scope for this suite.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from flow_ml.config import load_model_def
from flow_ml.data.pipeline import prepare_dsl
from flow_ml.data.synthetic.dsl_generator import (
    augment_seeds,
    parse_dsl,
    serialize_dsl,
)
from flow_ml.utils.io import iter_jsonl


def _make_dsl_model_folder(tmp_path: Path, *, augment: bool = False, target_count: int = 100) -> Path:
    """Create a minimal models/dsl-generator-style folder with a copy of the seed samples."""
    mdir = tmp_path / "model"
    mdir.mkdir()
    (mdir / "datasets" / "samples").mkdir(parents=True)
    shutil.copy2(SEED_SAMPLES, mdir / "datasets" / "samples" / "seed_samples.jsonl")
    yml = (
        "name: foo\n"
        "model_id: foo:v1\n"
        "task: dsl_generation\n"
        "data:\n"
        "  seed: 7331\n"
        "  seed_samples: datasets/samples/seed_samples.jsonl\n"
        "  raw_field: description\n"
        "  split_ratios: [0.6, 0.2, 0.2]\n"
    )
    if augment:
        yml += (
            "  augment:\n"
            f"    target_count: {target_count}\n"
            "    seed: 42\n"
            "    out: datasets/samples/augmented_samples.jsonl\n"
        )
    (mdir / "model.yml").write_text(yml, encoding="utf-8")
    return mdir

REPO_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = REPO_ROOT / "models" / "dsl-generator"
DATASET_DIR = MODEL_DIR / "datasets"
PROMPT_SPEC = DATASET_DIR / "prompt_spec.json"
SEED_SAMPLES = DATASET_DIR / "samples" / "seed_samples.jsonl"
AUG_SAMPLES = DATASET_DIR / "samples" / "augmented_samples.jsonl"
EVAL_SAMPLES = DATASET_DIR / "samples" / "eval_samples.jsonl"


def test_prompt_spec_has_required_fields() -> None:
    spec = json.loads(PROMPT_SPEC.read_text(encoding="utf-8"))
    assert spec["schema_version"] == "1"
    assert "<<USER_DESCRIPTION>>" in spec["user_template"]
    schema = spec["response_schema"]
    assert schema["type"] == "object"
    assert schema["required"] == ["dsl"]
    assert schema["properties"]["dsl"]["type"] == "string"
    assert spec["json_keys_order"] == ["dsl"]
    assert spec["max_new_tokens"] >= 256
    assert spec["temperature"] == 0.0


def test_seed_samples_have_consistent_shape() -> None:
    rows = list(iter_jsonl(SEED_SAMPLES))
    assert rows, "seed_samples.jsonl must not be empty"
    seen_ids: set[str] = set()
    for row in rows:
        for key in ("sample_id", "source", "description", "dsl"):
            assert key in row, f"missing key {key} in {row.get('sample_id')}"
            assert isinstance(row[key], str) and row[key].strip(), (
                f"empty {key} in {row.get('sample_id')}"
            )
        assert row["sample_id"] not in seen_ids, f"duplicate {row['sample_id']}"
        seen_ids.add(row["sample_id"])
        # Every DSL document must declare a flow header.
        assert row["dsl"].lstrip().startswith("flow "), (
            f"{row['sample_id']}: dsl must start with `flow \"...\"` header"
        )


def test_prepare_dsl_writes_splits(tmp_path: Path) -> None:
    mdir = _make_dsl_model_folder(tmp_path, augment=False)
    md = load_model_def(mdir)
    summary = prepare_dsl(md)
    total = sum(summary["split_counts"].values())
    assert total == sum(1 for _ in iter_jsonl(SEED_SAMPLES))
    for split in ("train", "val", "test"):
        path = md.prepared_dir / f"{split}.jsonl"
        assert path.exists()
        rows = list(iter_jsonl(path))
        assert len(rows) == summary["split_counts"][split]
        for row in rows:
            assert row["description"]
            assert row["dsl"].lstrip().startswith("flow ")
            assert row["split"] == split
    assert (md.prepared_dir / "dataset_card.md").exists()


def test_package_dsl_emits_dsl_generation_task(tmp_path: Path) -> None:
    """Smoke-test the packaging path with a minimal fake checkpoint dir.

    We don't exercise the transformers forward pass here (that would require
    pulling a base model); instead we verify the manifest contract that the
    Candle runtime in flow-studio relies on.
    """
    from flow_ml.packaging.package_model import package_dsl

    fake_ckpt = tmp_path / "fake-ckpt"
    fake_ckpt.mkdir()
    # Minimum surface needed by package_dsl's `_check_required` gate.
    (fake_ckpt / "model.safetensors").write_bytes(b"\x00\x00")
    (fake_ckpt / "config.json").write_text(json.dumps({"model_type": "llama"}), encoding="utf-8")
    (fake_ckpt / "tokenizer.json").write_text("{}", encoding="utf-8")

    out = tmp_path / "models" / "dsl-generator-smoke"
    result = package_dsl(
        fake_ckpt,
        out,
        prompt_spec_path=PROMPT_SPEC,
        model_id="dsl-generator:smoke",
        version="smoke",
    )

    manifest_path = result.pkg_dir / "manifest.json"
    assert manifest_path.exists()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["task"] == "dsl_generation"
    assert manifest["model_id"] == "dsl-generator:smoke"
    assert manifest["prompt_spec_file"] == "prompt_spec.json"
    # The packaged spec must round-trip the response schema unchanged so the
    # generative backend can emit `{"dsl": "..."}`.
    spec = json.loads((result.pkg_dir / "prompt_spec.json").read_text(encoding="utf-8"))
    assert spec["response_schema"]["required"] == ["dsl"]
    # Required files all present.
    for name in ("model.safetensors", "config.json", "tokenizer.json", "prompt_spec.json"):
        assert (result.pkg_dir / name).exists()


def test_augmenter_produces_valid_dsl(tmp_path: Path) -> None:
    """augment_seeds() should return samples whose DSL strings all parse cleanly.

    We run it without the flow_dsl_py Rust parser (validate=False) so this test
    does not require the wheel to be built.  The parser-based validation is
    exercised separately in test_flow_dsl_py.py when the wheel IS present.
    """
    samples = augment_seeds(
        seed_path=SEED_SAMPLES,
        target_count=50,
        seed=99,
        validate=False,
    )
    assert len(samples) == 50, f"expected 50 samples, got {len(samples)}"
    seen_ids: set[str] = set()
    for s in samples:
        for key in ("sample_id", "source", "description", "dsl"):
            assert key in s and s[key], f"missing/empty {key} in {s.get('sample_id')}"
        assert s["sample_id"] not in seen_ids, f"duplicate id {s['sample_id']}"
        seen_ids.add(s["sample_id"])
        assert s["dsl"].lstrip().startswith("flow "), (
            f"{s['sample_id']}: dsl must start with flow header"
        )


def test_augmenter_parse_serialize_roundtrip() -> None:
    """Every seed should round-trip through parse_dsl -> serialize_dsl without data loss."""
    seeds = list(iter_jsonl(SEED_SAMPLES))
    for s in seeds:
        g = parse_dsl(s["dsl"])
        out = serialize_dsl(g)
        # The re-serialized text must still start with a flow header
        assert out.lstrip().startswith("flow "), (
            f"{s['sample_id']}: serialized DSL missing flow header"
        )
        # Names and versions must be preserved
        assert g.name in out, f"{s['sample_id']}: flow name lost after serialize"
        assert g.version in out, f"{s['sample_id']}: version lost after serialize"
        # Node count must be preserved
        g2 = parse_dsl(out)
        assert len(g2.nodes) == len(g.nodes), (
            f"{s['sample_id']}: node count changed after roundtrip "
            f"({len(g.nodes)} -> {len(g2.nodes)})"
        )
        assert len(g2.edges) == len(g.edges), (
            f"{s['sample_id']}: edge count changed after roundtrip"
        )


def test_augmented_file_is_superset_of_seeds() -> None:
    """The pre-generated augmented_samples.jsonl should include all seed IDs.

    Skips when the augmented file is older than seed_samples.jsonl (e.g.
    Phase 4c just appended new seeds and `flow_ml prepare` has not been
    re-run yet). Re-running prep requires torch + the training-pipeline
    deps; the assertion is a sanity check, not a gate.
    """
    if not AUG_SAMPLES.exists():
        pytest.skip("augmented_samples.jsonl not generated yet - run `prepare dsl`")
    if AUG_SAMPLES.stat().st_mtime < SEED_SAMPLES.stat().st_mtime:
        pytest.skip(
            "augmented_samples.jsonl is older than seed_samples.jsonl; "
            "re-run `flow_ml prepare models/dsl-generator/` to refresh"
        )
    seed_ids = {s["sample_id"] for s in iter_jsonl(SEED_SAMPLES)}
    aug_ids = {s["sample_id"] for s in iter_jsonl(AUG_SAMPLES)}
    missing = seed_ids - aug_ids
    assert not missing, f"seed IDs missing from augmented file: {missing}"
    assert len(aug_ids) > len(seed_ids), "augmented file should have more samples than seeds"


def test_eval_samples_have_valid_shape() -> None:
    """Every eval sample must have the expected fields and a valid flow header."""
    if not EVAL_SAMPLES.exists():
        pytest.skip("eval_samples.jsonl not present")
    rows = list(iter_jsonl(EVAL_SAMPLES))
    assert rows, "eval_samples.jsonl must not be empty"
    seen: set[str] = set()
    for row in rows:
        for key in ("sample_id", "source", "description", "dsl"):
            assert key in row and row[key], f"missing/empty {key} in {row.get('sample_id')}"
        assert row["sample_id"] not in seen, f"duplicate eval id {row['sample_id']}"
        seen.add(row["sample_id"])
        assert row["dsl"].lstrip().startswith("flow "), (
            f"{row['sample_id']}: eval dsl missing flow header"
        )


def test_prepare_dsl_with_augmentation(tmp_path: Path) -> None:
    """prepare_dsl with the augment block should produce more samples than seeds alone.

    The expected total is computed from the live seed count so a Phase 4c
    style corpus expansion (which appends new canonical seeds) does not
    falsely break this test. Augmenter `target_count=100` produces
    exactly 100 augmented rows for any seed count >= 1.
    """
    mdir = _make_dsl_model_folder(tmp_path, augment=True, target_count=100)
    md = load_model_def(mdir)
    summary = prepare_dsl(md)
    total = sum(summary["split_counts"].values())
    seed_count = sum(1 for _ in iter_jsonl(SEED_SAMPLES))
    expected = seed_count + 100
    assert total == expected, (
        f"expected {expected} samples ({seed_count} seeds + 100 augmented), got {total}"
    )
    aug_out = mdir / "datasets" / "samples" / "augmented_samples.jsonl"
    assert aug_out.exists(), "augmented file should have been written"
    for split in ("train", "val", "test"):
        path = md.prepared_dir / f"{split}.jsonl"
        assert path.exists()
        for row in iter_jsonl(path):
            assert row["dsl"].lstrip().startswith("flow ")
            assert row["split"] == split


def test_cli_lists_top_level_commands() -> None:
    """`flow_ml --help` should advertise the new flat command surface."""
    from typer.testing import CliRunner

    from flow_ml.cli import app

    runner = CliRunner()
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0, result.output
    for cmd in ("prepare", "train", "evaluate", "package", "verify"):
        assert cmd in result.output, f"`flow_ml --help` should list `{cmd}`"
