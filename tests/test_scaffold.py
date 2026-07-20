"""Tests for model scaffolding and validate_model_dir."""
from __future__ import annotations

from pathlib import Path

from flow_ml.scaffold import scaffold_model, validate_model_dir


def test_scaffold_causal_sft_and_validate(tmp_path: Path) -> None:
    target = tmp_path / "my-sft"
    scaffold_model(target, architecture="causal_sft", name="my-sft")
    assert (target / "model.yml").is_file()
    assert (target / "README.md").is_file()
    assert (target / ".gitignore").is_file()
    assert (target / "datasets" / "schema.json").is_file()
    assert (target / "datasets" / "prompt_spec.json").is_file()
    assert (target / "datasets" / "samples" / "seed_samples.jsonl").is_file()
    errors = validate_model_dir(target)
    assert errors == [], errors


def test_scaffold_classifier(tmp_path: Path) -> None:
    target = tmp_path / "toy-jcl"
    scaffold_model(target, architecture="classifier")
    body = (target / "model.yml").read_text(encoding="utf-8")
    assert "architecture: classifier" in body
    assert "expected_validation_result" in (
        target / "datasets" / "samples" / "seed_samples.jsonl"
    ).read_text(encoding="utf-8")
    # Classifier sanitize tag jcl is fine; validate may fail if contracts path
    # is not declared — scaffold uses datasets/schema.json only.
    errors = validate_model_dir(target)
    assert errors == [], errors
