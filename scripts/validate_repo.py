from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from flow_ml.config import ModelDefinition, load_model_def  # noqa: E402


@dataclass(frozen=True)
class ModelScaffold:
    task: str
    required_files: tuple[str, ...]
    required_dirs: tuple[str, ...] = ("datasets", "output")


MODELS: dict[str, ModelScaffold] = {
    "jcl-validator": ModelScaffold(
        task="jcl_validation",
        required_files=(
            "README.md",
            "model.yml",
            "datasets/schema.json",
            "datasets/label_taxonomy.md",
            "datasets/samples/.gitkeep",
            "output/.gitkeep",
        ),
        required_dirs=("datasets", "datasets/templates", "datasets/samples", "output"),
    ),
    "spool-interpreter": ModelScaffold(
        task="spool_interpretation",
        required_files=(
            "README.md",
            "model.yml",
            "datasets/schema.json",
            "datasets/label_taxonomy.md",
            "datasets/prompt_spec.json",
            "datasets/samples/seed_samples.jsonl",
            "output/.gitkeep",
        ),
        required_dirs=("datasets", "datasets/samples", "output"),
    ),
    "dsl-generator": ModelScaffold(
        task="dsl_generation",
        required_files=(
            "README.md",
            "model.yml",
            "datasets/schema.json",
            "datasets/grammar.md",
            "datasets/prompt_spec.json",
            "datasets/samples/seed_samples.jsonl",
            "datasets/samples/eval_samples.jsonl",
            "output/.gitkeep",
        ),
        required_dirs=("datasets", "datasets/samples", "output"),
    ),
    "agent-planner": ModelScaffold(
        task="agent_planning",
        required_files=(
            "README.md",
            "model.yml",
            "datasets/schema.json",
            "datasets/prompt_spec.json",
            "datasets/samples/seed_samples.jsonl",
            "datasets/samples/eval_samples.jsonl",
            "output/.gitkeep",
        ),
        required_dirs=("datasets", "datasets/samples", "output"),
    ),
}


def _display(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def _require_file(errors: list[str], path: Path) -> None:
    if not path.is_file():
        errors.append(f"missing file: {_display(path)}")


def _require_dir(errors: list[str], path: Path) -> None:
    if not path.is_dir():
        errors.append(f"missing directory: {_display(path)}")


def _require_nonempty_dir(errors: list[str], path: Path, *, suffix: str | None = None) -> None:
    if not path.is_dir():
        errors.append(f"missing directory: {_display(path)}")
        return
    entries: Iterable[Path] = path.iterdir()
    if suffix is not None:
        entries = (entry for entry in entries if entry.name.endswith(suffix))
    if not any(entries):
        kind = f"*{suffix}" if suffix else "entries"
        errors.append(f"empty directory: {_display(path)} has no {kind}")


def _check_declared_data_paths(errors: list[str], model_def: ModelDefinition) -> None:
    data = model_def.data
    file_keys = ("schema", "prompt_spec", "seed_samples", "benchmark_samples")
    for key in file_keys:
        if key in data:
            _require_file(errors, model_def.resolve(data[key]))

    for source in data.get("sources", []):
        _require_file(errors, model_def.resolve(source))

    if "template_dir" in data:
        _require_nonempty_dir(errors, model_def.resolve(data["template_dir"]), suffix=".jcl")

    augment = data.get("augment")
    if isinstance(augment, dict) and "out" in augment:
        out_parent = model_def.resolve(augment["out"]).parent
        _require_dir(errors, out_parent)


def _check_model(name: str, scaffold: ModelScaffold, errors: list[str]) -> None:
    model_dir = ROOT / "models" / name
    _require_dir(errors, model_dir)
    for rel_dir in scaffold.required_dirs:
        _require_dir(errors, model_dir / rel_dir)
    for rel_file in scaffold.required_files:
        _require_file(errors, model_dir / rel_file)

    try:
        model_def = load_model_def(model_dir)
    except Exception as exc:  # noqa: BLE001
        errors.append(f"{_display(model_dir / 'model.yml')}: failed to load: {exc}")
        return

    if model_def.name != name:
        errors.append(f"{_display(model_dir / 'model.yml')}: name is {model_def.name!r}")
    if model_def.task != scaffold.task:
        errors.append(
            f"{_display(model_dir / 'model.yml')}: task is {model_def.task!r}, "
            f"expected {scaffold.task!r}"
        )
    if model_def.runtime != "candle":
        errors.append(f"{_display(model_dir / 'model.yml')}: runtime is not 'candle'")

    _check_declared_data_paths(errors, model_def)


def main() -> int:
    errors: list[str] = []
    for name, scaffold in MODELS.items():
        _check_model(name, scaffold, errors)

    if errors:
        print("validate_repo: failed")
        for error in errors:
            print(f"- {error}")
        return 1

    print(f"validate_repo: ok ({len(MODELS)} model scaffolds)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
