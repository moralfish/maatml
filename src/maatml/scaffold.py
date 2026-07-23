"""Scaffold a new model folder and validate existing ones."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

from .config import ModelDefinition, get_dataset_cfg, load_model_def
from .registry import (
    FORMATS,
    SCAFFOLD_HOOKS,
    TRAINERS,
    discover_plugins,
)

# Architecture aliases accepted in model.yml / --architecture.
_ARCH_ALIASES: dict[str, str] = {
    "classifier": "multi_head_classifier",
}


def normalize_architecture(architecture: str) -> str:
    return _ARCH_ALIASES.get(architecture, architecture)


def _training_defaults(architecture: str) -> dict[str, Any]:
    """Best-effort defaults mirrored from trainer configs (no ML imports)."""
    arch = normalize_architecture(architecture)
    if arch == "causal_sft":
        return {
            "model_id": "Qwen/Qwen3-1.7B",
            "max_input_tokens": 4096,
            "batch_size": 2,
            "grad_accum": 8,
            "learning_rate": 1e-4,
            "epochs": 4.0,
            "weight_decay": 0.01,
            "warmup_ratio": 0.05,
            "seed": 7331,
            "precision": "bf16",
            "grad_checkpointing": False,
            "eval_steps": 9999,
            "save_steps": 200,
            "logging_steps": 20,
            "max_steps": -1,
        }
    if arch in ("dpo", "orpo"):
        return {
            "model_id": "Qwen/Qwen3-0.6B",
            "max_input_tokens": 2048,
            "batch_size": 1,
            "grad_accum": 8,
            "learning_rate": 5.0e-5,
            "epochs": 1.0,
            "weight_decay": 0.0,
            "warmup_ratio": 0.1,
            "seed": 7331,
            "precision": "bf16",
            "grad_checkpointing": False,
            "eval_steps": 9999,
            "save_steps": 200,
            "logging_steps": 10,
            "max_steps": -1,
            "beta": 0.1,
            "lora": {
                "enabled": True,
                "r": 16,
                "alpha": 32,
                "dropout": 0.05,
                "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
            },
        }
    if arch == "seq2seq":
        return {
            "model_id": "google/flan-t5-base",
            "source_max_len": 1024,
            "target_max_len": 512,
            "batch_size": 8,
            "grad_accum": 2,
            "learning_rate": 3.0e-5,
            "epochs": 6,
            "weight_decay": 0.01,
            "warmup_ratio": 0.06,
            "seed": 1337,
            "precision": "bf16",
            "grad_checkpointing": False,
            "eval_steps": 9999,
            "save_steps": 250,
            "logging_steps": 25,
            "max_steps": -1,
            "generation": {"num_beams": 1, "max_new_tokens": 512},
        }
    if arch == "multi_head_classifier":
        return {
            "model_id": "answerdotai/ModernBERT-base",
            "max_input_tokens": 2048,
            "batch_size": 16,
            "grad_accum": 1,
            "learning_rate": 2.0e-5,
            "epochs": 4,
            "weight_decay": 0.01,
            "warmup_ratio": 0.06,
            "seed": 1337,
            "precision": "bf16",
            "grad_checkpointing": False,
            "eval_steps": 9999,
            "save_steps": 250,
            "logging_steps": 25,
            "max_steps": -1,
        }
    return {
        "model_id": "CHANGE_ME",
        "batch_size": 2,
        "epochs": 1,
        "learning_rate": 1.0e-4,
        "seed": 7,
    }


def _seed_row(architecture: str) -> dict[str, Any]:
    arch = normalize_architecture(architecture)
    if arch in ("dpo", "orpo"):
        return {
            "sample_id": "pref-001",
            "source": "scaffold",
            "family": "example",
            "prompt": "Say hello.",
            "chosen": "Hello!",
            "rejected": "I refuse.",
        }
    if arch == "multi_head_classifier":
        return {
            "sample_id": "seed-001",
            "source": "scaffold",
            "family": "example",
            "category": "basic",
            "request": "//JOBNAME JOB CLASS=A\n//STEP1 EXEC PGM=IEFBR14\n",
            "expected_output": {
                "valid": True,
                "errors": [],
                "confidence": 0.9,
            },
        }
    if arch == "seq2seq":
        return {
            "sample_id": "seed-001",
            "source": "scaffold",
            "family": "example",
            "category": "basic",
            "request": "IEFC452I JOB JOBNAME - JOB NOT RUN - JCL ERROR",
            "expected_output": {
                "summary": "JCL error prevented job run",
                "status": "failed",
                "returnCode": None,
                "rootCause": "JCL error",
                "suggestedFix": "Fix JCL and resubmit",
                "explanation": "IEFC452I indicates the job was not selected.",
                "relatedDocs": [],
                "failureCategory": "other",
                "confidence": 0.8,
            },
        }
    # causal_sft default
    return {
        "sample_id": "seed-001",
        "source": "scaffold",
        "family": "example",
        "category": "basic",
        "request": "Say hello.",
        "expected_output": {"answer": "Hello!"},
    }


def _dataset_section(architecture: str) -> dict[str, Any]:
    arch = normalize_architecture(architecture)
    if arch in ("dpo", "orpo"):
        return {
            "format": "preference_jsonl",
            "group_by": "family",
            "seed_samples": "datasets/samples/seed_samples.jsonl",
            "schema": "datasets/schema.json",
            "split_ratios": [0.8, 0.1, 0.1],
            "sanitize": [],
            "seed": 7,
        }
    base = {
        "format": "jsonl_seed",
        "request_field": "request",
        "group_by": "family",
        "seed_samples": "datasets/samples/seed_samples.jsonl",
        "schema": "datasets/schema.json",
        "split_ratios": [0.8, 0.1, 0.1],
        "sanitize": [],
        "seed": 7,
    }
    if arch == "multi_head_classifier":
        base["target_field"] = "expected_output"
    elif arch == "seq2seq":
        base["target_field"] = "expected_output"
        base["prompt_spec"] = "datasets/prompt_spec.json"
        base["source_prefix"] = ""
    else:
        base["target_field"] = "expected_output"
        base["prompt_spec"] = "datasets/prompt_spec.json"
        base["user_placeholder"] = "<<USER_REQUEST>>"
    return base


def _evaluation_section(architecture: str) -> dict[str, Any]:
    arch = normalize_architecture(architecture)
    if arch in ("dpo", "orpo"):
        return {"predictor": "causal_sft", "metrics": []}
    if arch == "multi_head_classifier":
        return {"predictor": "classifier", "metrics": []}
    if arch == "seq2seq":
        return {"predictor": "seq2seq", "metrics": []}
    return {"predictor": "causal_sft", "metrics": []}


def _yaml_dump(data: dict[str, Any], *, indent: int = 0) -> str:
    """Minimal YAML emitter for scaffolded model.yml (no pyyaml dependency)."""
    lines: list[str] = []

    def emit(key: str, value: Any, level: int) -> None:
        p = "  " * level
        if isinstance(value, dict):
            lines.append(f"{p}{key}:")
            if not value:
                return
            for k, v in value.items():
                emit(str(k), v, level + 1)
        elif isinstance(value, list):
            if not value:
                lines.append(f"{p}{key}: []")
            elif all(not isinstance(x, (dict, list)) for x in value):
                inner = ", ".join(_scalar(x) for x in value)
                lines.append(f"{p}{key}: [{inner}]")
            else:
                lines.append(f"{p}{key}:")
                for item in value:
                    if isinstance(item, dict):
                        lines.append(f"{p}  -")
                        for k, v in item.items():
                            emit(str(k), v, level + 2)
                    else:
                        lines.append(f"{p}  - {_scalar(item)}")
        else:
            lines.append(f"{p}{key}: {_scalar(value)}")

    for k, v in data.items():
        emit(str(k), v, indent)
    return "\n".join(lines) + "\n"


def _scalar(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return "null"
    if isinstance(value, float):
        # Prefer scientific for small LRs
        if 0 < abs(value) < 1e-2 or abs(value) >= 1e4:
            return f"{value:.1e}"
        return repr(value)
    if isinstance(value, int):
        return str(value)
    text = str(value)
    if any(c in text for c in ":#{}[]&*!|>'\"%@`") or text != text.strip():
        return json.dumps(text)
    return text


_README_TMPL = """# {name}

Scaffolded maatml model (`architecture: {architecture}`).

## Lifecycle

```bash
maatml prepare {name}/
maatml train {name}/ --smoke
maatml train {name}/
maatml evaluate {name}/
```

Edit `model.yml`, add seed samples under `datasets/samples/`, then prepare/train.
"""


_PROMPT_SPEC = {
    "system": "You are a helpful assistant. Reply with a single JSON object.",
    "user_template": "<<USER_REQUEST>>",
    "response_format": "json",
}


_SCHEMA_PLACEHOLDER = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "title": "ScaffoldOutput",
    "type": "object",
    "additionalProperties": True,
}


def scaffold_model(
    target_dir: Path,
    *,
    architecture: str,
    name: Optional[str] = None,
    force: bool = False,
) -> Path:
    """Create a model folder with model.yml, datasets, README, and .gitignore.

    Refuses to overwrite an existing ``model.yml`` or seed corpus unless
    ``force`` is set, so a mistaken re-scaffold cannot destroy hand-edited
    config or a curated seed file.
    """
    discover_plugins()
    arch = normalize_architecture(architecture)
    if TRAINERS.get(arch) is None and TRAINERS.get(architecture) is None:
        known = ", ".join(TRAINERS.names()) or "(none)"
        raise ValueError(
            f"Unknown architecture {architecture!r}. Registered trainers: {known}"
        )

    target_dir = Path(target_dir).resolve()
    folder_name = name or target_dir.name
    target_dir.mkdir(parents=True, exist_ok=True)

    seed_path = target_dir / "datasets" / "samples" / "seed_samples.jsonl"
    if not force:
        clash = [str(p) for p in (target_dir / "model.yml", seed_path) if p.exists()]
        if clash:
            raise FileExistsError(
                "refusing to overwrite existing "
                + ", ".join(clash)
                + ". Pass --force to regenerate (this replaces model.yml and the "
                "seed corpus)."
            )

    training = _training_defaults(architecture)
    smoke = {
        "epochs": 1,
        "max_steps": 4,
        "batch_size": 1,
        "grad_accum": 1,
    }
    dataset = _dataset_section(architecture)
    evaluation = _evaluation_section(architecture)

    model_yml: dict[str, Any] = {
        "name": folder_name,
        "model_id": folder_name,
        "task": folder_name.replace("-", "_"),
        "architecture": architecture,
        "version": "0.1.0",
        "description": f"Scaffolded {architecture} model.",
        "base_model": training.get("model_id", "CHANGE_ME"),
        "dataset": dataset,
        "training": training,
        "smoke": smoke,
        "evaluation": evaluation,
        "packaging": {
            "max_input_tokens": int(training.get("max_input_tokens") or 2048),
            "expected_latency_ms": 2000,
            "weights_dtype": "f16",
        },
    }

    (target_dir / "model.yml").write_text(_yaml_dump(model_yml), encoding="utf-8")
    (target_dir / "README.md").write_text(
        _README_TMPL.format(name=folder_name, architecture=architecture),
        encoding="utf-8",
    )
    (target_dir / ".gitignore").write_text("output/\n", encoding="utf-8")

    samples_dir = target_dir / "datasets" / "samples"
    samples_dir.mkdir(parents=True, exist_ok=True)
    seed_path.write_text(
        json.dumps(_seed_row(architecture), ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (target_dir / "datasets" / "schema.json").write_text(
        json.dumps(_SCHEMA_PLACEHOLDER, indent=2) + "\n", encoding="utf-8"
    )

    if normalize_architecture(architecture) in ("causal_sft", "seq2seq") or (
        "prompt_spec" in dataset
    ):
        (target_dir / "datasets" / "prompt_spec.json").write_text(
            json.dumps(_PROMPT_SPEC, indent=2) + "\n", encoding="utf-8"
        )

    # Optional architecture/task hooks.
    for hook_name in (architecture, arch, folder_name):
        hook = SCAFFOLD_HOOKS.get(hook_name)
        if hook is not None:
            hook(target_dir, architecture=architecture, name=folder_name)

    return target_dir


def validate_model_dir(model_dir: Path | str, *, load_plugins: bool = True) -> list[str]:
    """Validate a model folder; return a list of error strings (empty = OK).

    When ``load_plugins`` is False the model.yml schema and declared paths are
    still checked, but no trainer or model-folder plugin code is imported, and
    the architecture / dataset.format registration checks are skipped (the
    registries are intentionally empty). This lets ``maatml validate`` lint an
    untrusted folder without executing its plugins.
    """
    if load_plugins:
        discover_plugins()
    model_dir = Path(model_dir).resolve()
    errors: list[str] = []

    if not (model_dir / "model.yml").is_file():
        return [f"missing model.yml under {model_dir}"]

    try:
        md = load_model_def(model_dir, load_plugins=load_plugins)
    except Exception as exc:  # noqa: BLE001
        return [f"failed to load model.yml: {exc}"]

    try:
        md.validate_paths()
    except FileNotFoundError as exc:
        errors.append(str(exc))

    if not load_plugins:
        # Registration checks need the (unloaded) plugin registries; skip them.
        return errors

    arch = normalize_architecture(md.architecture)
    if TRAINERS.get(arch) is None and TRAINERS.get(md.architecture) is None:
        known = ", ".join(TRAINERS.names()) or "(none)"
        errors.append(
            f"architecture {md.architecture!r} is not a registered trainer. "
            f"Known: {known}"
        )

    cfg = get_dataset_cfg(md)
    fmt = cfg.get("format")
    if fmt is not None and FORMATS.get(str(fmt)) is None:
        known = ", ".join(FORMATS.names()) or "(none)"
        errors.append(
            f"dataset.format {fmt!r} is not registered. Known: {known}"
        )

    return errors


def validate_model(model_dir: Path | str) -> ModelDefinition:
    """Validate and return the loaded ModelDefinition; raise ValueError on errors."""
    errors = validate_model_dir(model_dir)
    if errors:
        raise ValueError("validate failed:\n  - " + "\n  - ".join(errors))
    return load_model_def(model_dir)
