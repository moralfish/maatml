"""Read-only environment and model-folder diagnostics behind ``maatml doctor``.

Answers "why did that not work here?" without running training: which optional
extras are installed, which device the CLI would pick, what the registries
hold (and what failed to load), and, for a model folder, whether its declared
paths, architecture, splits, and gates are actually in place.

Nothing here mutates state or imports a model folder's plugins beyond what
``load_model_def`` already does.
"""
from __future__ import annotations

import platform
import sys
from dataclasses import dataclass, field
from importlib import metadata
from pathlib import Path
from typing import Any, Optional

# (import name, why it matters) for the optional stacks maatml dispatches to.
_OPTIONAL_PACKAGES: tuple[tuple[str, str], ...] = (
    ("torch", "training and inference ([ml])"),
    ("transformers", "trainers and tokenizers ([ml])"),
    ("peft", "LoRA / QLoRA adapters ([ml])"),
    ("datasets", "preference trainers ([ml])"),
    ("safetensors", "checkpoint and export weights ([ml])"),
    ("trl", "DPO / ORPO ([pref])"),
    ("bitsandbytes", "4/8-bit quantized bases ([cuda])"),
    ("torchvision", "vision examples ([vision])"),
    ("onnxruntime", "ONNX export checks ([vision])"),
    ("httpx", "teacher-backed datagen ([teacher])"),
    ("jsonschema", "validator schema checks (core dependency)"),
)

OK = "ok"
WARN = "warn"
ERROR = "error"


@dataclass
class Check:
    """One diagnostic line: ``name``, ``status``, and a human-readable detail."""

    name: str
    status: str
    detail: str


@dataclass
class Diagnostics:
    sections: dict[str, list[Check]] = field(default_factory=dict)

    def add(self, section: str, name: str, status: str, detail: str) -> None:
        self.sections.setdefault(section, []).append(Check(name, status, detail))

    @property
    def errors(self) -> list[Check]:
        return [c for checks in self.sections.values() for c in checks if c.status == ERROR]

    def as_dict(self) -> dict[str, Any]:
        return {
            section: [
                {"name": c.name, "status": c.status, "detail": c.detail} for c in checks
            ]
            for section, checks in self.sections.items()
        }


def _version(package: str) -> Optional[str]:
    try:
        return metadata.version(package)
    except metadata.PackageNotFoundError:
        return None


def _environment(diag: Diagnostics) -> None:
    diag.add(
        "environment",
        "maatml",
        OK,
        _version("maatml") or "not installed (running from a checkout?)",
    )
    diag.add("environment", "python", OK, f"{platform.python_version()} at {sys.executable}")
    diag.add(
        "environment",
        "platform",
        OK,
        f"{platform.system()} {platform.release()} ({platform.machine()})",
    )


def _packages(diag: Diagnostics) -> None:
    for package, why in _OPTIONAL_PACKAGES:
        version = _version(package)
        if version is None:
            diag.add("packages", package, WARN, f"not installed: {why}")
        else:
            diag.add("packages", package, OK, version)


def _device(diag: Diagnostics) -> None:
    try:
        import torch  # noqa: F401
    except ImportError:
        diag.add(
            "device",
            "torch",
            WARN,
            "not installed, so training and evaluation are unavailable "
            '(pip install "maatml[ml]")',
        )
        return

    from .device import get_profile, is_distributed, resolve_device

    device = resolve_device("auto")
    profile = get_profile(device)
    diag.add("device", "auto resolves to", OK, str(device))
    diag.add(
        "device",
        f"profile {profile.name}",
        OK,
        f"mid_train_eval={profile.allow_mid_train_eval} "
        f"workers={profile.dataloader_workers} "
        f"grad_checkpointing={profile.allow_grad_checkpointing} "
        f"weights={profile.weights_dtype_policy} "
        f"quantized_load={profile.allow_quantized_load}",
    )
    diag.add("device", "distributed", OK, "yes" if is_distributed() else "no")


def _plugins(diag: Diagnostics) -> None:
    from .registry import discover_plugins, list_all_plugins, load_errors

    discover_plugins()
    # Predictors and exporters register on import of their modules.
    from .evaluation import predictors as _predictors  # noqa: F401

    for kind, entries in list_all_plugins().items():
        names = ", ".join(e.name for e in entries) or "(none)"
        diag.add("plugins", kind, OK, f"{len(entries)}: {names}")

    for source, error in load_errors():
        # An optional extra that is simply absent is a warning; anything else
        # is a plugin that meant to load and did not.
        status = WARN if source.startswith("module:maatml.") else ERROR
        diag.add("plugins", source, status, error)


def _count_rows(path: Path) -> int:
    with open(path, "r", encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def _model_folder(diag: Diagnostics, model_dir: Path) -> None:
    from .config import config_key_warnings, get_dataset_cfg, load_model_def
    from .evaluation.harness import GateConfigError, resolve_gate_spec, resolve_validator
    from .registry import FORMATS, TRAINERS
    from .runs import list_runs
    from .scaffold import normalize_architecture

    section = "model"
    try:
        md = load_model_def(model_dir)
    except Exception as exc:  # noqa: BLE001  doctor reports, never raises
        diag.add(section, str(model_dir), ERROR, f"failed to load model.yml: {exc}")
        return

    diag.add(section, "identity", OK, f"{md.identity} ({md.architecture})")

    arch = normalize_architecture(md.architecture)
    if TRAINERS.get(md.architecture) or TRAINERS.get(arch):
        diag.add(section, "architecture", OK, f"{md.architecture} is registered")
    else:
        diag.add(
            section,
            "architecture",
            ERROR,
            f"{md.architecture!r} has no registered trainer "
            f"(known: {', '.join(TRAINERS.names()) or 'none'})",
        )

    cfg = get_dataset_cfg(md)
    fmt = str(cfg.get("format", "jsonl_seed"))
    if FORMATS.get(fmt):
        diag.add(section, "dataset.format", OK, f"{fmt} is registered")
    else:
        diag.add(section, "dataset.format", ERROR, f"{fmt!r} is not registered")

    try:
        md.validate_paths()
        diag.add(section, "declared paths", OK, "all present")
    except FileNotFoundError as exc:
        diag.add(section, "declared paths", ERROR, str(exc).replace("\n", " "))

    for warning in config_key_warnings(md):
        diag.add(section, "config keys", WARN, warning)

    splits = []
    for split in ("train", "val", "test"):
        path = md.prepared_dir / f"{split}.jsonl"
        splits.append(f"{split}={_count_rows(path) if path.is_file() else 'missing'}")
    empty = [s for s in splits if s.endswith("=0") or s.endswith("=missing")]
    diag.add(
        section,
        "prepared splits",
        WARN if empty else OK,
        " ".join(splits) + ("  (run maatml prepare)" if empty else ""),
    )

    runs = list_runs(md)
    completed = [r for r in runs if r.status == "completed"]
    diag.add(
        section,
        "runs",
        OK if completed else WARN,
        f"{len(runs)} recorded, {len(completed)} completed"
        + ("" if completed else "  (run maatml train)"),
    )

    validator = (md.evaluation or {}).get("validator")
    if validator is None:
        diag.add(
            section,
            "evaluation.validator",
            WARN,
            "not configured: evaluate scores JSON parse only and datagen refuses "
            "to run without --allow-ungated",
        )
    else:
        try:
            resolve_validator(validator)
            diag.add(section, "evaluation.validator", OK, f"{validator} resolves")
        except GateConfigError as exc:
            diag.add(section, "evaluation.validator", ERROR, str(exc).replace("\n", " "))

    try:
        gates = resolve_gate_spec(md)
        diag.add(
            section,
            "evaluation.gates",
            OK,
            ", ".join(f"{k}>={v:g}" for k, v in sorted(gates.items())),
        )
    except GateConfigError:
        diag.add(
            section,
            "evaluation.gates",
            WARN,
            "none configured: `maatml evaluate --gate` fails rather than passing "
            "vacuously",
        )


def collect_diagnostics(model_dir: Optional[Path] = None) -> Diagnostics:
    """Gather environment (and optionally model-folder) diagnostics."""
    diag = Diagnostics()
    _environment(diag)
    _packages(diag)
    _device(diag)
    _plugins(diag)
    if model_dir is not None:
        _model_folder(diag, Path(model_dir))
    return diag
