"""JCL generator factory for ``maatml datagen``.

Thin wrapper around template + defect injection primitives already used by
``scripts/build_seeds.py``. Returns a zero-arg callable producing seed-shaped
dicts (``request`` + ``expected_validation_result``).

Registration happens in package ``__init__.py`` so it re-binds after registry
wipes (cached submodule imports do not re-run decorators).
"""
from __future__ import annotations

import random
from pathlib import Path
from typing import Any, Callable, Optional

from maatml.config import ModelDefinition
from maatml.utils.io import stable_hash

from .generator import (
    DEFAULT_TEMPLATE_DIR,
    INJECTORS,
    _load_templates,
    _render_template,
)
from .schemas import ErrorCategory

_SEVERITY: dict[str, str] = {
    ErrorCategory.missing_dd.value: "error",
    ErrorCategory.invalid_job_card.value: "error",
    ErrorCategory.invalid_exec_statement.value: "error",
    ErrorCategory.invalid_dataset_reference_structure.value: "error",
    ErrorCategory.unresolved_symbolic_parameter.value: "error",
    ErrorCategory.continuation_error.value: "error",
    ErrorCategory.other.value: "warning",
}

_MESSAGES: dict[str, str] = {
    ErrorCategory.missing_dd.value: "DD statement missing the 'DD' keyword.",
    ErrorCategory.invalid_job_card.value: "JOB card is malformed.",
    ErrorCategory.invalid_exec_statement.value: "EXEC statement malformed.",
    ErrorCategory.invalid_dataset_reference_structure.value: "Dataset name structure invalid.",
    ErrorCategory.unresolved_symbolic_parameter.value: "Symbolic parameter unresolved.",
    ErrorCategory.continuation_error.value: "Continuation line must begin with //.",
    ErrorCategory.other.value: "Statement label must begin with //.",
}

_ERROR_CATS = [c for c in ErrorCategory if c is not ErrorCategory.none]


def _template_dir(model_def: ModelDefinition) -> Path:
    cfg = {**(model_def.data or {}), **(model_def.dataset or {})}
    rel = cfg.get("template_dir")
    if isinstance(rel, str):
        path = model_def.resolve(rel)
        if path.is_dir():
            return path
    candidate = model_def.model_dir / "datasets" / "templates"
    return candidate if candidate.is_dir() else DEFAULT_TEMPLATE_DIR


def jcl_generator(
    model_def: ModelDefinition,
    *,
    seed: int = 0,
    **_kwargs: Any,
) -> Callable[[], Optional[dict[str, Any]]]:
    """Return a generate_fn for :func:`maatml.data.gated.build_gated_corpus`."""
    rng = random.Random(seed)
    templates = _load_templates(_template_dir(model_def))
    counter = {"n": 0}

    def _generate() -> Optional[dict[str, Any]]:
        counter["n"] += 1
        idx = counter["n"]
        # ~20% valid, rest cycle error categories.
        if rng.random() < 0.2:
            template_id, raw = rng.choice(templates)
            rendered = _render_template(rng, raw).rstrip() + "\n"
            return {
                "sample_id": f"syn-valid-{stable_hash(template_id, seed, idx)[:8]}",
                "source": f"synthetic:{template_id}",
                "family": template_id,
                "category": "valid",
                "request": rendered,
                "expected_validation_result": {
                    "valid": True,
                    "errors": [],
                    "confidence": round(rng.uniform(0.90, 0.97), 2),
                },
            }

        category = rng.choice(_ERROR_CATS)
        template_id, raw = rng.choice(templates)
        rendered = _render_template(rng, raw)
        lines = rendered.splitlines()
        result = INJECTORS[category].inject(rng, lines)
        if result is None:
            return None
        new_lines, error_line, error_column, suggestion = result
        error_obj: dict[str, Any] = {
            "line": int(error_line),
            "severity": _SEVERITY[category.value],
            "code": category.value,
            "message": _MESSAGES[category.value],
            "suggestion": suggestion,
        }
        if error_column is not None:
            error_obj["column"] = int(error_column)
        return {
            "sample_id": f"syn-{category.value}-{stable_hash(template_id, seed, idx)[:8]}",
            "source": f"synthetic:{template_id}",
            "family": template_id,
            "category": category.value,
            "request": "\n".join(new_lines).rstrip() + "\n",
            "expected_validation_result": {
                "valid": False,
                "errors": [error_obj],
                "confidence": round(rng.uniform(0.82, 0.95), 2),
            },
        }

    return _generate
