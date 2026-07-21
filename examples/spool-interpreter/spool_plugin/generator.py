"""Spool generator factory for ``maatml datagen``.

Mirrors the category builders in ``scripts/build_seeds.py``. Registration
happens in package ``__init__.py`` so it re-binds after registry wipes.
"""
from __future__ import annotations

import importlib.util
import json
import random
from pathlib import Path
from typing import Any, Callable, Optional

from maatml.config import ModelDefinition
from maatml.utils.io import stable_hash


def _load_builders():
    """Import CATEGORY_BUILDERS from the example's build_seeds script."""
    script = Path(__file__).resolve().parents[1] / "scripts" / "build_seeds.py"
    if not script.is_file():
        raise FileNotFoundError(f"Spool seed builders not found: {script}")
    spec = importlib.util.spec_from_file_location("maatml._spool_build_seeds", script)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {script}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.CATEGORY_BUILDERS, getattr(mod, "_enrich_interp_fields", None)


def spool_generator(
    model_def: ModelDefinition,
    *,
    seed: int = 0,
    **_kwargs: Any,
) -> Callable[[], Optional[dict[str, Any]]]:
    """Return a generate_fn for :func:`maatml.data.gated.build_gated_corpus`."""
    builders, enrich = _load_builders()
    categories = list(builders.keys())
    rng = random.Random(seed)
    counter = {"n": 0}

    related_docs: dict = {}
    contracts_rel = (model_def.data or {}).get("contracts") or (
        model_def.dataset or {}
    ).get("contracts")
    if isinstance(contracts_rel, str):
        contracts_path = model_def.resolve(contracts_rel)
        if contracts_path.is_file():
            related_docs = (
                json.loads(contracts_path.read_text(encoding="utf-8")).get(
                    "related_docs_catalog"
                )
                or {}
            )

    def _generate() -> Optional[dict[str, Any]]:
        counter["n"] += 1
        category = rng.choice(categories)
        request, interp = builders[category](rng)
        if enrich is not None:
            enrich(rng, category, interp, related_docs)
        sid = f"syn-{category}-{stable_hash(category, counter['n'], seed)[:8]}"
        return {
            "sample_id": sid,
            "source": "synthetic:template",
            "family": f"spool:{category}",
            "category": category,
            "request": request,
            "expected_interpretation": interp,
        }

    return _generate
