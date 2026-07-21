"""Generator factory for ``maatml datagen`` (synthetic_scenes)."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from .synth import build_sample_row


def synthetic_scenes_generator(model_def: Any, seed: int = 0) -> Callable[[], dict]:
    """Return a zero-arg generate_fn that appends PNGs under datasets/samples/images."""
    size = int((getattr(model_def, "training", None) or {}).get("image_size") or 320)
    images_rel = "datasets/samples/images"
    images_dir = Path(model_def.resolve(images_rel))
    images_dir.mkdir(parents=True, exist_ok=True)

    state = {"i": 0}

    def generate_fn() -> dict:
        idx = state["i"]
        state["i"] += 1
        return build_sample_row(
            idx,
            base_seed=seed,
            size=size,
            image_rel=images_rel + "/{id}.png",
            images_dir=images_dir,
        )

    return generate_fn
