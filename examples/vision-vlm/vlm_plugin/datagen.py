"""Generator factory for ``maatml datagen`` (described_scenes)."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from .describe import describe_from_expected
from .synth import build_sample_row as _build_vision_row


def build_described_row(
    index: int,
    *,
    base_seed: int = 0,
    size: int = 320,
    image_rel: str,
    images_dir: Path,
) -> dict[str, Any]:
    """Render a scene and attach a deterministic description target."""
    row = _build_vision_row(
        index,
        base_seed=base_seed,
        size=size,
        image_rel=image_rel,
        images_dir=images_dir,
    )
    expected = row.pop("expected")
    description, gt = describe_from_expected(expected)
    row["expected_output"] = {"description": description}
    row["gt"] = gt
    row["source"] = "synthetic:described_scene"
    return row


def described_scenes_generator(model_def: Any, seed: int = 0) -> Callable[[], dict]:
    """Return a zero-arg generate_fn that appends PNGs under datasets/samples/images."""
    size = int((getattr(model_def, "training", None) or {}).get("image_size") or 320)
    images_rel = "datasets/samples/images"
    images_dir = Path(model_def.resolve(images_rel))
    images_dir.mkdir(parents=True, exist_ok=True)
    state = {"i": 0}

    def generate_fn() -> dict:
        idx = state["i"]
        state["i"] += 1
        return build_described_row(
            idx,
            base_seed=seed,
            size=size,
            image_rel=images_rel + "/{id}.png",
            images_dir=images_dir,
        )

    return generate_fn
