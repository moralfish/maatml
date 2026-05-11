"""Model definition schema (`models/<name>/model.yml`).

A single `model.yml` per model is the source of truth for everything in that
model's lifecycle: data preparation, training, smoke training, evaluation,
and packaging.  This module owns the schema and the loader.

The nested `data:`, `training:`, `smoke:`, and `packaging:` sections are
intentionally typed as plain dicts so each pipeline stage (``prepare_jcl``,
``train_dsl``, etc.) can validate them against its existing Pydantic config
class without forcing a single rigid super-schema for three very different
models.

Layout:

    models/<name>/
      model.yml
      README.md
      datasets/
        schema.json
        prompt_spec.json        (optional)
        samples/
          seed_samples.jsonl    (tracked)
          augmented_*.jsonl     (gitignored, generated)
      output/                   (gitignored content)
        prepared/{train,val,test}.jsonl
        checkpoints/<run-name>/
        eval/<report>.{json,md}
        dist/<model_id>-<version>/
        dist/<model_id>-<version>.fm
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field

from .utils.io import read_yaml


class PackagingSpec(BaseModel):
    """Manifest fields that aren't derivable from `training:` or runtime introspection.

    Mirrors the relevant subset of ``ModelManifest``; ``ConfidenceThresholds`` is
    expressed here as a plain dict so the YAML stays light.
    """

    model_config = ConfigDict(extra="forbid")

    max_input_tokens: int = Field(gt=0, default=2048)
    expected_latency_ms: int = Field(gt=0, default=2000)
    confidence_thresholds: dict[str, float] = Field(
        default_factory=lambda: {"high": 0.9, "low": 0.6}
    )
    # Half-precision packaging knob: defaults to `"f32"` for back-compat.
    # Set to `"f16"` (recommended) or `"bf16"` for 7B+ bases. Only
    # consumed by `package_dsl` today; `package_jcl` and `package_spool`
    # ignore it because their bases are small enough that the F32 path
    # ships fine.
    weights_dtype: str = "f32"


class ModelDefinition(BaseModel):
    """Top-level schema for `models/<name>/model.yml`."""

    model_config = ConfigDict(extra="forbid")

    # Identity / runtime contract
    name: str = Field(..., description="Folder name; e.g. 'dsl-generator'")
    model_id: str = Field(..., description="What Flow Studio sees, e.g. 'dsl-generator:v1'")
    task: str = Field(
        ...,
        description=(
            "Pipeline dispatch key. One of: jcl_validation, spool_interpretation, "
            "dsl_generation, agent_planning"
        ),
    )
    runtime: str = "candle"
    # Architecture dispatch hint for the CLI's `train` subcommand. Allowed:
    # `generative` (default — Qwen3+LoRA SFT path used by Flow Graph
    # Generator), `classifier` (ModernBERT multi-head for JCL Validator
    # v2), `seq2seq` (T5/BART encoder-decoder for Spool Interpreter v2).
    # Loader uses this to route to the right trainer module.
    architecture: str = "generative"
    version: str = "v1"
    description: str = ""
    base_model: Optional[str] = None

    # The three nested sections are dicts so each pipeline stage validates them
    # against its existing typed config (JclTrainConfig, SpoolTrainConfig, ...).
    data: dict[str, Any] = Field(default_factory=dict)
    training: dict[str, Any] = Field(default_factory=dict)
    smoke: dict[str, Any] = Field(default_factory=dict)
    packaging: PackagingSpec = Field(default_factory=PackagingSpec)

    # ----- Filled in by load_model_def, not present in YAML -----
    model_dir: Path = Field(default_factory=Path, exclude=True)

    # --- helpers ---------------------------------------------------------

    def resolve(self, rel: str | Path) -> Path:
        """Resolve a YAML-declared path relative to the model folder."""
        p = Path(rel)
        return p if p.is_absolute() else (self.model_dir / p).resolve()

    @property
    def output_dir(self) -> Path:
        """`models/<name>/output/` - root of all generated artifacts."""
        return self.model_dir / "output"

    @property
    def prepared_dir(self) -> Path:
        """`models/<name>/output/prepared/` - train/val/test JSONL splits."""
        return self.output_dir / "prepared"

    @property
    def checkpoints_dir(self) -> Path:
        """`models/<name>/output/checkpoints/`."""
        return self.output_dir / "checkpoints"

    @property
    def eval_dir(self) -> Path:
        """`models/<name>/output/eval/`."""
        return self.output_dir / "eval"

    @property
    def dist_dir(self) -> Path:
        """`models/<name>/output/dist/` - packaged outputs (.fm + unpacked dir)."""
        return self.output_dir / "dist"

    def merged_smoke(self) -> dict[str, Any]:
        """Return ``training`` overlaid with ``smoke`` overrides.

        Used by `--smoke` to run a fast variant of the same training loop without
        a separate config file.
        """
        merged = dict(self.training)
        merged.update(self.smoke or {})
        # `smoke.base_model` overrides `training.model_id` if present
        if "base_model" in (self.smoke or {}):
            merged["model_id"] = self.smoke["base_model"]
            merged.pop("base_model", None)
        return merged


def load_model_def(model_dir: str | Path) -> ModelDefinition:
    """Read `<model_dir>/model.yml` and return a populated ModelDefinition.

    The returned object's `model_dir` attribute is set to the absolute path of
    the folder so `resolve(...)` and `output_dir` etc. work correctly.
    """
    model_dir = Path(model_dir).resolve()
    if not model_dir.is_dir():
        raise FileNotFoundError(f"Model directory not found: {model_dir}")
    yml_path = model_dir / "model.yml"
    if not yml_path.is_file():
        raise FileNotFoundError(
            f"{yml_path} not found - each model folder must contain model.yml"
        )

    raw = read_yaml(yml_path)
    if not isinstance(raw, dict):
        raise ValueError(f"{yml_path}: top-level must be a mapping")

    md = ModelDefinition(**raw)
    # Pydantic doesn't see `model_dir` from the YAML (it's not in the file);
    # set it post-construction.
    object.__setattr__(md, "model_dir", model_dir)
    return md
