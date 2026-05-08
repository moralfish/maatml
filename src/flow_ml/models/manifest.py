from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class ConfidenceThresholds(BaseModel):
    model_config = ConfigDict(extra="forbid")

    high: float = Field(ge=0.0, le=1.0, default=0.9)
    low: float = Field(ge=0.0, le=1.0, default=0.6)


class ModelManifest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model_id: str
    runtime: str = "candle"
    task: str
    weights: str = "model.safetensors"
    # Tensor dtype the runtime should load `weights` with. Defaults to
    # `"f32"` for backwards compatibility with previously-shipped
    # packages. Set to `"f16"` for 7B+ bases so the on-disk safetensors
    # stays at half precision and the runtime mmaps it directly without
    # having to dequantize on load. Mirrors the
    # `flow-model-runtime::ModelManifest::weights_dtype` field exactly.
    weights_dtype: str = "f32"
    tokenizer: str = "tokenizer.json"
    config: str = "config.json"
    labels_file: Optional[str] = None
    prompt_spec_file: Optional[str] = None
    max_input_tokens: int = Field(gt=0)
    expected_latency_ms: int = Field(gt=0)
    version: str = "v1"
    base_checkpoint: Optional[str] = None
    confidence_thresholds: ConfidenceThresholds = Field(default_factory=ConfidenceThresholds)
    sha256: dict[str, str] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=_utcnow)

    def write(self, path: str | Path) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        payload = self.model_dump(mode="json")
        p.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        return p

    @classmethod
    def read(cls, path: str | Path) -> "ModelManifest":
        return cls.model_validate_json(Path(path).read_text(encoding="utf-8"))
