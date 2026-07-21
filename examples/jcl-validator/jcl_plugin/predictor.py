"""JCL-shaped predictor: multi-head outputs → JclValidationResult JSON."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

from maatml.config import ModelDefinition
from maatml.evaluation.harness import resolve_eval_asset
from maatml.evaluation.predictors import MultiHeadClassifierPredictor
class JclClassifierPredictor(MultiHeadClassifierPredictor):
    """Assemble a ``JclValidationResult`` from generic multi-head outputs."""

    def __init__(self) -> None:
        super().__init__()
        self._templates: dict[str, Any] = {}

    def setup(
        self,
        checkpoint_dir: Path,
        *,
        model_def: Optional[ModelDefinition] = None,
        device: Any = "auto",
        max_input_tokens: int = 1024,
        schema_path: Optional[Path] = None,
        contracts_path: Optional[Path] = None,
        prompt_spec_path: Optional[Path] = None,
        **kwargs: Any,
    ) -> None:
        super().setup(
            checkpoint_dir,
            model_def=model_def,
            device=device,
            max_input_tokens=max_input_tokens,
            schema_path=schema_path,
            contracts_path=contracts_path,
            prompt_spec_path=prompt_spec_path,
            **kwargs,
        )
        if contracts_path is None:
            contracts_path = resolve_eval_asset(
                "contracts",
                model_def=model_def,
                checkpoint_dir=Path(checkpoint_dir),
                filenames=("node_contracts.json",),
            )
        contracts = json.loads(Path(contracts_path).read_text(encoding="utf-8"))
        self._templates = contracts.get("error_message_templates", {})

    def predict(self, row: dict) -> str:
        heads = self.predict_heads(row)
        validity = heads.get("validity") or {}
        code_h = heads.get("error_code") or {}
        sev_h = heads.get("severity") or {}
        line_h = heads.get("line") or {}

        label = validity.get("label", "valid")
        is_valid = label == "valid" or (
            validity.get("index") == 1 and label not in ("invalid",)
        )
        # Prefer explicit invalid/valid labels.
        if label == "invalid":
            is_valid = False
        elif label == "valid":
            is_valid = True

        valid_conf = float(validity.get("confidence") or 0.0)
        code = str(code_h.get("label") or "other")
        severity_str = str(sev_h.get("label") or "error")
        line_no = line_h.get("line")

        errors_out: list[dict] = []
        if not is_valid:
            tpl = self._templates.get(code) or self._templates.get("other") or {
                "message": f"{code} (no template registered)",
                "suggestion": "",
            }
            errors_out.append(
                {
                    "line": int(line_no) if line_no else 1,
                    "column": 1,
                    "severity": severity_str if severity_str != "none" else "error",
                    "code": code if code != "none" else "other",
                    "message": tpl.get("message", ""),
                    "suggestion": tpl.get("suggestion") or None,
                }
            )
        pred_json = {
            "valid": bool(is_valid),
            "errors": errors_out,
            "confidence": valid_conf,
        }
        return json.dumps(pred_json, ensure_ascii=False)
