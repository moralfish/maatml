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
            # The heads are independent, so they can disagree: the validity head
            # says invalid while the code head says "none". Normalise the code
            # BEFORE the template lookup. Looking it up under the raw "none" (a
            # deliberately empty template) while emitting "other" produced rows
            # with an empty message and a null suggestion, and the schema
            # requires strings for both. That capped schema conformance at 0.74
            # with the model never once emitting unparseable JSON.
            code_out = code if code != "none" else "other"
            tpl = self._templates.get(code_out) or {}
            message = tpl.get("message") or (
                f"Validator flagged this deck under {code_out}; no message "
                "template is registered for that code."
            )
            suggestion = tpl.get("suggestion") or (
                "Review the deck against the IBM z/OS JCL Reference."
            )
            errors_out.append(
                {
                    "line": int(line_no) if line_no else 1,
                    "column": 1,
                    "severity": severity_str if severity_str != "none" else "error",
                    "code": code_out,
                    # Both fields are required strings in the schema, so they
                    # always carry text: never "" and never null.
                    "message": message,
                    "suggestion": suggestion,
                }
            )
        pred_json = {
            "valid": bool(is_valid),
            "errors": errors_out,
            "confidence": valid_conf,
        }
        return json.dumps(pred_json, ensure_ascii=False)
