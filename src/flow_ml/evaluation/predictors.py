"""Registered evaluation predictors (model load + per-row generation).

Each predictor is a small class with ``setup(...)`` then ``predict(row) -> str``.
Asset resolution uses ``flow_ml.evaluation.harness.resolve_eval_asset`` —
never a hardcoded repo-root fallback.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

from ..config import ModelDefinition, get_dataset_cfg
from ..device import resolve_device
from ..registry import register_predictor
from .harness import resolve_eval_asset


class MultiHeadClassifierPredictor:
    """ModernBERT + 4 classification heads → JclValidationResult JSON text."""

    def __init__(self) -> None:
        self._encoder: Any = None
        self._tokenizer: Any = None
        self._heads: dict[str, dict[str, Any]] = {}
        self._templates: dict[str, Any] = {}
        self._device: Any = None
        self._max_input_tokens = 1024
        self._error_codes: list[str] = []
        self._severities: list[str] = []

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
        **_kwargs: Any,
    ) -> None:
        del schema_path, prompt_spec_path
        import torch
        from safetensors.torch import load_file
        from transformers import AutoModel, PreTrainedTokenizerFast

        from ..tokenization import pre_tokenize_jcl
        from ..training.jcl_classifier import ERROR_CODES, SEVERITIES

        self._pre_tokenize = pre_tokenize_jcl
        self._torch = torch
        self._error_codes = list(ERROR_CODES)
        self._severities = list(SEVERITIES)
        self._max_input_tokens = max_input_tokens
        self._device = (
            device if hasattr(device, "type") else resolve_device(str(device))
        )

        checkpoint_dir = Path(checkpoint_dir)
        tokenizer_path = resolve_eval_asset(
            "tokenizer",
            model_def=model_def,
            checkpoint_dir=checkpoint_dir,
            filenames=("tokenizer.json",),
        )
        self._tokenizer = PreTrainedTokenizerFast(
            tokenizer_file=str(tokenizer_path),
            model_max_length=max_input_tokens,
            pad_token="<PAD>",
            unk_token="<UNK>",
            cls_token="<CLS>",
            sep_token="<SEP>",
            mask_token="<MASK>",
            additional_special_tokens=["<COL1>", "<CONT>"],
        )

        if contracts_path is None:
            contracts_path = resolve_eval_asset(
                "contracts",
                model_def=model_def,
                checkpoint_dir=checkpoint_dir,
                filenames=("node_contracts.json",),
            )
        contracts = json.loads(Path(contracts_path).read_text(encoding="utf-8"))
        self._templates = contracts.get("error_message_templates", {})

        encoder = AutoModel.from_pretrained(checkpoint_dir).to(self._device)
        encoder.eval()
        self._encoder = encoder

        head_state = load_file(checkpoint_dir / "classifier_heads.safetensors")
        self._heads = {}
        for name in ("validity", "error_code", "severity", "line"):
            self._heads[name] = {
                "weight": head_state[f"heads.{name}.weight"].to(self._device),
                "bias": head_state[f"heads.{name}.bias"].to(self._device),
            }

    def _head_forward(self, name: str, x):
        h = self._heads[name]
        return self._torch.nn.functional.linear(x, h["weight"], h["bias"])

    def _first_error_line(self, line_logits, encoding, pre: str) -> Optional[int]:
        cls_per_tok = line_logits.squeeze(0).argmax(dim=-1).tolist()
        tokens = encoding.tokens()
        offsets = encoding.offsets
        for i, label in enumerate(cls_per_tok):
            if label != 1:
                continue
            tok = tokens[i] if i < len(tokens) else ""
            if tok.startswith("<") and tok.endswith(">"):
                continue
            char_offset = offsets[i][0] if i < len(offsets) else 0
            return max(1, pre[: min(char_offset, len(pre))].count("<COL1>"))
        return None

    def predict(self, row: dict) -> str:
        assert self._encoder is not None and self._tokenizer is not None
        torch = self._torch
        pre = self._pre_tokenize(row["request"])
        encoding = self._tokenizer(
            pre,
            max_length=self._max_input_tokens,
            truncation=True,
            return_offsets_mapping=True,
            return_tensors=None,
        )
        input_ids = torch.tensor(
            [encoding["input_ids"]], dtype=torch.long, device=self._device
        )
        attention_mask = torch.tensor(
            [encoding["attention_mask"]], dtype=torch.long, device=self._device
        )

        class _Enc:
            def __init__(self, e):
                self._e = e

            def tokens(self):
                return self._e.tokens(0) if hasattr(self._e, "tokens") else []

            @property
            def offsets(self):
                return self._e["offset_mapping"]

        with torch.inference_mode():
            out = self._encoder(input_ids=input_ids, attention_mask=attention_mask)
            seq = out.last_hidden_state
            pooled = seq[:, 0, :]
            validity_logits = self._head_forward("validity", pooled)
            error_code_logits = self._head_forward("error_code", pooled)
            severity_logits = self._head_forward("severity", pooled)
            line_logits = self._head_forward("line", seq)

        validity_probs = torch.softmax(validity_logits.squeeze(0), dim=-1)
        valid_idx = int(validity_probs.argmax().item())
        valid_conf = float(validity_probs[valid_idx].item())
        code_idx = int(error_code_logits.squeeze(0).argmax().item())
        severity_idx = int(severity_logits.squeeze(0).argmax().item())

        is_valid = valid_idx == 1
        code = (
            self._error_codes[code_idx]
            if code_idx < len(self._error_codes)
            else "other"
        )
        severity_str = (
            self._severities[severity_idx]
            if severity_idx < len(self._severities)
            else "error"
        )

        line_no: Optional[int] = None
        if not is_valid:
            line_no = self._first_error_line(line_logits, _Enc(encoding), pre)

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
                    "code": code,
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


class Seq2SeqPredictor:
    """Encoder-decoder generate (T5 / flan-t5) with optional task prefix."""

    def __init__(self, task_prefix: str = "interpret spool: ") -> None:
        self._model: Any = None
        self._tokenizer: Any = None
        self._device: Any = None
        self._max_input_tokens = 1024
        self._task_prefix = task_prefix
        self._max_new_tokens = 512

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
        **_kwargs: Any,
    ) -> None:
        del schema_path, contracts_path, prompt_spec_path
        import torch
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        from ..training.spool_seq2seq import TASK_PREFIX

        self._torch = torch
        self._task_prefix = TASK_PREFIX
        self._max_input_tokens = max_input_tokens
        self._device = (
            device if hasattr(device, "type") else resolve_device(str(device))
        )
        if model_def is not None:
            gen = (model_def.training or {}).get("generation") or {}
            if "max_new_tokens" in gen:
                self._max_new_tokens = int(gen["max_new_tokens"])

        checkpoint_dir = Path(checkpoint_dir)
        self._tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir, use_fast=True)
        inference_dtype = (
            torch.float16 if self._device.type in ("mps", "cuda") else torch.float32
        )
        model = AutoModelForSeq2SeqLM.from_pretrained(
            checkpoint_dir, dtype=inference_dtype
        ).to(self._device)
        model.eval()
        self._model = model

    def predict(self, row: dict) -> str:
        assert self._model is not None and self._tokenizer is not None
        torch = self._torch
        source = self._task_prefix + row["request"]
        enc = self._tokenizer(
            source,
            max_length=self._max_input_tokens,
            truncation=True,
            return_tensors="pt",
        ).to(self._device)

        with torch.inference_mode():
            generated = self._model.generate(
                input_ids=enc["input_ids"],
                attention_mask=enc["attention_mask"],
                max_new_tokens=self._max_new_tokens,
                num_beams=1,
                do_sample=False,
            )

        gen_text = self._tokenizer.decode(
            generated[0], skip_special_tokens=True
        ).strip()
        # T5 SentencePiece maps `{`/`}` to <unk>; skip_special_tokens strips them.
        if gen_text and not gen_text.startswith("{"):
            gen_text = "{" + gen_text
        if gen_text and not gen_text.endswith("}"):
            gen_text = gen_text + "}"
        return gen_text


class CausalSFTPredictor:
    """Causal LM generate with chat template / prompt_spec."""

    def __init__(self) -> None:
        self._model: Any = None
        self._tokenizer: Any = None
        self._device: Any = None
        self._max_input_tokens = 2048
        self._max_new_tokens = 512
        self._prompt_spec: dict = {}
        self._user_placeholder = "<<USER_REQUEST>>"
        self._request_field = "request"

    def setup(
        self,
        checkpoint_dir: Path,
        *,
        model_def: Optional[ModelDefinition] = None,
        device: Any = "auto",
        max_input_tokens: int = 2048,
        schema_path: Optional[Path] = None,
        contracts_path: Optional[Path] = None,
        prompt_spec_path: Optional[Path] = None,
        **_kwargs: Any,
    ) -> None:
        del schema_path, contracts_path
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        from ..training.sft_base import render_inference_prompt
        from ..utils.io import read_json

        self._torch = torch
        self._render = render_inference_prompt
        self._max_input_tokens = max_input_tokens
        self._device = (
            device if hasattr(device, "type") else resolve_device(str(device))
        )

        checkpoint_dir = Path(checkpoint_dir)
        if prompt_spec_path is None:
            prompt_spec_path = resolve_eval_asset(
                "prompt_spec",
                model_def=model_def,
                checkpoint_dir=checkpoint_dir,
                filenames=("prompt_spec.json",),
            )
        self._prompt_spec = read_json(prompt_spec_path)

        if model_def is not None:
            cfg = get_dataset_cfg(model_def)
            self._request_field = (
                cfg.get("request_field") or cfg.get("raw_field") or "request"
            )
            self._user_placeholder = cfg.get("user_placeholder") or "<<USER_REQUEST>>"
            gen = (model_def.training or {}).get("generation") or {}
            if "max_new_tokens" in gen:
                self._max_new_tokens = int(gen["max_new_tokens"])

        self._tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir, use_fast=True)
        inference_dtype = (
            torch.float16 if self._device.type in ("mps", "cuda") else torch.float32
        )
        model = AutoModelForCausalLM.from_pretrained(
            checkpoint_dir, dtype=inference_dtype
        ).to(self._device)
        model.eval()
        self._model = model

    def predict(self, row: dict) -> str:
        assert self._model is not None and self._tokenizer is not None
        torch = self._torch
        request = row[self._request_field]
        input_ids = self._render(
            request,
            self._prompt_spec,
            self._tokenizer,
            user_placeholder=self._user_placeholder,
        )
        if self._max_input_tokens and len(input_ids) > self._max_input_tokens:
            input_ids = input_ids[-self._max_input_tokens :]
        tensor = torch.tensor([input_ids], dtype=torch.long, device=self._device)
        with torch.inference_mode():
            generated = self._model.generate(
                input_ids=tensor,
                max_new_tokens=self._max_new_tokens,
                do_sample=False,
                pad_token_id=self._tokenizer.pad_token_id
                or self._tokenizer.eos_token_id,
            )
        new_tokens = generated[0, tensor.shape[-1] :]
        return self._tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


register_predictor("multi_head_classifier")(MultiHeadClassifierPredictor)
register_predictor("classifier")(MultiHeadClassifierPredictor)
register_predictor("seq2seq")(Seq2SeqPredictor)
register_predictor("causal_sft")(CausalSFTPredictor)
