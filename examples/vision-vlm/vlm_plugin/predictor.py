"""Dual-backend predictor: transformers locally, or OpenAI-compatible vLLM endpoint."""
from __future__ import annotations

import base64
import io
import json
import os
import re
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Optional

_DEFAULT_USER_PROMPT = (
    "Describe this synthetic scene in one short factual sentence covering "
    "the background style, any colored shapes, and the stick figure's pose."
)


def _resolve_image_bytes(value: Any, *, model_dir: Optional[Path] = None) -> bytes:
    if isinstance(value, (bytes, bytearray)):
        return bytes(value)
    if not isinstance(value, str):
        raise TypeError(f"image must be str/bytes; got {type(value)}")
    if value.startswith("data:"):
        m = re.match(r"data:image/[^;]+;base64,(.+)$", value, re.DOTALL)
        if not m:
            raise ValueError("Invalid data-URI image")
        return base64.b64decode(m.group(1))
    if "/" not in value and "\\" not in value and len(value) > 200:
        try:
            return base64.b64decode(value, validate=False)
        except Exception:  # noqa: BLE001
            pass
    path = Path(value)
    if not path.is_file() and model_dir is not None:
        path = Path(model_dir) / value
    if not path.is_file():
        raise FileNotFoundError(f"Image not found: {value}")
    return path.read_bytes()


def _to_pil(data: bytes):
    from PIL import Image

    return Image.open(io.BytesIO(data)).convert("RGB")


def _normalize_description(text: str) -> str:
    """Wrap free text into ``{\"description\": ...}`` JSON."""
    text = text.strip()
    try:
        from maatml.validation.base import strip_fences

        text = strip_fences(text)
    except Exception:  # noqa: BLE001
        pass
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict) and "description" in parsed:
            return json.dumps(
                {"description": str(parsed["description"]).strip()},
                ensure_ascii=False,
            )
    except json.JSONDecodeError:
        pass
    # Strip accidental quotes / role prefixes.
    text = re.sub(r"^(Assistant|assistant)\s*:\s*", "", text).strip()
    return json.dumps({"description": text}, ensure_ascii=False)


class VisionVlmPredictor:
    """Predict a short description for a dataset-shaped row with an ``image`` field."""

    def __init__(self) -> None:
        self.model = None
        self.processor = None
        self.device = "cpu"
        self.model_dir: Path | None = None
        self.checkpoint_dir: Path | None = None
        self.backend: str = "none"
        self.vllm_endpoint: str | None = None
        self.max_new_tokens = 64
        self.user_prompt = _DEFAULT_USER_PROMPT
        self.model_name = "vision-vlm"

    def setup(
        self,
        checkpoint_dir: Path,
        *,
        model_def: Any = None,
        device: Any = "cpu",
        max_input_tokens: Optional[int] = None,
        schema_path: Optional[Path] = None,
        contracts_path: Optional[Path] = None,
        prompt_spec_path: Optional[Path] = None,
    ) -> None:
        del max_input_tokens, schema_path, contracts_path
        self.checkpoint_dir = Path(checkpoint_dir)
        self.model_dir = Path(model_def.model_dir) if model_def is not None else None
        self.device = str(device)
        if model_def is not None:
            self.model_name = model_def.name
            gen = (model_def.training or {}).get("generation") or {}
            if "max_new_tokens" in gen:
                self.max_new_tokens = int(gen["max_new_tokens"])
            if prompt_spec_path is None:
                from maatml.config import get_dataset_cfg

                cfg = get_dataset_cfg(model_def)
                if isinstance(cfg.get("prompt_spec"), str):
                    prompt_spec_path = model_def.resolve(cfg["prompt_spec"])
            up = (model_def.training or {}).get("user_prompt")
            if isinstance(up, str) and up.strip():
                self.user_prompt = up.strip()

        if prompt_spec_path is not None and Path(prompt_spec_path).is_file():
            try:
                spec = json.loads(Path(prompt_spec_path).read_text(encoding="utf-8"))
                ut = spec.get("user_template")
                if isinstance(ut, str) and ut.strip() and "<<USER_REQUEST>>" not in ut:
                    self.user_prompt = ut.strip()
            except Exception:  # noqa: BLE001
                pass

        endpoint = os.environ.get("MAATML_VLLM_ENDPOINT", "").strip()
        if endpoint:
            self.vllm_endpoint = endpoint.rstrip("/")
            self.backend = "vllm"
            return

        self._setup_transformers()

    def _setup_transformers(self) -> None:
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor

        assert self.checkpoint_dir is not None
        self.processor = AutoProcessor.from_pretrained(self.checkpoint_dir)
        dtype = torch.float32
        if self.device.startswith("cuda") or self.device == "mps":
            dtype = torch.bfloat16
        self.model = AutoModelForImageTextToText.from_pretrained(
            self.checkpoint_dir, dtype=dtype
        )
        self.model.to(self.device)
        self.model.eval()
        self.backend = "transformers"

    def predict(self, row: dict[str, Any]) -> str:
        image_val = row.get("image")
        if image_val is None:
            raise KeyError("row missing 'image' field")
        data = _resolve_image_bytes(image_val, model_dir=self.model_dir)
        if self.backend == "vllm":
            text = self._infer_vllm(data)
        elif self.backend == "transformers":
            text = self._infer_transformers(data)
        else:
            raise RuntimeError(f"Unknown backend {self.backend!r}; call setup() first")
        return _normalize_description(text)

    def _infer_transformers(self, data: bytes) -> str:
        import torch

        assert self.model is not None and self.processor is not None
        image = _to_pil(data)
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": self.user_prompt},
                ],
            }
        ]
        prompt = self.processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )
        inputs = self.processor(
            text=prompt, images=[image], return_tensors="pt"
        ).to(self.device)
        with torch.inference_mode():
            generated = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
            )
        # Decode only the new tokens when possible.
        in_len = inputs["input_ids"].shape[-1]
        new_tokens = generated[0, in_len:]
        return self.processor.tokenizer.decode(
            new_tokens, skip_special_tokens=True
        ).strip()

    def _infer_vllm(self, data: bytes) -> str:
        assert self.vllm_endpoint is not None
        b64 = base64.b64encode(data).decode("ascii")
        data_url = f"data:image/png;base64,{b64}"
        payload = {
            "model": self.model_name,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": self.user_prompt},
                        {"type": "image_url", "image_url": {"url": data_url}},
                    ],
                }
            ],
            "max_tokens": self.max_new_tokens,
            "temperature": 0,
        }
        url = f"{self.vllm_endpoint}/v1/chat/completions"
        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                body = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"vLLM request failed ({exc.code}): {detail}") from exc
        try:
            return str(body["choices"][0]["message"]["content"])
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError(f"Unexpected vLLM response: {body!r}") from exc
