"""OpenAI-compatible teacher client for validator-gated datagen.

Configured via ``MAATML_TEACHER_BASE_URL`` and ``MAATML_TEACHER_API_KEY``.
Requires the optional ``[teacher]`` extra (``httpx``).
"""
from __future__ import annotations

import json
import os
from typing import Any, Optional

_INSTALL_HINT = "Teacher client requires httpx; install with `pip install maatml[teacher]`"


def _require_httpx():
    try:
        import httpx
    except ImportError as exc:
        raise ImportError(_INSTALL_HINT) from exc
    return httpx


class TeacherClient:
    """Minimal chat-completions client (OpenAI-compatible HTTP API)."""

    def __init__(
        self,
        *,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        model: str = "gpt-4o-mini",
        timeout: float = 60.0,
    ) -> None:
        self.base_url = (
            base_url
            or os.environ.get("MAATML_TEACHER_BASE_URL")
            or "https://api.openai.com/v1"
        ).rstrip("/")
        self.api_key = api_key or os.environ.get("MAATML_TEACHER_API_KEY") or ""
        self.model = model
        self.timeout = timeout

    def chat_completions(
        self,
        messages: list[dict[str, str]],
        *,
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        **kwargs: Any,
    ) -> str:
        """POST ``/chat/completions`` and return the assistant message content."""
        httpx = _require_httpx()
        url = f"{self.base_url}/chat/completions"
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        payload: dict[str, Any] = {
            "model": model or self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        payload.update(kwargs)
        with httpx.Client(timeout=self.timeout) as client:
            resp = client.post(url, headers=headers, json=payload)
            resp.raise_for_status()
            data = resp.json()
        try:
            return str(data["choices"][0]["message"]["content"])
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError(
                f"Unexpected teacher response shape: {json.dumps(data)[:500]}"
            ) from exc

    def propose_json_row(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        model: Optional[str] = None,
    ) -> dict[str, Any]:
        """Ask the teacher for a JSON object row; parse and return it."""
        content = self.chat_completions(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            model=model,
            temperature=0.8,
        )
        text = content.strip()
        if text.startswith("```"):
            # Strip optional ```json fences.
            lines = text.splitlines()
            if lines and lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            text = "\n".join(lines)
        return json.loads(text)
