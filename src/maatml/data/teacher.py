"""OpenAI-compatible teacher client for validator-gated datagen.

Configured via ``MAATML_TEACHER_BASE_URL`` and ``MAATML_TEACHER_API_KEY``.
Requires the optional ``[teacher]`` extra (``httpx``).
"""

from __future__ import annotations

import json
import os
from typing import Any, Optional
from urllib.parse import urlparse

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
        resolved = base_url or os.environ.get("MAATML_TEACHER_BASE_URL")
        if not resolved or not resolved.strip():
            # No implicit third-party default: datagen and distill send the
            # prompt pool to whatever this points at, and for the shipped
            # domains (JCL, spool output, support tickets) the prompt pool is
            # the sensitive asset. The destination is always a stated choice.
            raise ValueError(
                "teacher base URL is not set. Export MAATML_TEACHER_BASE_URL "
                "(for example http://127.0.0.1:8000/v1 for a local server, or "
                "https://api.openai.com/v1) or pass base_url explicitly. "
                "maatml does not default to a third-party endpoint because "
                "your prompts are sent to it."
            )
        resolved = resolved.strip().rstrip("/")
        scheme = urlparse(resolved).scheme
        if scheme not in ("http", "https"):
            raise ValueError(f"teacher base URL must be http or https, got {resolved!r}")
        self.base_url = resolved
        self.api_key = api_key or os.environ.get("MAATML_TEACHER_API_KEY") or ""
        self.model = model
        self.timeout = timeout

    def chat_completions(
        self,
        messages: list[dict[str, str]],
        *,
        model: Optional[str] = None,
        temperature: Optional[float] = 0.7,
        max_tokens: int = 1024,
        **kwargs: Any,
    ) -> str:
        """POST ``/chat/completions`` and return the assistant message content.

        ``temperature=None`` omits the field entirely: some endpoints reject
        the parameter itself, not just particular values.
        """
        httpx = _require_httpx()
        url = f"{self.base_url}/chat/completions"
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        payload: dict[str, Any] = {
            "model": model or self.model,
            "messages": messages,
            "max_tokens": max_tokens,
        }
        if temperature is not None:
            payload["temperature"] = temperature
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
        request_params: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        """Ask the teacher for a JSON object row; parse and return it.

        ``request_params`` is merged into the request payload. Reasoning
        teachers need it: the default 1024-token budget is spent on hidden
        reasoning before any content arrives, and switches like
        ``chat_template_kwargs`` have no other way in.
        """
        content = self.chat_completions(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            model=model,
            temperature=0.8,
            **(request_params or {}),
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
