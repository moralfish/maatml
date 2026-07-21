"""Teacher client unit tests (no network)."""
from __future__ import annotations

from typing import Any

import pytest

from maatml.data.teacher import TeacherClient


class _FakeResponse:
    def __init__(self, payload: dict[str, Any], status: int = 200) -> None:
        self._payload = payload
        self.status_code = status

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")

    def json(self) -> dict[str, Any]:
        return self._payload


class _FakeClient:
    last_payload: dict[str, Any] | None = None

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        del args, kwargs

    def __enter__(self) -> "_FakeClient":
        return self

    def __exit__(self, *args: Any) -> None:
        return None

    def post(self, url: str, headers: dict, json: dict) -> _FakeResponse:
        assert url.endswith("/chat/completions")
        assert "Authorization" in headers
        _FakeClient.last_payload = json
        return _FakeResponse(
            {
                "choices": [
                    {"message": {"content": '{"request":"hi","target":{"ok":true}}'}}
                ]
            }
        )


def test_teacher_chat_completions(monkeypatch: pytest.MonkeyPatch) -> None:
    import maatml.data.teacher as teacher_mod

    class _Httpx:
        Client = _FakeClient

    monkeypatch.setattr(teacher_mod, "_require_httpx", lambda: _Httpx)
    client = TeacherClient(
        base_url="https://example.test/v1",
        api_key="sk-test",
        model="toy",
    )
    text = client.chat_completions([{"role": "user", "content": "ping"}])
    assert "request" in text
    row = client.propose_json_row("sys", "user")
    assert row["request"] == "hi"
    assert row["target"]["ok"] is True


def test_teacher_install_hint(monkeypatch: pytest.MonkeyPatch) -> None:
    import builtins

    import maatml.data.teacher as teacher_mod

    real_import = builtins.__import__

    def _fake_import(name, *args, **kwargs):
        if name == "httpx" or name.startswith("httpx."):
            raise ImportError("No module named 'httpx'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _fake_import)
    # Clear any cached import path by calling the real helper.
    client = TeacherClient(api_key="x")
    with pytest.raises(ImportError, match=r"maatml\[teacher\]"):
        client.chat_completions([{"role": "user", "content": "x"}])
    assert "httpx" in teacher_mod._INSTALL_HINT
