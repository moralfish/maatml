"""Built-in server backends for ``maatml serve --server …``.

Servers are registered plugins. Core dispatches; each backend owns its process
model (HTTP predictor, DeepStream pipeline, vLLM, llama.cpp, …). Lifecycle
hooks below are the common envelope every long-lived backend should honour.
"""

from __future__ import annotations

import signal
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional, Protocol

from .registry import SERVERS, register_server


class ServerHandle(Protocol):
    """Lifecycle envelope for a long-lived model service."""

    def verify(self) -> None:
        """Refuse to start when the artifact / manifest is wrong."""

    def warmup(self) -> None:
        """Load engines, run a readiness probe, mark ready."""

    def capabilities(self) -> dict[str, Any]:
        """Modality, task, runtime, identity, resource budget, protocols."""

    def serve_forever(self) -> None:
        """Block until drained / interrupted."""

    def close(self) -> None:
        """Release resources after drain."""


@dataclass
class LifecycleServer:
    """Adapter that runs verify → warmup → serve with SIGINT/SIGTERM drain."""

    name: str
    verify_fn: Callable[[], None]
    warmup_fn: Callable[[], None]
    serve_fn: Callable[[], None]
    close_fn: Callable[[], None]
    capabilities_fn: Callable[[], dict[str, Any]] = dict
    _draining: bool = False
    _closed: bool = field(default=False, init=False)

    def verify(self) -> None:
        self.verify_fn()

    def warmup(self) -> None:
        self.warmup_fn()

    def capabilities(self) -> dict[str, Any]:
        return dict(self.capabilities_fn() or {})

    def serve_forever(self) -> None:
        previous_int = signal.getsignal(signal.SIGINT)
        previous_term = signal.getsignal(signal.SIGTERM)

        def _drain(signum: int, _frame: Any) -> None:
            self._draining = True
            self.close()
            if signum == signal.SIGINT:
                raise KeyboardInterrupt

        signal.signal(signal.SIGINT, _drain)
        signal.signal(signal.SIGTERM, _drain)
        try:
            self.serve_fn()
        finally:
            signal.signal(signal.SIGINT, previous_int)
            signal.signal(signal.SIGTERM, previous_term)
            self.close()

    def close(self) -> None:
        if not self._closed:
            self._closed = True
            self.close_fn()

    def run(self) -> dict[str, Any]:
        """verify → warmup → serve_forever. Returns capability metadata."""
        self.verify()
        self.warmup()
        caps = self.capabilities()
        self.serve_forever()
        return caps


def dispatch_server(
    name: str,
    model_def: Any,
    *,
    checkpoint: str | Path | None = None,
    options: Optional[dict[str, str]] = None,
    **kwargs: Any,
) -> Any:
    """Look up ``name`` and invoke it with the shared serve kwargs."""
    server = SERVERS.require(name)
    return server(
        model_def,
        checkpoint=checkpoint,
        options=dict(options or {}),
        **kwargs,
    )


@register_server("http")
def http_server(
    model_def: Any,
    *,
    checkpoint: str | Path | None = None,
    options: Optional[dict[str, str]] = None,
    host: str = "127.0.0.1",
    port: int = 8080,
    device: str = "auto",
    cors_origin: str | None = None,
    max_body_bytes: int = 1_048_576,
    enforce: bool = False,
    debug: bool = False,
    auth_token: Optional[str] = None,
    max_retries: int = 0,
    capture_path: Optional[str | Path] = None,
    allow_unauthenticated: bool = False,
    **_ignored: Any,
) -> None:
    """Default development server: JSON over HTTP (see ``maatml.serve``)."""
    del options  # http takes typed CLI flags, not free-form KEY=VALUE
    from .serve import run_server

    run_server(
        model_def,
        checkpoint=checkpoint,
        host=host,
        port=port,
        device=device,
        cors_origin=cors_origin,
        max_body_bytes=max_body_bytes,
        enforce=enforce,
        debug=debug,
        auth_token=auth_token,
        max_retries=max_retries,
        capture_path=capture_path,
        allow_unauthenticated=allow_unauthenticated,
    )


@register_server("anthropic")
def anthropic_server(
    model_def: Any,
    *,
    checkpoint: str | Path | None = None,
    options: Optional[dict[str, str]] = None,
    host: str = "127.0.0.1",
    port: int = 8080,
    max_body_bytes: int = 8 * 1024 * 1024,
    auth_token: Optional[str] = None,
    **_ignored: Any,
) -> None:
    """Anthropic's Messages API in front of an OpenAI-compatible upstream.

    A translating proxy, not a runtime: llama.cpp holds the weights and renders
    the chat template, and this supplies only the protocol.

        llama-server --jinja -m model.gguf --port 8081 &
        maatml serve <model-dir> --server anthropic \\
            --server-option upstream=http://127.0.0.1:8081

    Options: ``upstream`` (default http://127.0.0.1:8081), ``model`` (the name
    echoed back, default the folder's model_id), ``timeout`` in seconds, and
    ``tool_style`` — ``native`` to declare tools upstream, ``inline`` to carry
    them in message text.
    """
    del checkpoint  # the upstream holds the weights; nothing is loaded here
    from .wire.anthropic import DEFAULT_UPSTREAM, build

    opts = dict(options or {})
    named = getattr(model_def, "model_id", None) or getattr(model_def, "name", None)
    server = build(
        upstream=opts.get("upstream", DEFAULT_UPSTREAM),
        model=opts.get("model") or named or "maatml-local",
        host=host,
        port=port,
        max_body_bytes=max_body_bytes,
        timeout=float(opts.get("timeout", 600)),
        auth_token=auth_token,
        tool_style=opts.get("tool_style", "native"),
    )
    print(f"maatml anthropic wire on http://{host}:{port}/v1/messages "
          f"-> {opts.get('upstream', DEFAULT_UPSTREAM)} "
          f"(tools: {opts.get('tool_style', 'native')})", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
