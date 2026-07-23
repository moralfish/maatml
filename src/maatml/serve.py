"""HTTP inference server for any maatml model folder.

Uses stdlib ``ThreadingHTTPServer`` — no FastAPI/uvicorn — so the base install
stays light (important on JetPack). The predictor is loaded once at startup via
the same registry path the eval harness uses.

Endpoints:

* ``GET /health`` — liveness + identity
* ``GET /info`` — model summary + packaging hints
* ``POST /predict`` — dataset-shaped JSON row → prediction
  (``?validate=1`` runs the registered validator when configured)
"""
from __future__ import annotations

import json
import threading
import time
import traceback
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

from .config import ModelDefinition, get_dataset_cfg
from .device import resolve_device
from .registry import PREDICTORS, VALIDATORS, discover_plugins, load_model_plugins
from .runs import resolve_checkpoint
from .scaffold import normalize_architecture
from .validation.base import ValidationResult, strip_fences

# Reject request bodies larger than this by default (1 MiB). Predict rows are
# small JSON; anything larger is almost certainly abuse or a mistake.
DEFAULT_MAX_BODY_BYTES = 1_048_576


class RequestTooLarge(Exception):
    """Request body exceeded the configured size cap (maps to HTTP 413)."""


@dataclass
class ServeContext:
    """Shared state for the request handler (one per server)."""

    model_def: ModelDefinition
    checkpoint_dir: Path
    device: str
    predictor: Any
    validator: Any | None = None
    schema_path: Path | None = None
    contracts_path: Path | None = None
    prompt_spec_path: Path | None = None
    # CORS is opt-in: when None, no Access-Control-Allow-Origin header is sent,
    # so a browser on another origin cannot read responses. Set to "*" or a
    # specific origin to enable cross-origin access deliberately.
    cors_origin: str | None = None
    max_body_bytes: int = DEFAULT_MAX_BODY_BYTES
    started_at: float = field(default_factory=time.time)
    lock: threading.Lock = field(default_factory=threading.Lock)


def _resolve_eval_asset(
    key: str,
    model_def: ModelDefinition,
    checkpoint_dir: Path,
    filenames: tuple[str, ...],
) -> Path | None:
    cfg = get_dataset_cfg(model_def)
    rel = cfg.get(key)
    if isinstance(rel, str):
        path = model_def.resolve(rel)
        if path.is_file():
            return path
    for name in filenames:
        cand = checkpoint_dir / name
        if cand.is_file():
            return cand
    return None


def _resolve_predictor_name(model_def: ModelDefinition) -> str:
    ev = model_def.evaluation or {}
    predictor = ev.get("predictor")
    if isinstance(predictor, str) and predictor:
        return predictor
    arch = normalize_architecture(model_def.architecture)
    if PREDICTORS.get(model_def.architecture):
        return model_def.architecture
    if PREDICTORS.get(arch):
        return arch
    raise KeyError(
        f"No evaluation.predictor in model.yml and no predictor registered for "
        f"architecture={model_def.architecture!r}. Known: {PREDICTORS.names()}"
    )


def build_serve_context(
    model_def: ModelDefinition,
    *,
    checkpoint: str | Path | None = None,
    device: str = "auto",
    cors_origin: str | None = None,
    max_body_bytes: int = DEFAULT_MAX_BODY_BYTES,
) -> ServeContext:
    """Load plugins, resolve checkpoint, instantiate and setup the predictor."""
    discover_plugins()
    if model_def.plugins:
        load_model_plugins(model_def.model_dir, model_def.plugins)

    checkpoint_dir = resolve_checkpoint(model_def, checkpoint)
    try:
        target_device = resolve_device(device)
    except ImportError:
        # torch absent: real serving needs it, but keep a plain device string so
        # torch-free predictors (and tests) can still build a serve context.
        target_device = "cpu" if device in (None, "auto") else device

    pred_name = _resolve_predictor_name(model_def)
    pred_obj = PREDICTORS.require(pred_name)
    if isinstance(pred_obj, type):
        pred_obj = pred_obj()

    schema_path = _resolve_eval_asset(
        "schema",
        model_def,
        checkpoint_dir,
        ("schema.json",),
    )
    contracts_path = _resolve_eval_asset(
        "contracts",
        model_def,
        checkpoint_dir,
        ("node_contracts.json",),
    )
    prompt_spec_path = _resolve_eval_asset(
        "prompt_spec",
        model_def,
        checkpoint_dir,
        ("prompt_spec.json",),
    )

    setup = getattr(pred_obj, "setup", None)
    if callable(setup):
        setup(
            checkpoint_dir,
            model_def=model_def,
            device=target_device,
            max_input_tokens=model_def.packaging.max_input_tokens,
            schema_path=schema_path,
            contracts_path=contracts_path,
            prompt_spec_path=prompt_spec_path,
        )

    validator = None
    ev = model_def.evaluation or {}
    val_name = ev.get("validator")
    if isinstance(val_name, str) and val_name:
        validator = VALIDATORS.require(val_name)

    return ServeContext(
        model_def=model_def,
        checkpoint_dir=checkpoint_dir,
        device=str(target_device),
        predictor=pred_obj,
        validator=validator,
        schema_path=schema_path,
        contracts_path=contracts_path,
        prompt_spec_path=prompt_spec_path,
        cors_origin=cors_origin,
        max_body_bytes=max_body_bytes,
    )


def _try_parse_json(raw: str) -> Any | None:
    text = strip_fences(raw)
    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return None


def _validation_payload(result: ValidationResult) -> dict[str, Any]:
    return {
        "valid": bool(result.ok),
        "errors": [
            {
                "layer": e.layer,
                "code": e.code,
                "message": e.message,
                "location": e.location,
            }
            for e in result.errors
        ],
    }


def _run_predict(
    ctx: ServeContext,
    row: dict[str, Any],
    *,
    do_validate: bool,
) -> dict[str, Any]:
    predict = ctx.predictor.predict if hasattr(ctx.predictor, "predict") else ctx.predictor
    t0 = time.perf_counter()
    with ctx.lock:
        raw = predict(row)
    latency_ms = (time.perf_counter() - t0) * 1000.0

    payload: dict[str, Any] = {
        "output": _try_parse_json(raw) if isinstance(raw, str) else raw,
        "raw": raw if isinstance(raw, str) else json.dumps(raw),
        "latency_ms": round(latency_ms, 3),
    }

    if do_validate:
        if ctx.validator is None:
            payload["valid"] = None
            payload["errors"] = [
                {
                    "layer": 0,
                    "code": "no_validator",
                    "message": "No evaluation.validator configured in model.yml",
                    "location": None,
                }
            ]
        else:
            cfg = get_dataset_cfg(ctx.model_def)
            request_field = cfg.get("request_field") or cfg.get("raw_field") or "request"
            user_prompt = row.get(request_field)
            if not isinstance(user_prompt, str):
                user_prompt = None
            kwargs: dict[str, Any] = {"user_prompt": user_prompt}
            if ctx.schema_path is not None:
                kwargs["schema_path"] = ctx.schema_path
            if ctx.contracts_path is not None:
                kwargs["contracts_path"] = ctx.contracts_path
            try:
                result = ctx.validator(raw if isinstance(raw, str) else json.dumps(raw), **kwargs)
            except TypeError:
                # Some validators are methods / callables with different signatures.
                result = ctx.validator(raw if isinstance(raw, str) else json.dumps(raw))
            payload.update(_validation_payload(result))
    return payload


def _make_handler(ctx: ServeContext) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        server_version = "maatml-serve/0.4"

        def log_message(self, fmt: str, *args: Any) -> None:  # noqa: A003
            # Quiet by default; rich console prints startup banner.
            return

        def _send_json(self, status: int, body: dict[str, Any]) -> None:
            data = json.dumps(body, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            if ctx.cors_origin:
                self.send_header("Access-Control-Allow-Origin", ctx.cors_origin)
            self.end_headers()
            self.wfile.write(data)

        def _read_json_body(self) -> dict[str, Any]:
            length = int(self.headers.get("Content-Length") or 0)
            if length <= 0:
                raise ValueError("POST body required (JSON object)")
            if length > ctx.max_body_bytes:
                raise RequestTooLarge(
                    f"request body {length} bytes exceeds cap "
                    f"{ctx.max_body_bytes} bytes"
                )
            raw = self.rfile.read(length)
            try:
                payload = json.loads(raw.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise ValueError(f"Invalid JSON body: {exc}") from exc
            if not isinstance(payload, dict):
                raise ValueError("JSON body must be an object")
            return payload

        def do_OPTIONS(self) -> None:  # noqa: N802
            self.send_response(204)
            if ctx.cors_origin:
                self.send_header("Access-Control-Allow-Origin", ctx.cors_origin)
                self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
                self.send_header("Access-Control-Allow-Headers", "Content-Type")
            self.end_headers()

        def do_GET(self) -> None:  # noqa: N802
            path = urlparse(self.path).path.rstrip("/") or "/"
            md = ctx.model_def
            if path == "/health":
                self._send_json(
                    200,
                    {
                        "status": "ok",
                        "identity": md.identity,
                        "architecture": md.architecture,
                        "checkpoint": str(ctx.checkpoint_dir),
                        "device": ctx.device,
                        "uptime_s": round(time.time() - ctx.started_at, 1),
                    },
                )
                return
            if path == "/info":
                pkg = md.packaging
                self._send_json(
                    200,
                    {
                        "name": md.name,
                        "model_id": md.model_id,
                        "version": md.version,
                        "identity": md.identity,
                        "architecture": md.architecture,
                        "task": md.task,
                        "base_model": md.base_model,
                        "description": md.description,
                        "checkpoint": str(ctx.checkpoint_dir),
                        "device": ctx.device,
                        "packaging": {
                            "max_input_tokens": pkg.max_input_tokens,
                            "expected_latency_ms": pkg.expected_latency_ms,
                            "weights_dtype": pkg.weights_dtype,
                            "confidence_thresholds": pkg.confidence_thresholds,
                        },
                        "sidecars": {
                            "schema": str(ctx.schema_path) if ctx.schema_path else None,
                            "contracts": (
                                str(ctx.contracts_path) if ctx.contracts_path else None
                            ),
                            "prompt_spec": (
                                str(ctx.prompt_spec_path) if ctx.prompt_spec_path else None
                            ),
                        },
                        "predictor": _resolve_predictor_name(md),
                        "validator": (md.evaluation or {}).get("validator"),
                    },
                )
                return
            self._send_json(404, {"error": f"Unknown path {path!r}"})

        def do_POST(self) -> None:  # noqa: N802
            parsed = urlparse(self.path)
            path = parsed.path.rstrip("/") or "/"
            if path != "/predict":
                self._send_json(404, {"error": f"Unknown path {path!r}"})
                return
            qs = parse_qs(parsed.query)
            do_validate = any(
                v.lower() in ("1", "true", "yes") for v in qs.get("validate", [])
            )
            try:
                row = self._read_json_body()
                result = _run_predict(ctx, row, do_validate=do_validate)
                self._send_json(200, result)
            except RequestTooLarge as exc:
                self._send_json(413, {"error": str(exc)})
            except ValueError as exc:
                self._send_json(400, {"error": str(exc)})
            except Exception as exc:  # noqa: BLE001 — surface predict errors to client
                self._send_json(
                    500,
                    {
                        "error": str(exc),
                        "traceback": traceback.format_exc(),
                    },
                )

    return Handler


def serve_model(
    model_def: ModelDefinition,
    *,
    checkpoint: str | Path | None = None,
    host: str = "127.0.0.1",
    port: int = 8080,
    device: str = "auto",
    cors_origin: str | None = None,
    max_body_bytes: int = DEFAULT_MAX_BODY_BYTES,
    context: ServeContext | None = None,
) -> ThreadingHTTPServer:
    """Build context (unless provided) and return a ready ``ThreadingHTTPServer``.

    Caller is responsible for ``serve_forever()`` / shutdown. Useful for tests
    that want an ephemeral port without blocking.
    """
    ctx = context or build_serve_context(
        model_def,
        checkpoint=checkpoint,
        device=device,
        cors_origin=cors_origin,
        max_body_bytes=max_body_bytes,
    )
    handler = _make_handler(ctx)
    server = ThreadingHTTPServer((host, port), handler)
    # Attach context for callers / tests.
    server.maatml_context = ctx  # type: ignore[attr-defined]
    return server


def run_server(
    model_def: ModelDefinition,
    *,
    checkpoint: str | Path | None = None,
    host: str = "127.0.0.1",
    port: int = 8080,
    device: str = "auto",
    cors_origin: str | None = None,
    max_body_bytes: int = DEFAULT_MAX_BODY_BYTES,
) -> None:
    """Block serving until KeyboardInterrupt."""
    from rich.console import Console

    console = Console()
    ctx = build_serve_context(
        model_def,
        checkpoint=checkpoint,
        device=device,
        cors_origin=cors_origin,
        max_body_bytes=max_body_bytes,
    )
    server = serve_model(
        model_def,
        checkpoint=checkpoint,
        host=host,
        port=port,
        device=device,
        context=ctx,
    )
    console.print(
        f"[green]serving[/] {model_def.identity} ({model_def.architecture}) "
        f"ckpt={ctx.checkpoint_dir} device={ctx.device}"
    )
    console.print(f"  GET  http://{host}:{port}/health")
    console.print(f"  GET  http://{host}:{port}/info")
    console.print(f"  POST http://{host}:{port}/predict")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        console.print("\n[yellow]shutting down[/]")
    finally:
        server.shutdown()
        server.server_close()
