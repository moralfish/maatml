"""Shared generate → validate → accept/reject corpus builder."""
from __future__ import annotations

from typing import Any, Callable, Optional

GenerateFn = Callable[[], Optional[dict[str, Any]]]
ValidateFn = Callable[[dict[str, Any]], bool]


def build_gated_corpus(
    *,
    generate_fn: GenerateFn,
    validate_fn: ValidateFn,
    target_n: int,
    max_attempts: Optional[int] = None,
    on_reject: Optional[Callable[[dict[str, Any], Any], None]] = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Loop until ``target_n`` accepted rows or ``max_attempts`` exhausted.

    ``generate_fn`` returns a candidate row (or ``None`` to skip).
    ``validate_fn`` returns True to accept. Rejected rows (and optionally
    generate-None skips are not recorded) are collected for reporting.

    Returns ``(accepted, rejected)``.
    """
    if target_n < 0:
        raise ValueError(f"target_n must be >= 0; got {target_n}")
    attempts_cap = max_attempts if max_attempts is not None else max(target_n * 20, 1)

    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    attempts = 0

    while len(accepted) < target_n and attempts < attempts_cap:
        attempts += 1
        try:
            row = generate_fn()
        except Exception as exc:  # noqa: BLE001 — treat generate errors as rejects
            rejected.append({"error": str(exc), "_generate_failed": True})
            if on_reject is not None:
                on_reject({"error": str(exc)}, exc)
            continue
        if row is None:
            continue
        try:
            ok = bool(validate_fn(row))
        except Exception as exc:  # noqa: BLE001
            ok = False
            row = dict(row)
            row["_validate_error"] = str(exc)
            if on_reject is not None:
                on_reject(row, exc)
        if ok:
            accepted.append(row)
        else:
            rejected.append(row)
            if on_reject is not None:
                on_reject(row, None)

    return accepted, rejected
