"""CLI config overrides and offline HPO / sweep helpers.

``--set training.learning_rate=1e-4`` and ``maatml sweep --param ...`` both
route through :func:`apply_overrides` / :func:`expand_param_grid` so nested
``model.yml`` dicts stay editable without Optuna.
"""
from __future__ import annotations

import itertools
import json
import re
from typing import Any, Optional

from .config import ModelDefinition

_OVERRIDE_RX = re.compile(r"^([^=]+)=(.*)$", re.DOTALL)

# Top-level ModelDefinition fields that are plain nested dicts.
_DICT_SECTIONS = frozenset({"data", "dataset", "training", "smoke", "evaluation", "extensions"})


def coerce_override_value(raw: str) -> Any:
    """Parse a CLI override value into bool/int/float/JSON/str."""
    text = raw.strip()
    if text == "":
        return ""
    lower = text.lower()
    if lower == "true":
        return True
    if lower == "false":
        return False
    if lower == "null" or lower == "none":
        return None
    # JSON literals (lists/dicts/quoted strings/numbers)
    if text[0] in "[{\"'" or text[0].isdigit() or text[0] in "+-":
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
    # Bare numbers (including scientific) when JSON didn't accept (e.g. 1e-4
    # is valid JSON; but keep a float fallback for odd forms).
    try:
        if re.fullmatch(r"[+-]?\d+", text):
            return int(text)
        if re.fullmatch(r"[+-]?(\d+\.\d*|\.\d+|\d+)([eE][+-]?\d+)?", text):
            return float(text)
    except ValueError:
        pass
    return text


def parse_override(spec: str) -> tuple[str, Any]:
    """Parse ``KEY=VALUE`` into ``(dotted_key, coerced_value)``."""
    m = _OVERRIDE_RX.match(spec.strip())
    if not m:
        raise ValueError(
            f"Invalid override {spec!r}; expected KEY=VALUE "
            "(e.g. training.learning_rate=1e-4)"
        )
    key = m.group(1).strip()
    if not key:
        raise ValueError(f"Invalid override {spec!r}: empty key")
    return key, coerce_override_value(m.group(2))


def _set_dotted(root: dict[str, Any], dotted: str, value: Any) -> None:
    parts = [p for p in dotted.split(".") if p]
    if not parts:
        raise ValueError("empty dotted path")
    cur: Any = root
    for part in parts[:-1]:
        if not isinstance(cur, dict):
            raise ValueError(f"Cannot set {dotted!r}: {part!r} is not a mapping")
        nxt = cur.get(part)
        if nxt is None or not isinstance(nxt, dict):
            nxt = {}
            cur[part] = nxt
        cur = nxt
    if not isinstance(cur, dict):
        raise ValueError(f"Cannot set {dotted!r}: parent is not a mapping")
    cur[parts[-1]] = value


def apply_overrides(
    model_def: ModelDefinition,
    overrides: list[str] | tuple[str, ...] | None,
) -> ModelDefinition:
    """Apply ``KEY=VALUE`` overrides onto a loaded :class:`ModelDefinition`.

    Nested paths under ``training`` / ``dataset`` / ``smoke`` / ``evaluation``
    (and ``data`` / ``extensions``) mutate those dicts in place. Other
    top-level scalar fields (``architecture``, ``base_model``, …) are set via
    ``setattr`` when present on the model.
    """
    if not overrides:
        return model_def
    for spec in overrides:
        key, value = parse_override(spec)
        head, _, rest = key.partition(".")
        if head in _DICT_SECTIONS:
            section = getattr(model_def, head)
            if not isinstance(section, dict):
                raise ValueError(f"{head} is not a dict section on ModelDefinition")
            if rest:
                _set_dotted(section, rest, value)
            else:
                if not isinstance(value, dict):
                    raise ValueError(
                        f"Replacing entire {head!r} requires a JSON object value"
                    )
                setattr(model_def, head, value)
        elif rest:
            # e.g. packaging.max_input_tokens — only packaging is a nested model
            if head == "packaging":
                pkg = model_def.packaging
                if not hasattr(pkg, rest.split(".")[0]):
                    raise ValueError(f"Unknown packaging field path {key!r}")
                # Only one level deep is common; set leaf on packaging model.
                leaf = rest.split(".")
                if len(leaf) != 1:
                    raise ValueError(
                        f"Nested packaging overrides beyond one level unsupported: {key!r}"
                    )
                object.__setattr__(pkg, leaf[0], value)
            else:
                raise ValueError(
                    f"Unknown override path {key!r}; "
                    "use training./dataset./smoke./evaluation. prefixes"
                )
        else:
            if not hasattr(model_def, head) or head in ("model_dir",):
                raise ValueError(f"Unknown ModelDefinition field {key!r}")
            setattr(model_def, head, value)
    return model_def


def parse_param_values(spec: str) -> tuple[str, list[Any]]:
    """Parse ``KEY=v1,v2,v3`` into ``(key, [coerced values])``.

    Commas inside JSON brackets are preserved (``training.lora.target_modules=["q","v"]``
    is a single value — prefer separate ``--param`` entries for grids).
    """
    key, raw = parse_override(spec)
    # Split on commas not inside [], {}, or quotes.
    parts: list[str] = []
    buf: list[str] = []
    depth = 0
    in_str: Optional[str] = None
    for ch in raw:
        if in_str:
            buf.append(ch)
            if ch == in_str:
                in_str = None
            continue
        if ch in "'\"":
            in_str = ch
            buf.append(ch)
            continue
        if ch in "[{":
            depth += 1
            buf.append(ch)
            continue
        if ch in "]}":
            depth = max(0, depth - 1)
            buf.append(ch)
            continue
        if ch == "," and depth == 0:
            parts.append("".join(buf).strip())
            buf = []
            continue
        buf.append(ch)
    if buf or not parts:
        parts.append("".join(buf).strip())
    values = [coerce_override_value(p) for p in parts if p != "" or len(parts) == 1]
    if not values:
        raise ValueError(f"No values in param spec {spec!r}")
    return key, values


def expand_param_grid(
    params: list[str] | tuple[str, ...] | None,
    *,
    max_trials: Optional[int] = None,
) -> list[dict[str, Any]]:
    """Cartesian product of ``--param KEY=a,b`` specs → list of override dicts.

    Each dict maps dotted keys to a single value (ready for
    ``apply_overrides`` via ``[f"{k}={v}" ...]`` or direct set).
    """
    if not params:
        return [{}]
    keys: list[str] = []
    value_lists: list[list[Any]] = []
    for spec in params:
        key, values = parse_param_values(spec)
        keys.append(key)
        value_lists.append(values)
    combos = list(itertools.product(*value_lists))
    if max_trials is not None and max_trials >= 0:
        combos = combos[:max_trials]
    return [dict(zip(keys, combo)) for combo in combos]


def overrides_from_mapping(mapping: dict[str, Any]) -> list[str]:
    """Convert a trial dict to ``KEY=VALUE`` strings for :func:`apply_overrides`."""
    out: list[str] = []
    for key, value in mapping.items():
        if isinstance(value, str):
            out.append(f"{key}={value}")
        else:
            out.append(f"{key}={json.dumps(value)}")
    return out


def pick_metric(
    metrics: dict[str, Any] | None,
    preferred: Optional[str] = None,
) -> tuple[Optional[str], Optional[float]]:
    """Select a numeric metric for sweep ranking."""
    if not metrics:
        return None, None
    if preferred and preferred in metrics:
        try:
            return preferred, float(metrics[preferred])
        except (TypeError, ValueError):
            pass
    for key, val in metrics.items():
        if isinstance(val, (int, float)) and not isinstance(val, bool):
            return key, float(val)
    return None, None
