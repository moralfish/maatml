"""Generic regex-rule sanitization engine.

Task-specific rule sets (JCL / Spool / …) live in example plugins and
register themselves via ``@register_sanitizer``. Core only provides the
rule loader + apply helpers.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Iterable

from ..utils.io import read_yaml


@dataclass(frozen=True)
class SanitizationRule:
    name: str
    pattern: re.Pattern[str]
    replacement: str
    applies_to: frozenset[str]
    length_preserving: bool


def _compile_rule(raw: dict) -> SanitizationRule:
    return SanitizationRule(
        name=raw["name"],
        pattern=re.compile(raw["pattern"]),
        replacement=raw["replacement"],
        applies_to=frozenset(raw.get("applies_to", [])),
        length_preserving=bool(raw.get("length_preserving", False)),
    )


def load_rules(path: str | Path) -> list[SanitizationRule]:
    """Load sanitization rules from a YAML file with a top-level ``rules:`` list."""
    raw = read_yaml(path)
    return [_compile_rule(r) for r in raw["rules"]]


def _expand(rule: SanitizationRule, m: re.Match[str]) -> str:
    return m.expand(rule.replacement)


def _apply_one(text: str, rule: SanitizationRule) -> str:
    if not rule.length_preserving:
        return rule.pattern.sub(lambda m: _expand(rule, m), text)

    def _pad(m: re.Match[str]) -> str:
        repl = _expand(rule, m)
        original_len = m.end() - m.start()
        if len(repl) > original_len:
            return repl[:original_len]
        return repl + (" " * (original_len - len(repl)))

    return rule.pattern.sub(_pad, text)


def apply_rules(text: str, rules: Iterable[SanitizationRule]) -> str:
    out = text
    for rule in rules:
        out = _apply_one(out, rule)
    return out


def make_tag_sanitizer(
    rules_path: str | Path,
    *,
    tag: str,
    length_preserving_only: bool = False,
):
    """Build a ``(text) -> text`` sanitizer from a rules YAML + tag filter.

    Used by example plugins to register sanitizers without duplicating the
    apply loop. Rules are loaded once and cached on the returned closure.
    """
    path = Path(rules_path)

    @lru_cache(maxsize=1)
    def _rules() -> tuple[SanitizationRule, ...]:
        chosen = [r for r in load_rules(path) if tag in r.applies_to]
        if length_preserving_only:
            chosen = [r for r in chosen if r.length_preserving]
        return tuple(chosen)

    def sanitize(text: str) -> str:
        return apply_rules(text, _rules())

    sanitize.__name__ = f"sanitize_{tag}"
    sanitize.__qualname__ = f"sanitize_{tag}"
    return sanitize
