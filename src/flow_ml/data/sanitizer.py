from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Iterable

from ..utils.io import read_yaml


# PII / secret redaction rules ride with the package itself rather than living
# under a top-level configs/ folder.  They are global (apply to all models that
# call sanitize_jcl / sanitize_spool).
DEFAULT_RULES_PATH = Path(__file__).resolve().parent / "sanitization.yaml"


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
        applies_to=frozenset(raw.get("applies_to", ["jcl", "spool"])),
        length_preserving=bool(raw.get("length_preserving", False)),
    )


def load_rules(path: str | Path | None = None) -> list[SanitizationRule]:
    p = Path(path) if path else DEFAULT_RULES_PATH
    raw = read_yaml(p)
    return [_compile_rule(r) for r in raw["rules"]]


@lru_cache(maxsize=1)
def _default_rules() -> tuple[SanitizationRule, ...]:
    return tuple(load_rules())


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


def sanitize_jcl(text: str, rules: Iterable[SanitizationRule] | None = None) -> str:
    chosen = list(rules) if rules is not None else list(_default_rules())
    chosen = [r for r in chosen if "jcl" in r.applies_to and r.length_preserving]
    return apply_rules(text, chosen)


def sanitize_spool(text: str, rules: Iterable[SanitizationRule] | None = None) -> str:
    chosen = list(rules) if rules is not None else list(_default_rules())
    chosen = [r for r in chosen if "spool" in r.applies_to]
    return apply_rules(text, chosen)
