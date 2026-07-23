"""Task constants for support-ticket triage.

The routing contract (``category`` → ``team``) is a deterministic rule the
model must honour. It is stated in ``datasets/prompt_spec.json`` and enforced
out-of-model here, so a triage response is checked against a contract rather
than trusted because it looks well-formed.
"""
from __future__ import annotations

PRIORITIES = frozenset({"p1", "p2", "p3", "p4"})
CATEGORIES = frozenset({"billing", "access", "bug", "how_to", "other"})
TEAMS = frozenset({"payments", "identity", "platform", "docs", "general"})

# category → the team it must route to (from prompt_spec.json).
ROUTING: dict[str, str] = {
    "billing": "payments",
    "access": "identity",
    "bug": "platform",
    "how_to": "docs",
    "other": "general",
}

# Summaries are one-line agent hints, not paragraphs.
MAX_SUMMARY_WORDS = 20
