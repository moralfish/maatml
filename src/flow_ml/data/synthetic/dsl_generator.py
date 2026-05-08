"""
Rule-based augmenter for Flow DSL training samples.

Starting from ~30 hand-authored seed samples, produces ~1500 augmented
(description, dsl) pairs by applying random substitutions to the structured
DSL representation, then validating each candidate through the Rust parser
via the flow_dsl_py binding (falls back to a regex heuristic when the wheel
is not installed).

Usage
-----
    from flow_ml.data.synthetic.dsl_generator import augment_seeds

    samples = augment_seeds(
        seed_path="datasets/dsl_generator/samples/seed_samples.jsonl",
        target_count=1500,
        seed=42,
    )
    # Each item: {"sample_id": "aug-...", "source": "aug:<seed_id>", "description": ..., "dsl": ...}
"""
from __future__ import annotations

import copy
import json
import random
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

try:
    import flow_dsl_py as _dsl_py  # type: ignore[import]
    _HAVE_PARSER = True
except ImportError:
    _dsl_py = None
    _HAVE_PARSER = False


# ---------------------------------------------------------------------------
# DSL structural representation
# ---------------------------------------------------------------------------

@dataclass
class DslNode:
    node_id: str
    kind: str          # action | ai | cloud_ai | utility
    label: str
    fields: dict[str, Any]   # preserves insertion order; values typed (str/int/float)


@dataclass
class DslEdge:
    source: str
    outcome: Optional[str]   # "pass" | "fail" | "always" | None (bare arrow)
    target: str


@dataclass
class DslGraph:
    name: str
    version: str
    nodes: list[DslNode] = field(default_factory=list)
    edges: list[DslEdge] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------

_FLOW_HDR = re.compile(r'^flow\s+"([^"]+)"\s+(v\S+)')
_NODE_HDR = re.compile(r'^([\w][\w-]*)\[(\w+):\s*"([^"]*)"\]\s*\{')
_FIELD_STR = re.compile(r'^\s+([\w]+):\s+"([^"]*)"')
_FIELD_NUM = re.compile(r'^\s+([\w]+):\s+(-?[0-9]+(?:\.[0-9]+)?)\s*$')
_EDGE = re.compile(r'^([\w][\w-]*)(?:\.(pass|fail|always))?\s+-->\s+([\w][\w-]*)')


def parse_dsl(text: str) -> DslGraph:
    """Parse a canonical Flow DSL document into a DslGraph.

    Raises ValueError on unrecognised structure.
    """
    lines = text.splitlines()
    graph: Optional[DslGraph] = None
    current_node: Optional[DslNode] = None
    in_node = False

    for raw in lines:
        line = raw.rstrip()
        if not line:
            if in_node:
                pass  # blank lines inside blocks are not expected but tolerated
            continue

        if graph is None:
            m = _FLOW_HDR.match(line)
            if m:
                graph = DslGraph(name=m.group(1), version=m.group(2))
            continue

        if in_node:
            if line.strip() == "}":
                assert current_node is not None
                graph.nodes.append(current_node)
                current_node = None
                in_node = False
                continue
            m = _FIELD_STR.match(line)
            if m:
                current_node.fields[m.group(1)] = m.group(2)  # type: ignore[union-attr]
                continue
            m = _FIELD_NUM.match(line)
            if m:
                raw_val = m.group(2)
                current_node.fields[m.group(1)] = (  # type: ignore[union-attr]
                    int(raw_val) if "." not in raw_val else float(raw_val)
                )
                continue
            continue

        m = _NODE_HDR.match(line)
        if m:
            current_node = DslNode(
                node_id=m.group(1),
                kind=m.group(2),
                label=m.group(3),
                fields={},
            )
            in_node = True
            continue

        m = _EDGE.match(line)
        if m and graph is not None:
            graph.edges.append(DslEdge(
                source=m.group(1),
                outcome=m.group(2),
                target=m.group(3),
            ))
            continue

    if graph is None:
        raise ValueError("no flow header found")
    return graph


# ---------------------------------------------------------------------------
# Serializer  (matches the canonical Rust serializer layout)
# ---------------------------------------------------------------------------

def _fval(v: Any) -> str:
    if isinstance(v, str):
        return f'"{v}"'
    if isinstance(v, float):
        s = f"{v:.10f}".rstrip("0")
        if s.endswith("."):
            s += "0"
        return s
    return str(v)


def serialize_dsl(g: DslGraph) -> str:
    parts: list[str] = []
    parts.append(f'flow "{g.name}" {g.version}')
    for node in g.nodes:
        parts.append("")
        parts.append(f'{node.node_id}[{node.kind}: "{node.label}"] {{')
        for k, v in node.fields.items():
            parts.append(f"  {k}: {_fval(v)}")
        parts.append("}")
    if g.edges:
        parts.append("")
        for edge in g.edges:
            qual = f".{edge.outcome}" if edge.outcome else ""
            parts.append(f"{edge.source}{qual} --> {edge.target}")
    parts.append("")
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

_FLOW_HDR_RE = re.compile(r'flow\s+"[^"]+"\s+v\S+')


def _valid(dsl: str) -> bool:
    if _HAVE_PARSER:
        return _dsl_py.parses(dsl)  # type: ignore[union-attr]
    # Heuristic fallback: header present and no obvious syntax issues
    return bool(_FLOW_HDR_RE.search(dsl))


# ---------------------------------------------------------------------------
# Substitution pools
# ---------------------------------------------------------------------------

_VERSIONS = ["v1.0.0", "v1.1.0", "v1.2.0", "v2.0.0", "v0.9.0", "v1.0.1"]

_ACTION_CONFIGS: list[dict[str, str]] = [
    {"adapter": "zowe",   "actionId": "submit-jcl"},
    {"adapter": "zowe",   "actionId": "get-job-output"},
    {"adapter": "zosmf",  "actionId": "get-job-status"},
    {"adapter": "zosmf",  "actionId": "list-datasets"},
    {"adapter": "ssh",    "actionId": "deploy.sh"},
    {"adapter": "ssh",    "actionId": "run-command"},
    {"adapter": "shell",  "actionId": "run-command"},
    {"adapter": "mock",   "actionId": "echo"},
    {"adapter": "mock",   "actionId": "noop"},
]

_CLOUD_AI_CONFIGS: list[dict[str, Any]] = [
    {"provider": "claude",  "modelId": "claude-3-5-sonnet-latest"},
    {"provider": "claude",  "modelId": "claude-3-haiku-20240307"},
    {"provider": "openai",  "modelId": "gpt-4o"},
    {"provider": "openai",  "modelId": "gpt-4o-mini"},
    {"provider": "gemini",  "modelId": "gemini-1.5-pro"},
    {"provider": "gemini",  "modelId": "gemini-1.5-flash"},
]

_CLOUD_MAX_TOKENS: list[int] = [256, 512, 1024]
_CLOUD_TEMPERATURES: list[float] = [0.0, 0.1, 0.2, 0.3]

_AI_THRESHOLDS: list[tuple[float, float]] = [
    (0.9, 0.6),
    (0.95, 0.7),
    (0.85, 0.5),
    (0.8, 0.4),
    (0.92, 0.65),
    (0.88, 0.55),
]

# Utility IDs grouped by semantic category so mutations stay coherent:
# swapping within a category preserves the description's intent.
_UTILITY_CATEGORIES: dict[str, list[str]] = {
    "notify":  ["send-email", "post-slack"],
    "archive": ["archive-s3"],
    "oncall":  ["page-on-call", "create-ticket", "create-jira"],
    "cleanup": ["cleanup-tmp"],
    "storage": ["write-db"],
}
# Flat list kept for fallback / general use
_UTILITY_IDS: list[str] = [uid for ids in _UTILITY_CATEGORIES.values() for uid in ids]

def _utility_category(uid: str) -> str:
    for cat, ids in _UTILITY_CATEGORIES.items():
        if uid in ids:
            return cat
    return "other"

# Node-ID synonym groups (key = canonical form used in seeds; values = alternates)
_ID_SYNONYMS: dict[str, list[str]] = {
    "validate": ["validate", "check", "verify", "inspect", "lint"],
    "submit":   ["submit",   "deploy", "dispatch", "run",    "execute"],
    "archive":  ["archive",  "store",  "persist",  "save",   "backup"],
    "notify":   ["notify",   "alert",  "email",    "ping",   "escalate"],
    "analyze":  ["analyze",  "process","review",   "scan",   "examine"],
    "triage":   ["triage",   "evaluate","diagnose","classify"],
    "draft":    ["draft",    "generate","compose", "write",  "produce"],
    "summarize":["summarize","recap",  "condense", "digest"],
    "cleanup":  ["cleanup",  "clean",  "purge",    "prune",  "sweep"],
    "explain":  ["explain",  "describe","clarify", "elaborate"],
    "monitor":  ["monitor",  "watch",  "track",    "observe"],
    "check":    ["check",    "probe",  "test",     "ping",   "verify"],
    "health":   ["health",   "ping",   "probe",    "check"],
    "deploy":   ["deploy",   "run",    "execute",  "launch", "trigger"],
    "install":  ["install",  "setup",  "init"],
    "status":   ["status",   "check",  "query",    "probe"],
    "pwd":      ["pwd",      "cwd",    "dir",      "path"],
    "act":      ["act",      "run",    "exec",     "step"],
    "email":    ["email",    "notify", "alert",    "ping"],
    "ticket":   ["ticket",   "issue",  "task",     "jira"],
    "diskcheck":["diskcheck","capacity","space",   "df"],
    "summarize_spool": ["summarize", "digest", "recap", "condense"],
    "explain_fail": ["explain", "diagnose", "analyze", "describe"],
}

# Label synonym groups
_LABEL_SYNONYMS: dict[str, list[str]] = {
    "Validate JCL":      ["Validate JCL",    "Check JCL",      "Inspect JCL",   "Lint JCL"],
    "Validate":          ["Validate",         "Check",          "Inspect",       "Verify"],
    "Submit JCL":        ["Submit JCL",       "Run JCL Job",    "Deploy JCL",    "Execute JCL"],
    "Submit":            ["Submit",           "Deploy",         "Execute",       "Run"],
    "Archive":           ["Archive",          "Store",          "Persist",       "Backup"],
    "Archive spool":     ["Archive spool",    "Store spool",    "Persist output"],
    "Archive output":    ["Archive output",   "Store output",   "Persist result"],
    "Notify on-call":    ["Notify on-call",   "Alert on-call",  "Page on-call"],
    "Email":             ["Email",            "Send email",     "Notify",        "Alert"],
    "Email on-call":     ["Email on-call",    "Alert team",     "Page on-call",  "Escalate"],
    "Escalate":          ["Escalate",         "Page on-call",   "Alert team",    "Raise incident"],
    "Explain failure":   ["Explain failure",  "Analyze failure","Diagnose issue","Describe error"],
    "Summarize spool":   ["Summarize spool",  "Recap spool",    "Digest output", "Condense log"],
    "Draft note":        ["Draft note",       "Generate note",  "Compose note",  "Write note"],
    "Triage failure":    ["Triage failure",   "Diagnose failure","Assess error", "Evaluate failure"],
    "Mock":              ["Mock",             "Test",           "Echo",          "Stub"],
    "Job Status":        ["Job Status",       "Check Status",   "Query Status",  "Get Status"],
    "Run deploy":        ["Run deploy",       "Execute deploy", "Deploy",        "Launch deploy"],
    "Print PWD":         ["Print PWD",        "Show CWD",       "Print dir",     "Get path"],
    "Git Status":        ["Git Status",       "Check git",      "Git check",     "Repo status"],
    "Health":            ["Health",           "Health check",   "Ping",          "Probe"],
    "Cleanup":           ["Cleanup",          "Clean up",       "Purge",         "Clear"],
    "Send email":        ["Send email",       "Email",          "Notify",        "Alert"],
    "Create JIRA":       ["Create JIRA",      "Open ticket",    "Create issue",  "Log ticket"],
    "Check disk":        ["Check disk",       "Disk space",     "Capacity check","df"],
    "npm install":       ["npm install",      "Install deps",   "Setup packages"],
}


def _synonym(pool: dict[str, list[str]], key: str, rng: random.Random) -> str:
    """Pick a random synonym; falls back to key itself if not in pool."""
    candidates = pool.get(key)
    if candidates:
        return rng.choice(candidates)
    # Try a lower-cased lookup
    candidates = pool.get(key.lower())
    if candidates:
        chosen = rng.choice(candidates)
        return chosen[0].upper() + chosen[1:] if key[0].isupper() else chosen
    return key


# ---------------------------------------------------------------------------
# Description vocabulary for generating matching descriptions
# ---------------------------------------------------------------------------

_DESC_ADAPTER_MAP: dict[str, str] = {
    "zowe":  "via zowe",
    "zosmf": "via zosmf",
    "ssh":   "over ssh",
    "shell": "via shell command",
    "mock":  "using a mock adapter",
}

_DESC_PROVIDER_MAP: dict[str, str] = {
    "claude": "Claude",
    "openai": "OpenAI",
    "gemini": "Gemini",
}

_DESC_UTILITY_MAP: dict[str, str] = {
    "send-email":   "send an email",
    "archive-s3":   "archive to s3",
    "page-on-call": "page on-call",
    "create-ticket":"create a ticket",
    "post-slack":   "post to Slack",
    "write-db":     "write to the database",
    "cleanup-tmp":  "clean up temp files",
    "create-jira":  "create a JIRA issue",
}


def _rebuild_desc(original_desc: str, new_graph: DslGraph,
                  original_graph: DslGraph) -> str:
    """
    Attempt a lightweight description update when adapter / provider / utility
    fields changed.  Returns the original if no substitution rules apply.
    """
    desc = original_desc
    for orig_node, new_node in zip(original_graph.nodes, new_graph.nodes):
        if orig_node.kind == "action" and new_node.kind == "action":
            orig_adapter = orig_node.fields.get("adapter", "")
            new_adapter  = new_node.fields.get("adapter", "")
            if orig_adapter != new_adapter:
                orig_phrase = _DESC_ADAPTER_MAP.get(orig_adapter, f"via {orig_adapter}")
                new_phrase  = _DESC_ADAPTER_MAP.get(new_adapter, f"via {new_adapter}")
                desc = desc.replace(orig_phrase, new_phrase)
        elif orig_node.kind == "cloud_ai" and new_node.kind == "cloud_ai":
            orig_provider = orig_node.fields.get("provider", "")
            new_provider  = new_node.fields.get("provider", "")
            if orig_provider != new_provider:
                orig_name = _DESC_PROVIDER_MAP.get(orig_provider, orig_provider)
                new_name  = _DESC_PROVIDER_MAP.get(new_provider, new_provider)
                desc = desc.replace(orig_name, new_name)
        elif orig_node.kind == "utility" and new_node.kind == "utility":
            orig_uid = orig_node.fields.get("utilityId", "")
            new_uid  = new_node.fields.get("utilityId", "")
            if orig_uid != new_uid:
                orig_phrase = _DESC_UTILITY_MAP.get(orig_uid, orig_uid)
                new_phrase  = _DESC_UTILITY_MAP.get(new_uid, new_uid)
                desc = desc.replace(orig_phrase, new_phrase)
    return desc


# ---------------------------------------------------------------------------
# Core mutation functions
# ---------------------------------------------------------------------------

def _mutate_version(g: DslGraph, rng: random.Random) -> DslGraph:
    new = copy.deepcopy(g)
    new.version = rng.choice([v for v in _VERSIONS if v != g.version] or _VERSIONS)
    return new


def _mutate_node_ids(g: DslGraph, rng: random.Random) -> DslGraph:
    """Randomly rename some node IDs (and update edges to match)."""
    new = copy.deepcopy(g)
    rename: dict[str, str] = {}
    for node in new.nodes:
        nid = node.node_id
        # Find a synonym group that contains this ID
        for canonical, synonyms in _ID_SYNONYMS.items():
            if nid in synonyms or nid == canonical:
                alts = [s for s in synonyms if s != nid]
                if alts:
                    new_id = rng.choice(alts)
                    rename[nid] = new_id
                    node.node_id = new_id
                break
        # If multi-part ID like "validate-jcl", try the first part
        if nid not in rename and "-" in nid:
            prefix = nid.split("-")[0]
            for canonical, synonyms in _ID_SYNONYMS.items():
                if prefix in synonyms or prefix == canonical:
                    alts = [s for s in synonyms if s != prefix]
                    if alts:
                        new_prefix = rng.choice(alts)
                        new_id = nid.replace(prefix, new_prefix, 1)
                        rename[nid] = new_id
                        node.node_id = new_id
                    break
    for edge in new.edges:
        if edge.source in rename:
            edge.source = rename[edge.source]
        if edge.target in rename:
            edge.target = rename[edge.target]
    return new


def _mutate_labels(g: DslGraph, rng: random.Random) -> DslGraph:
    new = copy.deepcopy(g)
    for node in new.nodes:
        node.label = _synonym(_LABEL_SYNONYMS, node.label, rng)
    return new


def _mutate_flow_name(g: DslGraph, rng: random.Random) -> DslGraph:
    """Apply a minor variation to the flow name."""
    new = copy.deepcopy(g)
    prefixes = ["", "Auto-", "My ", "Prod "]
    suffixes = ["", " v2", " Pipeline", " Flow"]
    prefix = rng.choice(prefixes)
    suffix = rng.choice(suffixes)
    if prefix or suffix:
        new.name = prefix + g.name + suffix
    return new


def _mutate_action_fields(g: DslGraph, rng: random.Random) -> DslGraph:
    """Randomly swap adapter/actionId for action nodes."""
    new = copy.deepcopy(g)
    for node in new.nodes:
        if node.kind == "action" and rng.random() < 0.7:
            cfg = rng.choice(_ACTION_CONFIGS)
            fields: dict[str, Any] = {"adapter": cfg["adapter"], "actionId": cfg["actionId"]}
            if "args" in cfg:
                fields["args"] = cfg["args"]
            # Keep any extra fields like cwd, command if action allows them
            if cfg["adapter"] == "shell" and "command" in node.fields:
                fields["command"] = node.fields["command"]
            node.fields = fields
    return new


def _mutate_cloud_ai_fields(g: DslGraph, rng: random.Random) -> DslGraph:
    """Randomly swap provider/modelId for cloud_ai nodes; vary maxTokens/temperature."""
    new = copy.deepcopy(g)
    for node in new.nodes:
        if node.kind == "cloud_ai":
            cfg = rng.choice(_CLOUD_AI_CONFIGS)
            fields: dict[str, Any] = {
                "provider": cfg["provider"],
                "modelId":  cfg["modelId"],
            }
            if "prompt" in node.fields:
                fields["prompt"] = node.fields["prompt"]
            if "maxTokens" in node.fields:
                fields["maxTokens"] = rng.choice(_CLOUD_MAX_TOKENS)
            if "temperature" in node.fields:
                fields["temperature"] = rng.choice(_CLOUD_TEMPERATURES)
            node.fields = fields
    return new


def _mutate_ai_thresholds(g: DslGraph, rng: random.Random) -> DslGraph:
    """Vary thresholdHigh/thresholdLow for ai nodes that already carry thresholds."""
    new = copy.deepcopy(g)
    for node in new.nodes:
        if node.kind == "ai" and "thresholdHigh" in node.fields:
            hi, lo = rng.choice(_AI_THRESHOLDS)
            node.fields["thresholdHigh"] = hi
            node.fields["thresholdLow"]  = lo
    return new


def _mutate_utility_ids(g: DslGraph, rng: random.Random) -> DslGraph:
    """Swap utilityId within the same semantic category to keep descriptions coherent.

    e.g. "send-email" stays in the "notify" category and may become "post-slack",
    but will not become "archive-s3" (which would contradict a description that
    says "send a notification").
    """
    new = copy.deepcopy(g)
    for node in new.nodes:
        if node.kind != "utility" or rng.random() >= 0.5:
            continue
        current_uid = node.fields.get("utilityId", "")
        cat = _utility_category(current_uid)
        siblings = [u for u in _UTILITY_CATEGORIES.get(cat, []) if u != current_uid]
        if siblings:
            node.fields["utilityId"] = rng.choice(siblings)
    return new


# ---- Mutation menu ---------------------------------------------------------

_ALL_MUTATIONS = [
    _mutate_version,
    _mutate_node_ids,
    _mutate_labels,
    _mutate_flow_name,
    _mutate_action_fields,
    _mutate_cloud_ai_fields,
    _mutate_ai_thresholds,
    _mutate_utility_ids,
]


def _apply_random_mutations(g: DslGraph, rng: random.Random,
                             n_min: int = 1, n_max: int = 4) -> DslGraph:
    """Apply 1-4 random independent mutations."""
    n = rng.randint(n_min, n_max)
    mutations = rng.sample(_ALL_MUTATIONS, min(n, len(_ALL_MUTATIONS)))
    mutated = g
    for mut in mutations:
        mutated = mut(mutated, rng)
    return mutated


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def augment_seeds(
    seed_path: str | Path,
    target_count: int = 1500,
    seed: int = 42,
    validate: bool = True,
) -> list[dict[str, Any]]:
    """
    Augment the seed JSONL file and return a list of training sample dicts.

    Parameters
    ----------
    seed_path:
        Path to the seed JSONL file (each line has sample_id, source,
        description, dsl).
    target_count:
        Desired number of augmented samples (excluding seeds).
    seed:
        RNG seed for reproducibility.
    validate:
        If True, every generated DSL is validated before inclusion.  Set False
        only for unit tests that run without the flow_dsl_py wheel.

    Returns
    -------
    List of sample dicts (same schema as the seed file, source = "aug:<seed_id>").
    """
    rng = random.Random(seed)
    seed_path = Path(seed_path)
    raw_seeds: list[dict[str, Any]] = [
        json.loads(line)
        for line in seed_path.read_text().splitlines()
        if line.strip()
    ]

    # Parse seeds into structural form; skip any that fail to parse
    parsed: list[tuple[dict[str, Any], DslGraph]] = []
    for raw in raw_seeds:
        try:
            g = parse_dsl(raw["dsl"])
            parsed.append((raw, g))
        except Exception:
            pass  # skip malformed seeds

    if not parsed:
        raise ValueError(f"No seeds could be parsed from {seed_path}")

    results: list[dict[str, Any]] = []
    seen_dsl: set[str] = {r["dsl"] for r in raw_seeds}
    aug_index = 0
    attempts = 0
    max_attempts = target_count * 30  # safety cap

    while len(results) < target_count and attempts < max_attempts:
        attempts += 1
        raw_seed, base_graph = rng.choice(parsed)

        mutated = _apply_random_mutations(base_graph, rng)
        dsl_text = serialize_dsl(mutated)

        if dsl_text in seen_dsl:
            continue
        if validate and not _valid(dsl_text):
            continue

        new_desc = _rebuild_desc(raw_seed["description"], mutated, base_graph)

        aug_index += 1
        results.append({
            "sample_id": f"aug-{aug_index:05d}",
            "source":    f"aug:{raw_seed['sample_id']}",
            "description": new_desc,
            "dsl": dsl_text,
        })
        seen_dsl.add(dsl_text)

    return results


def generate_augmented_jsonl(
    seed_path: str | Path,
    out_path: str | Path,
    target_count: int = 1500,
    seed: int = 42,
    include_seeds: bool = True,
) -> int:
    """
    Generate augmented JSONL file and write to out_path.

    Returns total number of lines written.
    """
    seed_path = Path(seed_path)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    raw_seeds: list[dict[str, Any]] = [
        json.loads(line)
        for line in seed_path.read_text().splitlines()
        if line.strip()
    ]
    augmented = augment_seeds(seed_path=seed_path, target_count=target_count, seed=seed)

    lines: list[dict[str, Any]] = []
    if include_seeds:
        lines.extend(raw_seeds)
    lines.extend(augmented)

    with out_path.open("w", encoding="utf-8") as fh:
        for rec in lines:
            fh.write(json.dumps(rec) + "\n")

    return len(lines)
