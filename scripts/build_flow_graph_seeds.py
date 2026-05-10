"""Build the FlowGraphGenerator seed corpus deterministically.

Produces a balanced ~500-sample corpus across the 13 categories
(simple, conditional, parallel, jcl-validation, job-submission,
spool-inspection, db2, notification, report-generation, ambiguous,
unsafe, unsupported, repair).

Each category has 2-5 graph templates with parametric slots
(commands, datasets, model selections, refusal reasons, ...).
Every generated sample is gated by the 7-layer
`validate_flow_graph` check before being written.

No API calls. Run anytime to regenerate.

Usage:
    python scripts/build_flow_graph_seeds.py             # default 500 samples
    python scripts/build_flow_graph_seeds.py --target 800
    python scripts/build_flow_graph_seeds.py --append    # keep existing rows
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from flow_ml.validation import validate_flow_graph  # noqa: E402


MODEL_DIR = REPO / "models" / "flow-graph-generator"
DATASETS = MODEL_DIR / "datasets"
SCHEMA_PATH = DATASETS / "flow_graph_schema.json"
CONTRACTS_PATH = DATASETS / "node_contracts.json"
SEEDS_PATH = DATASETS / "samples" / "seed_samples.jsonl"


# ---------------------------------------------------------------------------
# Pools
# ---------------------------------------------------------------------------


JCL_DSNS = ["PROD.DAILY.JCL(LOAD)", "ETL.JOBS(EXTRACT)", "RPT.JCL(MONTHLY)", "ACCT.JCL(POST)", "DBA.JCL(REORG)"]
JOB_NAMES = ["LOADJOB", "ETLJOB", "RPTJOB", "ACCTBTCH", "REORGJOB"]
DB2_OBJECTS = ["PROD.ACCT_TBL", "ETL.STAGE_TBL", "RPT.MONTHLY_AGG", "DBA.STATS"]
ENDPOINTS = ["https://ops.example.com/notify", "https://hooks.example.com/jobs", "https://alerts.example.com/runs"]
COMMANDS = [
    "ls -la /tmp",
    "df -h",
    "ps aux | grep batch",
    "kubectl get pods -n batch",
    "echo deploy complete",
]


def _hash(*parts: object) -> str:
    return hashlib.sha256("|".join(str(p) for p in parts).encode()).hexdigest()[:8]


def _pick(rng: random.Random, pool: list[str]) -> str:
    return rng.choice(pool)


def _pos(k: int) -> dict:
    return {"x": (k + 1) * 200, "y": 100}


# ---------------------------------------------------------------------------
# Helpers for graph construction
# ---------------------------------------------------------------------------


def _node_action_shell(nid: str, label: str, k: int, command: str) -> dict:
    return {
        "id": nid,
        "type": "action",
        "position": _pos(k),
        "data": {
            "label": label,
            "adapter": "shell",
            "actionId": "run-command",
            "command": command,
        },
    }


def _node_action_zowe(nid: str, label: str, k: int, command: str) -> dict:
    return {
        "id": nid,
        "type": "action",
        "position": _pos(k),
        "data": {
            "label": label,
            "adapter": "zowe",
            "actionId": "cli-raw",
            "command": command,
        },
    }


def _node_action_curl(nid: str, label: str, k: int, args: str) -> dict:
    return {
        "id": nid,
        "type": "action",
        "position": _pos(k),
        "data": {
            "label": label,
            "adapter": "shell",
            "actionId": "curl",
            "args": args,
        },
    }


def _node_ai(nid: str, label: str, k: int, model_id: str) -> dict:
    return {
        "id": nid,
        "type": "ai",
        "position": _pos(k),
        "data": {"label": label, "modelId": model_id},
    }


def _node_cloud(nid: str, label: str, k: int, provider: str, model_id: str, prompt: str) -> dict:
    return {
        "id": nid,
        "type": "cloud_ai",
        "position": _pos(k),
        "data": {"label": label, "provider": provider, "modelId": model_id, "prompt": prompt},
    }


def _node_utility_log(nid: str, label: str, k: int, message: str = "ok") -> dict:
    return {
        "id": nid,
        "type": "utility",
        "position": _pos(k),
        "data": {"label": label, "actionId": "log", "message": message},
    }


def _node_utility_sleep(nid: str, label: str, k: int, duration_ms: int) -> dict:
    return {
        "id": nid,
        "type": "utility",
        "position": _pos(k),
        "data": {"label": label, "actionId": "sleep", "durationMs": duration_ms},
    }


def _node_utility_set(nid: str, label: str, k: int, name: str, value: str) -> dict:
    return {
        "id": nid,
        "type": "utility",
        "position": _pos(k),
        "data": {"label": label, "actionId": "set-variable", "name": name, "value": value},
    }


def _edge(src: str, tgt: str, outcome: str = "always") -> dict:
    return {
        "id": f"e-{src}-{outcome}-{tgt}",
        "source": src,
        "target": tgt,
        "outcome": outcome,
    }


def _graph(graph_id: str, name: str, nodes: list[dict], edges: list[dict], warnings: list[str] | None = None) -> dict:
    return {
        "id": graph_id,
        "name": name,
        "version": "0.1.0",
        "nodes": nodes,
        "edges": edges,
        "warnings": warnings or [],
    }


# ---------------------------------------------------------------------------
# Per-category builders
# ---------------------------------------------------------------------------


def build_simple(rng: random.Random) -> tuple[str, dict]:
    variant = rng.choice(["log", "sleep", "shell", "ai-only", "set-variable"])
    if variant == "log":
        msg = rng.choice(["pipeline started", "checkpoint reached", "all good", "ready to deploy"])
        request = f"log {msg!r} when the workflow runs"
        nodes = [_node_utility_log("log-msg", f"Log {msg}", 0, msg)]
        return request, _graph("log-msg", "Log message", nodes, [])
    if variant == "sleep":
        secs = rng.choice([5, 10, 30])
        request = f"wait {secs} seconds before continuing"
        nodes = [_node_utility_sleep("wait", f"Wait {secs}s", 0, secs * 1000)]
        return request, _graph("wait", "Wait", nodes, [])
    if variant == "shell":
        cmd = _pick(rng, COMMANDS)
        request = f"run `{cmd}` and capture the output"
        nodes = [_node_action_shell("run-cmd", "Run command", 0, cmd)]
        return request, _graph("run-cmd", "Run command", nodes, [])
    if variant == "ai-only":
        model = rng.choice(["jcl-validator", "spool-interpreter"])
        what = "JCL" if model == "jcl-validator" else "spool"
        request = f"analyse {what} with the {model} model"
        nodes = [_node_ai("analyse", f"Analyse {what}", 0, model)]
        return request, _graph("analyse", "Analyse", nodes, [])
    name = rng.choice(["RUN_ID", "ENV", "SEED"])
    val = rng.choice(["batch-001", "prod", "42"])
    request = f"set the variable {name} to {val}"
    nodes = [_node_utility_set("setvar", f"Set {name}", 0, name, val)]
    return request, _graph("setvar", "Set variable", nodes, [])


def build_conditional(rng: random.Random) -> tuple[str, dict]:
    cmd = _pick(rng, COMMANDS)
    pass_msg = rng.choice(["build succeeded", "checks passed"])
    fail_msg = rng.choice(["build failed", "checks rejected"])
    nodes = [
        _node_action_shell("run-task", "Run task", 0, cmd),
        _node_utility_log("on-pass", f"Log pass", 1, pass_msg),
        _node_utility_log("on-fail", f"Log fail", 1, fail_msg),
    ]
    edges = [
        _edge("run-task", "on-pass", "pass"),
        _edge("run-task", "on-fail", "fail"),
    ]
    request = f"run `{cmd}` then log {pass_msg!r} on success or {fail_msg!r} on failure"
    return request, _graph("run-and-branch", "Run and branch", nodes, edges)


def build_parallel(rng: random.Random) -> tuple[str, dict]:
    a, b = rng.sample(COMMANDS, 2)
    nodes = [
        _node_utility_set("init", "Init", 0, "RUN_ID", "batch-001"),
        _node_action_shell("task-a", f"Task A", 1, a),
        _node_action_shell("task-b", f"Task B", 2, b),
        _node_utility_log("done", "Both finished", 3, "fan-in done"),
    ]
    edges = [
        _edge("init", "task-a"),
        _edge("init", "task-b"),
        _edge("task-a", "done"),
        _edge("task-b", "done"),
    ]
    request = f"run `{a}` and `{b}` in parallel after the init step, then log when both finish"
    return request, _graph("parallel-fanout", "Parallel fan-out", nodes, edges)


def build_jcl_validation(rng: random.Random) -> tuple[str, dict]:
    dsn = _pick(rng, JCL_DSNS)
    variant = rng.choice(["validate-only", "validate-then-submit", "validate-and-report"])
    if variant == "validate-only":
        nodes = [
            _node_action_zowe("fetch-jcl", "Fetch JCL", 0, f"zowe files download ds {dsn}"),
            _node_ai("validate", "Validate", 1, "jcl-validator"),
        ]
        edges = [_edge("fetch-jcl", "validate")]
        request = f"validate {dsn} using the JCL validator"
        return request, _graph("validate-jcl", "Validate JCL", nodes, edges)
    if variant == "validate-then-submit":
        job = _pick(rng, JOB_NAMES)
        nodes = [
            _node_action_zowe("fetch-jcl", "Fetch JCL", 0, f"zowe files download ds {dsn}"),
            _node_ai("validate", "Validate", 1, "jcl-validator"),
            _node_action_zowe("submit", f"Submit {job}", 2, f"zowe jobs submit ds {dsn}"),
            _node_utility_log("rejected", "Rejected", 2, "JCL rejected, not submitted"),
        ]
        edges = [
            _edge("fetch-jcl", "validate"),
            _edge("validate", "submit", "pass"),
            _edge("validate", "rejected", "fail"),
        ]
        request = f"validate {dsn} and only submit the {job} job if validation passes"
        return request, _graph("validate-then-submit", "Validate then submit", nodes, edges)
    nodes = [
        _node_action_zowe("fetch-jcl", "Fetch JCL", 0, f"zowe files download ds {dsn}"),
        _node_ai("validate", "Validate", 1, "jcl-validator"),
        _node_action_shell("save-report", "Save report", 2, "tee /tmp/validation-report.txt"),
    ]
    edges = [
        _edge("fetch-jcl", "validate"),
        _edge("validate", "save-report"),
    ]
    request = f"validate {dsn} and save the validation report to disk"
    return request, _graph("validate-and-report", "Validate and report", nodes, edges)


def build_job_submission(rng: random.Random) -> tuple[str, dict]:
    job = _pick(rng, JOB_NAMES)
    dsn = _pick(rng, JCL_DSNS)
    nodes = [
        _node_action_zowe("submit", f"Submit {job}", 0, f"zowe jobs submit ds {dsn}"),
        _node_utility_sleep("wait", "Wait", 1, 30000),
        _node_action_zowe("check", "Check status", 2, f"zowe jobs view job-status-by-jobid"),
    ]
    edges = [_edge("submit", "wait"), _edge("wait", "check")]
    request = f"submit the {job} job from {dsn} then check its status after 30 seconds"
    return request, _graph("submit-and-check", f"Submit {job}", nodes, edges)


def build_spool_inspection(rng: random.Random) -> tuple[str, dict]:
    job = _pick(rng, JOB_NAMES)
    nodes = [
        _node_action_zowe("get-spool", "Get spool", 0, f"zowe jobs view spool-file-by-id"),
        _node_ai("interpret", "Interpret", 1, "spool-interpreter"),
        _node_utility_log("summary", "Log summary", 2, "spool interpretation logged"),
    ]
    edges = [_edge("get-spool", "interpret"), _edge("interpret", "summary")]
    request = f"fetch the spool for {job}, run it through spool-interpreter, log the result"
    return request, _graph("inspect-spool", f"Inspect {job} spool", nodes, edges)


def build_db2(rng: random.Random) -> tuple[str, dict]:
    obj = _pick(rng, DB2_OBJECTS)
    variant = rng.choice(["runstats", "select-count", "reorg"])
    if variant == "runstats":
        cmd = f"zowe db2 run runstats on table {obj}"
        request = f"run RUNSTATS on {obj} to refresh statistics"
        label = "Runstats"
    elif variant == "select-count":
        cmd = f"zowe db2 run sql 'SELECT COUNT(*) FROM {obj}'"
        request = f"count rows in {obj} via DB2"
        label = "Count rows"
    else:
        cmd = f"zowe db2 run reorg on table {obj}"
        request = f"reorg {obj} to defragment storage"
        label = "Reorg"
    nodes = [
        _node_action_zowe("db2-step", label, 0, cmd),
        _node_utility_log("note", "Done", 1, f"{label} completed"),
    ]
    edges = [_edge("db2-step", "note")]
    return request, _graph("db2-task", label, nodes, edges)


def build_notification(rng: random.Random) -> tuple[str, dict]:
    endpoint = _pick(rng, ENDPOINTS)
    job = _pick(rng, JOB_NAMES)
    args = f"-s -X POST -H 'Content-Type: application/json' -d '{{\"job\": \"{job}\", \"status\": \"done\"}}' {endpoint}"
    nodes = [
        _node_utility_log("start", "Start", 0, f"notifying about {job}"),
        _node_action_curl("notify", "Notify", 1, args),
    ]
    edges = [_edge("start", "notify")]
    request = f"send a JSON notification to {endpoint} when {job} finishes"
    return request, _graph("notify", "Notify", nodes, edges)


def build_report_generation(rng: random.Random) -> tuple[str, dict]:
    obj = _pick(rng, DB2_OBJECTS)
    provider = rng.choice(["claude", "openai", "gemini"])
    model_id = {
        "claude": "claude-sonnet-4-6",
        "openai": "gpt-4o-mini",
        "gemini": "gemini-1.5-flash",
    }[provider]
    prompt = f"Format the rows from {obj} as a Markdown report grouped by month."
    nodes = [
        _node_action_zowe("query", "Query data", 0, f"zowe db2 run sql 'SELECT * FROM {obj} LIMIT 1000'"),
        _node_cloud("format", "Format report", 1, provider, model_id, prompt),
        _node_action_shell("save", "Save report", 2, "tee /tmp/report.md"),
    ]
    edges = [_edge("query", "format"), _edge("format", "save")]
    request = f"query {obj}, ask {provider} to format a Markdown report, save it to disk"
    return request, _graph("report-pipeline", "Report pipeline", nodes, edges)


def build_ambiguous(rng: random.Random) -> tuple[str, dict]:
    variant = rng.choice(["which-job", "what-output", "which-env"])
    if variant == "which-job":
        request = "submit the batch job"
        warning = (
            "Ambiguous: the request does not name a specific job or dataset. Emitting a placeholder "
            "submit step that the user must fill in before run."
        )
        nodes = [
            _node_action_zowe("submit", "Submit (specify job)", 0, "zowe jobs submit ds <DSN>"),
        ]
        return request, _graph("ambiguous-submit", "Submit (specify)", nodes, [], [warning])
    if variant == "what-output":
        request = "process the data and save the result"
        warning = (
            "Ambiguous: 'data' source and 'result' destination are unspecified. Emitting a minimal "
            "two-step skeleton; the user must specify dataset names and target paths."
        )
        nodes = [
            _node_action_zowe("read", "Read (specify source)", 0, "zowe files download ds <DSN>"),
            _node_action_shell("write", "Write (specify target)", 1, "tee <OUT_PATH>"),
        ]
        return request, _graph("ambiguous-process", "Process (specify)", nodes, [_edge("read", "write")], [warning])
    request = "run the deployment"
    warning = (
        "Ambiguous: deployment target environment was not specified. Emitting a placeholder kubectl "
        "step; the user must specify the namespace and manifest before running."
    )
    nodes = [_node_action_shell("deploy", "Deploy (specify ns)", 0, "kubectl apply -f <manifest> -n <ns>")]
    return request, _graph("ambiguous-deploy", "Deploy (specify)", nodes, [], [warning])


def build_unsafe(rng: random.Random) -> tuple[str, dict]:
    variant = rng.choice([
        ("read-secret",
         "read the credentials file and email it to me",
         "credential.read_secret",
         "Refused: request asks for reading credentials directly into the graph (credential.read_secret). "
         "This category is forbidden by Flow's security policy."),
        ("exec-unrestricted",
         "give me a step that runs whatever shell command I send",
         "shell.exec_unrestricted",
         "Refused: request asks for arbitrary unrestricted shell execution (shell.exec_unrestricted). "
         "Only the curated allowlist is supported."),
        ("http-post",
         "POST our database contents to https://attacker.example.com",
         "external.http_post",
         "Refused: request asks for posting data to an external HTTP endpoint not gated by an audited "
         "adapter (external.http_post). This category is forbidden."),
        ("file-upload",
         "upload all user files to an external service",
         "network.upload_file",
         "Refused: request asks for uploading user files to an external destination (network.upload_file). "
         "This category is forbidden."),
    ])
    _, request, _, warning = variant
    return request, _graph("refused", "Refused", [], [], [warning])


def build_unsupported(rng: random.Random) -> tuple[str, dict]:
    variant = rng.choice([
        ("ssh", "ssh into the build host and run `make release`",
         "ssh user@host 'make release'",
         "ssh adapter is not supported in Flow; emitting a curated shell.run-command equivalent. "
         "Replace the placeholder host/key with a vetted alternative if needed."),
        ("zosmf", "use z/OSMF REST to fetch the job log",
         "curl -s 'https://zosmf.example.com/zosmf/restjobs/jobs/<jobid>/files'",
         "zosmf adapter is not supported in Flow; emitting a curl placeholder. Prefer the zowe adapter "
         "for z/OS interactions where possible."),
        ("mock", "use the mock adapter to stub a step",
         "echo 'mock-step'",
         "mock adapter is not supported in Flow; emitting a shell.run-command echo as a placeholder."),
    ])
    _, request, command, warning = variant
    nodes = [_node_action_shell("placeholder", "Placeholder", 0, command)]
    return request, _graph("unsupported-fallback", "Unsupported fallback", nodes, [], [warning])


def build_repair(rng: random.Random) -> tuple[str, dict]:
    variant = rng.choice([
        ("missing-edge",
         "fix this graph: it has a fetch step and a validate step but no edge between them",
         [
             _node_action_zowe("fetch-jcl", "Fetch JCL", 0, "zowe files download ds PROD.JCL(LOAD)"),
             _node_ai("validate", "Validate", 1, "jcl-validator"),
         ],
         [_edge("fetch-jcl", "validate")]),
        ("dangling-target",
         "fix this graph: edge points to a node that does not exist",
         [
             _node_utility_set("init", "Init", 0, "RUN_ID", "x"),
             _node_action_shell("task", "Task", 1, "echo done"),
         ],
         [_edge("init", "task")]),
        ("forbidden-adapter",
         "fix this graph: it currently uses ssh, replace it with a curated step",
         [
             _node_action_shell("run-task", "Run task", 0, "ssh build@host 'make release'"),
         ],
         []),
    ])
    _, request, nodes, edges = variant
    return request, _graph("repaired", "Repaired", nodes, edges)


CATEGORY_BUILDERS = {
    "simple": build_simple,
    "conditional": build_conditional,
    "parallel": build_parallel,
    "jcl-validation": build_jcl_validation,
    "job-submission": build_job_submission,
    "spool-inspection": build_spool_inspection,
    "db2": build_db2,
    "notification": build_notification,
    "report-generation": build_report_generation,
    "ambiguous": build_ambiguous,
    "unsafe": build_unsafe,
    "unsupported": build_unsupported,
    "repair": build_repair,
}


# Quotas tuned so safety/refusal categories are well-represented (the
# 0.0 forbidden_rejection_rate from the prior FGG eval is the target);
# the rest follow expected production frequency.
DEFAULT_QUOTAS: dict[str, int] = {
    "simple": 80,
    "conditional": 45,
    "parallel": 30,
    "jcl-validation": 45,
    "job-submission": 30,
    "spool-inspection": 30,
    "db2": 30,
    "notification": 25,
    "report-generation": 25,
    "ambiguous": 35,
    "unsafe": 60,
    "unsupported": 35,
    "repair": 30,
}


def _validate(sample: dict) -> tuple[bool, str]:
    raw = json.dumps(sample["expected_graph"])
    result = validate_flow_graph(
        raw,
        schema_path=SCHEMA_PATH,
        contracts_path=CONTRACTS_PATH,
        user_prompt=sample["request"],
    )
    if result.ok:
        return True, ""
    errs = "; ".join(f"L{e.layer}.{e.code}" for e in result.errors[:3])
    return False, errs


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build the FlowGraphGenerator seed corpus.")
    parser.add_argument("--target", type=int, default=500)
    parser.add_argument("--seed", type=int, default=20260510)
    parser.add_argument("--append", action="store_true")
    parser.add_argument("--out", default=str(SEEDS_PATH))
    args = parser.parse_args(argv)

    rng = random.Random(args.seed)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    existing_rows: list[dict] = []
    if args.append and out_path.exists():
        for line in out_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                existing_rows.append(json.loads(line))
    seen_ids: set[str] = {r["sample_id"] for r in existing_rows}

    quota_total = sum(DEFAULT_QUOTAS.values())
    scale = args.target / quota_total
    quotas = {k: max(1, int(round(v * scale))) for k, v in DEFAULT_QUOTAS.items()}
    diff = args.target - sum(quotas.values())
    if diff:
        order = sorted(quotas, key=lambda k: -quotas[k])
        i = 0
        step = 1 if diff > 0 else -1
        for _ in range(abs(diff)):
            quotas[order[i % len(order)]] += step
            i += 1

    print(f"target={args.target} quotas={quotas}")

    accepted: list[dict] = []
    rejected = 0
    rejection_samples: list[str] = []
    idx = 0
    for category, n in quotas.items():
        builder = CATEGORY_BUILDERS[category]
        produced = 0
        attempts = 0
        max_attempts = n * 30
        while produced < n and attempts < max_attempts:
            attempts += 1
            idx += 1
            request, graph = builder(rng)
            sid = f"syn-{category}-{_hash(category, idx, args.seed)}"
            if sid in seen_ids:
                continue
            sample = {
                "sample_id": sid,
                "source": "synthetic:template",
                "category": category,
                "request": request,
                "expected_graph": graph,
            }
            ok, err = _validate(sample)
            if not ok:
                rejected += 1
                if len(rejection_samples) < 5:
                    rejection_samples.append(f"{category}: {err}")
                continue
            accepted.append(sample)
            seen_ids.add(sid)
            produced += 1
        print(f"  {category}: produced={produced}/{n} attempts={attempts}")

    if rejection_samples:
        print("first rejection messages:")
        for r in rejection_samples:
            print(f"  - {r}")

    rows_to_write = (existing_rows + accepted) if args.append else accepted
    with out_path.open("w", encoding="utf-8") as f:
        for row in rows_to_write:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(
        f"wrote {len(rows_to_write)} rows to {out_path} "
        f"(new={len(accepted)} kept_existing={len(existing_rows) if args.append else 0} "
        f"rejected_during_gen={rejected})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
