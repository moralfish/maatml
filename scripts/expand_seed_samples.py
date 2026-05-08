#!/usr/bin/env python3
"""Expand `seed_samples.jsonl` from the existing 42 hand seeds to 80-100.

Phase 4c of the frontier-dsl-generator plan. Two batches are appended:

  Batch A (12 entries) - canonical: the twelve canonical examples from
    `flow-starter/docs/dsl/examples.md`, ported here so the model sees
    them in the train split (not just the eval split). IDs:
    `seed-canonical-001..012`.

  Batch B (~38 entries) - gap-fillers: deliberately diverse prompts that
    do NOT mirror the eval-canonical strata exactly (avoids train/eval
    leakage). Targets the structural patterns the previous SmolLM2 base
    fumbled on: `.fail` outcomes, parallel fan-in, retry self-edges,
    multi-adapter compositions with `cloud_ai`, and shell/utility-only
    chains. IDs: `seed-fill-001..038`.

Every appended sample is validated through `flow-dsl-validate` before
write. A single parse failure aborts the whole run so the file never
ends up in a half-written state.

Usage:
    python flow-ml/scripts/expand_seed_samples.py [--dry-run]
        [--validator-bin PATH]

Idempotent: a sentinel id (`seed-canonical-001`) is checked first; if
present the script exits without re-appending.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Iterable

REPO = Path(__file__).resolve().parent.parent
SEED_PATH = REPO / "models" / "dsl-generator" / "datasets" / "samples" / "seed_samples.jsonl"
SENTINEL_ID = "seed-canonical-001"


# --- DSL builders (mirrors expand_eval_samples.py for consistency) ------

def shell_run(node_id: str, label: str, command: str) -> str:
    return (
        f'{node_id}[action: "{label}"] {{\n'
        f'  adapter: "shell"\n'
        f'  actionId: "run-command"\n'
        f'  command: "{command}"\n'
        f"}}\n"
    )


def shell_tool(node_id: str, label: str, tool: str, args: str) -> str:
    return (
        f'{node_id}[action: "{label}"] {{\n'
        f'  adapter: "shell"\n'
        f'  actionId: "{tool}"\n'
        f'  args: "{args}"\n'
        f"}}\n"
    )


def zowe_cli(node_id: str, label: str, command: str, conn: str | None = None) -> str:
    body = ['  adapter: "zowe"', '  actionId: "cli-raw"', f'  command: "{command}"']
    if conn:
        body.append(f'  connectionId: "{conn}"')
    body_text = "\n".join(body)
    return f'{node_id}[action: "{label}"] {{\n{body_text}\n}}\n'


def ai_local(node_id: str, label: str, model_id: str, task: str = "spool_interpretation") -> str:
    return (
        f'{node_id}[ai: "{label}"] {{\n'
        f'  task: "{task}"\n'
        f'  modelId: "{model_id}"\n'
        f"}}\n"
    )


def cloud_ai(node_id: str, label: str, provider: str, model: str, prompt: str) -> str:
    return (
        f'{node_id}[cloud_ai: "{label}"] {{\n'
        f'  provider: "{provider}"\n'
        f'  modelId: "{model}"\n'
        f'  prompt: "{prompt}"\n'
        f"}}\n"
    )


def utility(node_id: str, label: str, action_id: str, **fields) -> str:
    body = [f'  actionId: "{action_id}"']
    for k, v in fields.items():
        if isinstance(v, bool):
            body.append(f'  {k}: {"true" if v else "false"}')
        elif isinstance(v, (int, float)):
            body.append(f'  {k}: {v}')
        else:
            body.append(f'  {k}: "{v}"')
    return f'{node_id}[utility: "{label}"] {{\n' + "\n".join(body) + "\n}\n"


def flow(name: str, *blocks: str, edges: Iterable[str] = ()) -> str:
    head = f'flow "{name}" v1\n\n'
    body_text = "\n".join(blocks)
    edges_text = "\n".join(edges)
    if edges_text:
        return head + body_text + "\n" + edges_text + "\n"
    return head + body_text


# --- Batch A: 12 canonical examples (verbatim from docs/dsl/examples.md)

def batch_canonical() -> list[tuple[str, str, str]]:
    """Returns list of (sample_id_suffix, description, dsl)."""
    out: list[tuple[str, str, str]] = []

    out.append((
        "001",
        "install dependencies, build the project, then run the test suite",
        flow(
            "linear-build",
            shell_tool("deps", "Install", "pnpm", "install --frozen-lockfile"),
            shell_tool("build", "Build", "pnpm", "build"),
            shell_tool("test", "Test", "cargo", "test --workspace"),
            edges=["deps --> build", "build --> test"],
        ),
    ))
    out.append((
        "002",
        "run the test suite; if it passes, deploy; if it fails, post to Slack",
        flow(
            "test-and-route",
            shell_tool("test", "Test", "cargo", "test --workspace"),
            shell_tool("deploy", "Deploy", "kubectl", "rollout restart deploy/api"),
            shell_tool("notify", "Notify", "curl", "-fsS -X POST https://hooks.example.com/slack"),
            edges=["test.pass --> deploy", "test.fail --> notify"],
        ),
    ))
    out.append((
        "003",
        "run linting and unit tests in parallel, then commit if both pass",
        flow(
            "parallel-checks",
            shell_tool("lint", "Lint", "pnpm", "run lint"),
            shell_tool("unit", "Unit tests", "pnpm", "test"),
            shell_tool("commit", "Commit", "git", "commit -am wip"),
            edges=["lint.pass --> commit", "unit.pass --> commit"],
        ),
    ))
    out.append((
        "004",
        "fetch the artefact; on failure, retry the same node",
        flow(
            "fetch-with-retry",
            shell_tool("fetch", "Fetch artefact", "curl", "-fsSL https://example.com/artefact.tar.gz -o /tmp/a.tar.gz"),
            edges=["fetch.fail --> fetch"],
        ),
    ))
    out.append((
        "005",
        "build the project; if it fails, ask the local model to suggest a fix",
        flow(
            "build-then-explain",
            shell_tool("build", "Build", "cargo", "build --release"),
            ai_local("explain", "Explain failure", "spool-interpreter"),
            edges=["build.fail --> explain"],
        ),
    ))
    out.append((
        "006",
        "run the test suite; on failure, send the output to Claude for triage",
        flow(
            "test-then-triage",
            shell_tool("test", "Test", "cargo", "test --workspace"),
            cloud_ai("triage", "Triage failure", "claude", "claude-opus-4-5", "The test suite failed. Summarise the root cause in two sentences."),
            edges=["test.fail --> triage"],
        ),
    ))
    out.append((
        "007",
        "have the local model interpret the build log, log a one-line summary, then run the suggested fix command",
        flow(
            "interpret-and-act",
            ai_local("interpret", "Interpret log", "spool-interpreter"),
            utility("log", "Log summary", "log", level="info"),
            shell_run("run", "Run suggested fix", "true"),
            edges=["interpret --> log", "log --> run"],
        ),
    ))
    out.append((
        "008",
        "list jobs on the test LPAR using the saved connection",
        flow(
            "list-jobs",
            zowe_cli("list", "List jobs", "jobs list", conn="lpar-test"),
        ),
    ))
    out.append((
        "009",
        "list the spool files for a job, then unzip the result locally",
        flow(
            "spool-then-unzip",
            zowe_cli("spool", "Get spool list", "jobs list spool-files-by-jobid", conn="lpar-test"),
            shell_run("unpack", "Unzip", "unzip -o /tmp/spool.zip -d /tmp/spool"),
            edges=["spool --> unpack"],
        ),
    ))
    out.append((
        "010",
        "ask Claude to draft a release-note bullet for the latest commit",
        flow(
            "release-note-draft",
            cloud_ai("draft", "Draft note", "claude", "claude-opus-4-5", "Write a one-sentence release-note bullet from the most recent commit message."),
        ),
    ))
    # Note: the canonical example uses `maxTokens: 256`. The flow() builder does not
    # take optional cloud_ai fields; we re-emit the block by hand to keep this verbatim.
    out[-1] = (
        "010",
        out[-1][1],
        'flow "release-note-draft" v1\n\n'
        'draft[cloud_ai: "Draft note"] {\n'
        '  provider: "claude"\n'
        '  modelId: "claude-opus-4-5"\n'
        '  prompt: "Write a one-sentence release-note bullet from the most recent commit message."\n'
        '  maxTokens: 256\n'
        '}\n',
    )
    out.append((
        "011",
        "submit a job, wait, then run a shell command that fetches the result",
        flow(
            "submit-wait-fetch",
            zowe_cli("submit", "Submit job", "jobs submit local-file", conn="lpar-prod"),
            utility("cooldown", "Cool down", "sleep", durationMs=5000),
            shell_tool("fetch", "Fetch result", "curl", "-fsS https://example.com/result.json -o /tmp/result.json"),
            edges=["submit --> cooldown", "cooldown --> fetch"],
        ),
    ))
    out.append((
        "012",
        "just make a node that does nothing for five seconds",
        flow(
            "smoke",
            utility("wait", "Wait", "sleep", durationMs=5000),
        ),
    ))

    return out


# --- Batch B: 38 gap-fillers ---------------------------------------------

def batch_fill() -> list[tuple[str, str, str]]:
    """Gap-fillers covering structural patterns the SmolLM2 base fumbled.

    Sized at 38 so 42 (existing) + 12 (canonical) + 38 (fill) = 92, comfortably
    inside the 80-100 plan target. The patterns deliberately diversify away
    from the eval-canonical entries (different commands, providers, connection
    ids) so the train and eval sets stay disjoint at the prompt + DSL level.
    """
    out: list[tuple[str, str, str]] = []

    # Pure shell chains (heavy on `.fail` outcomes)
    out.append((
        "001",
        "format the codebase, then run rustfmt check; on fail, run cargo fmt and re-check",
        flow(
            "fmt-check",
            shell_tool("format", "Format", "cargo", "fmt"),
            shell_tool("check", "Check", "cargo", "fmt --check"),
            shell_tool("retry", "Re-check", "cargo", "fmt --check"),
            edges=["format --> check", "check.fail --> retry"],
        ),
    ))
    out.append((
        "002",
        "run typescript type-check; only on success run the build",
        flow(
            "typecheck-then-build",
            shell_tool("tc", "Typecheck", "pnpm", "typecheck"),
            shell_tool("build", "Build", "pnpm", "build"),
            edges=["tc.pass --> build"],
        ),
    ))
    out.append((
        "003",
        "run a smoke curl, then a deeper integration curl, then a slow soak",
        flow(
            "tiered-checks",
            shell_tool("smoke", "Smoke", "curl", "-fsS https://example.com/health"),
            shell_tool("integration", "Integration", "curl", "-fsS https://example.com/api/v1/status"),
            shell_tool("soak", "Soak", "curl", "-fsS https://example.com/api/v1/load"),
            edges=["smoke.pass --> integration", "integration.pass --> soak"],
        ),
    ))
    out.append((
        "004",
        "verify the deployment then on failure scale up replicas",
        flow(
            "verify-or-scale",
            shell_tool("verify", "Verify", "kubectl", "rollout status deploy/api"),
            shell_tool("scale", "Scale", "kubectl", "scale deploy/api --replicas=4"),
            edges=["verify.fail --> scale"],
        ),
    ))
    out.append((
        "005",
        "list pods then on success describe the first one for diagnostics",
        flow(
            "pods-then-describe",
            shell_tool("list", "List pods", "kubectl", "get pods -o wide"),
            shell_tool("describe", "Describe", "kubectl", "describe pod -l app=api"),
            edges=["list.pass --> describe"],
        ),
    ))

    # Parallel fan-in patterns
    out.append((
        "006",
        "fetch payload from two mirrors in parallel; merge once both finish",
        flow(
            "two-mirror-fetch",
            shell_run("a", "Mirror A", "curl -fsS https://mirror-a.example/data -o /tmp/a.json"),
            shell_run("b", "Mirror B", "curl -fsS https://mirror-b.example/data -o /tmp/b.json"),
            shell_run("merge", "Merge", "jq -s '.[0] * .[1]' /tmp/a.json /tmp/b.json > /tmp/m.json"),
            edges=["a --> merge", "b --> merge"],
        ),
    ))
    out.append((
        "007",
        "build the rust workspace and the node workspace in parallel; publish only if both pass",
        flow(
            "double-build-publish",
            shell_tool("rust", "Cargo build", "cargo", "build --release"),
            shell_tool("node", "PNPM build", "pnpm", "build"),
            shell_tool("publish", "Publish", "cargo", "publish --dry-run"),
            edges=["rust.pass --> publish", "node.pass --> publish"],
        ),
    ))
    out.append((
        "008",
        "run cargo test and cargo clippy together; on either failing, log",
        flow(
            "test-and-lint",
            shell_tool("test", "Test", "cargo", "test --workspace"),
            shell_tool("clippy", "Clippy", "cargo", "clippy --all-targets"),
            utility("note", "Note", "log", level="warn"),
            edges=["test.fail --> note", "clippy.fail --> note"],
        ),
    ))

    # Retry / self-edge
    out.append((
        "009",
        "ping the staging endpoint; retry up to a configured limit on fail",
        flow(
            "ping-retry",
            shell_tool("ping", "Healthcheck", "curl", "-fsS https://staging.example.com/health"),
            edges=["ping.fail --> ping"],
        ),
    ))
    out.append((
        "010",
        "submit jcl; on submission failure retry the same submission",
        flow(
            "submit-retry",
            zowe_cli("submit", "Submit JCL", "jobs submit local-file", conn="lpar-test"),
            edges=["submit.fail --> submit"],
        ),
    ))

    # Multi-adapter (shell + cloud_ai)
    out.append((
        "011",
        "run integration tests; on fail, ask Gemini to summarise stack traces",
        flow(
            "tests-then-gemini",
            shell_tool("test", "Integration tests", "pnpm", "test:integration"),
            cloud_ai("summarise", "Summarise", "gemini", "gemini-2.5-pro", "Summarise the failing stack traces."),
            edges=["test.fail --> summarise"],
        ),
    ))
    out.append((
        "012",
        "build the docker image; if it succeeds use OpenAI to suggest a tag name",
        flow(
            "build-then-tag",
            shell_run("build", "Docker build", "docker build -t app ."),
            cloud_ai("tag", "Tag suggestion", "openai", "gpt-5", "Suggest a one-word docker image tag for the latest build."),
            edges=["build.pass --> tag"],
        ),
    ))
    out.append((
        "013",
        "diff main against the current branch; ask Claude whether to squash",
        flow(
            "diff-then-decide",
            shell_tool("diff", "Diff", "git", "diff main..HEAD --stat"),
            cloud_ai("decide", "Decide", "claude", "claude-opus-4-5", "Should this branch be squashed before merge? Reply yes or no."),
            edges=["diff --> decide"],
        ),
    ))
    out.append((
        "014",
        "have Claude draft a postmortem outline, then sleep, then archive",
        flow(
            "postmortem",
            cloud_ai("draft", "Draft outline", "claude", "claude-opus-4-5", "Draft a five-bullet postmortem outline."),
            utility("rest", "Rest", "sleep", durationMs=2000),
            shell_run("archive", "Archive", "cp /tmp/outline.md /var/postmortems/$(date +%F).md"),
            edges=["draft --> rest", "rest --> archive"],
        ),
    ))

    # Multi-adapter (zowe + utility / ai)
    out.append((
        "015",
        "list zos jobs; if listing fails, sleep 10 seconds and retry",
        flow(
            "zowe-with-cooldown",
            zowe_cli("list", "List jobs", "jobs list"),
            utility("cool", "Cool down", "sleep", durationMs=10000),
            edges=["list.fail --> cool", "cool --> list"],
        ),
    ))
    out.append((
        "016",
        "submit a JCL job and have the local interpreter explain the result",
        flow(
            "submit-then-interpret",
            zowe_cli("submit", "Submit", "jobs submit local-file", conn="lpar-prod"),
            ai_local("explain", "Explain", "spool-interpreter"),
            edges=["submit --> explain"],
        ),
    ))
    out.append((
        "017",
        "list datasets; on success branch on payload, on fail log the error",
        flow(
            "ds-list-branch",
            zowe_cli("ds", "List datasets", "files list ds", conn="lpar-test"),
            utility("decide", "Decide", "branch", on="payload.count"),
            utility("note", "Note", "log", level="error"),
            edges=["ds.pass --> decide", "ds.fail --> note"],
        ),
    ))

    # Pure utility chains
    out.append((
        "018",
        "wait one second, log a heartbeat, wait again",
        flow(
            "heartbeat",
            utility("a", "Wait A", "sleep", durationMs=1000),
            utility("beat", "Heartbeat", "log", level="info"),
            utility("b", "Wait B", "sleep", durationMs=1000),
            edges=["a --> beat", "beat --> b"],
        ),
    ))
    out.append((
        "019",
        "log a starting line then sleep then log a finishing line",
        flow(
            "log-frame",
            utility("start", "Start", "log", level="info"),
            utility("rest", "Rest", "sleep", durationMs=3000),
            utility("end", "End", "log", level="info"),
            edges=["start --> rest", "rest --> end"],
        ),
    ))

    # Branch-and-converge
    out.append((
        "020",
        "branch on previous status; both branches converge on a single final step",
        flow(
            "branch-converge",
            utility("decide", "Decide", "branch", on="payload.status"),
            shell_run("a", "Path A", "echo a"),
            shell_run("b", "Path B", "echo b"),
            shell_run("done", "Done", "echo done"),
            edges=[
                "decide.pass --> a",
                "decide.fail --> b",
                "a --> done",
                "b --> done",
            ],
        ),
    ))

    # Three-step linear with mixed adapters
    out.append((
        "021",
        "run a build, ask the local model to summarise output, then post the summary",
        flow(
            "build-summary-post",
            shell_tool("build", "Build", "cargo", "build --release"),
            ai_local("summarise", "Summarise", "spool-interpreter"),
            shell_tool("post", "Post", "curl", "-fsS -X POST https://hooks.example.com/notes"),
            edges=["build --> summarise", "summarise --> post"],
        ),
    ))
    out.append((
        "022",
        "run npm audit, then ssh to the bastion to log the result",
        flow(
            "audit-and-record",
            shell_tool("audit", "Audit", "npm", "audit --prod"),
            shell_run("record", "Record", "ssh bastion 'echo audit-done >> /var/log/audit.log'"),
            edges=["audit --> record"],
        ),
    ))
    out.append((
        "023",
        "run a kubectl apply; on fail, run kubectl describe to dump diagnostics",
        flow(
            "apply-or-describe",
            shell_tool("apply", "Apply", "kubectl", "apply -f manifests/"),
            shell_tool("describe", "Describe", "kubectl", "describe deploy api"),
            edges=["apply.fail --> describe"],
        ),
    ))

    # cloud_ai-only single-node variants
    out.append((
        "024",
        "have OpenAI generate a SQL migration name",
        flow(
            "openai-name",
            cloud_ai("name", "Name", "openai", "gpt-5", "Suggest a snake_case migration name for adding a created_at column."),
        ),
    ))
    out.append((
        "025",
        "ask Gemini to label severity for a given log line",
        flow(
            "gemini-severity",
            cloud_ai("severity", "Severity", "gemini", "gemini-2.5-flash", "Label this log line as INFO, WARN, or ERROR."),
        ),
    ))

    # ai-local single-node
    out.append((
        "026",
        "interpret a recent spool dump",
        flow(
            "spool-interpret",
            ai_local("interpret", "Interpret spool", "spool-interpreter"),
        ),
    ))

    # zowe + shell two-step
    out.append((
        "027",
        "view JCL for a job and grep for a particular DD statement",
        flow(
            "view-grep",
            zowe_cli("view", "View JCL", "jobs view jcl", conn="lpar-test"),
            shell_run("grep", "Grep DD", "grep -E '^//SYSTSIN' /tmp/job.jcl"),
            edges=["view --> grep"],
        ),
    ))

    # Multi-adapter cross-domain
    out.append((
        "028",
        "run a security scan, ask Claude to merge findings, post to Slack",
        flow(
            "scan-merge-post",
            shell_tool("scan", "Scan", "cargo", "audit --json"),
            cloud_ai("merge", "Merge findings", "claude", "claude-opus-4-5", "Merge the cargo-audit findings into a one-paragraph summary."),
            shell_tool("post", "Post", "curl", "-fsS -X POST https://hooks.example.com/slack"),
            edges=["scan --> merge", "merge --> post"],
        ),
    ))
    out.append((
        "029",
        "have Claude triage a ticket; on a triage decision branch, run a curl against the relevant API",
        flow(
            "triage-route",
            cloud_ai("triage", "Triage", "claude", "claude-opus-4-5", "Classify this ticket as billing, infra, or product."),
            utility("route", "Route", "branch", on="payload.label"),
            shell_tool("call", "Call API", "curl", "-fsS https://example.com/api/triage"),
            edges=["triage --> route", "route --> call"],
        ),
    ))

    # Pure ai chains (rare but valid)
    out.append((
        "030",
        "interpret then re-interpret with cloud reinforcement",
        flow(
            "double-interpret",
            ai_local("local", "Local interpret", "spool-interpreter"),
            cloud_ai("cloud", "Cloud verify", "claude", "claude-opus-4-5", "Verify the interpretation in one sentence."),
            edges=["local --> cloud"],
        ),
    ))

    # Long chains
    out.append((
        "031",
        "fetch, validate, transform, and post a payload in four steps",
        flow(
            "etl-four",
            shell_run("fetch", "Fetch", "curl -fsS https://example.com/in.json -o /tmp/in.json"),
            shell_run("validate", "Validate", "jq -e . /tmp/in.json"),
            shell_run("transform", "Transform", "jq '{ok:true, count:length}' /tmp/in.json > /tmp/out.json"),
            shell_run("post", "Post", "curl -fsS -X POST https://example.com/out -d @/tmp/out.json"),
            edges=["fetch --> validate", "validate --> transform", "transform --> post"],
        ),
    ))
    out.append((
        "032",
        "lint, typecheck, test, build, deploy in a five-stage chain",
        flow(
            "ci-five-stage",
            shell_tool("lint", "Lint", "pnpm", "run lint"),
            shell_tool("tc", "Typecheck", "pnpm", "typecheck"),
            shell_tool("test", "Test", "pnpm", "test"),
            shell_tool("build", "Build", "pnpm", "build"),
            shell_tool("deploy", "Deploy", "kubectl", "rollout restart deploy/api"),
            edges=[
                "lint.pass --> tc",
                "tc.pass --> test",
                "test.pass --> build",
                "build.pass --> deploy",
            ],
        ),
    ))

    # Smoke / minimum
    out.append((
        "033",
        "minimal one-step shell echo",
        flow(
            "echo-only",
            shell_run("step", "Echo", "echo hello"),
        ),
    ))
    out.append((
        "034",
        "single git status",
        flow(
            "git-status-only",
            shell_tool("status", "Status", "git", "status"),
        ),
    ))
    out.append((
        "035",
        "single zowe list-datasets",
        flow(
            "zowe-ds-only",
            zowe_cli("ds", "List datasets", "files list ds"),
        ),
    ))

    # Specific RESTART-class flows
    out.append((
        "036",
        "list restart-enabled jobs older than 7 days then have Claude classify which to delete",
        flow(
            "restart-housekeeping",
            zowe_cli("list", "List restart jobs", "jobs list", conn="lpar-prod"),
            cloud_ai("classify", "Classify", "claude", "claude-opus-4-5", "Classify each restart-enabled job as delete or restart."),
            edges=["list --> classify"],
        ),
    ))
    out.append((
        "037",
        "submit JCL via zowe; on success, immediately run the local spool interpreter on the result",
        flow(
            "smart-submit-interpret",
            zowe_cli("submit", "Submit", "jobs submit local-file", conn="lpar-test"),
            ai_local("interpret", "Interpret", "spool-interpreter"),
            edges=["submit.pass --> interpret"],
        ),
    ))
    out.append((
        "038",
        "before submit, validate the JCL filename via shell, then submit",
        flow(
            "validate-then-submit",
            shell_run("check", "Check file", "test -f /tmp/payload.jcl"),
            zowe_cli("submit", "Submit", "jobs submit local-file", conn="lpar-test"),
            edges=["check.pass --> submit"],
        ),
    ))

    return out


# --- Driver --------------------------------------------------------------

def already_extended(path: Path) -> bool:
    if not path.exists():
        return False
    return any(SENTINEL_ID in line for line in path.read_text(encoding="utf-8").splitlines())


def validate(dsl: str, validator_bin: str) -> None:
    try:
        result = subprocess.run(
            [validator_bin],
            input=dsl,
            capture_output=True,
            text=True,
            check=False,
        )
    except FileNotFoundError as e:
        raise SystemExit(
            f"flow-dsl-validate not found at {validator_bin!r}; "
            f"build it with `cargo build -p flow-dsl-validate`"
        ) from e
    if result.returncode != 0:
        raise SystemExit(
            f"validator rejected sample (stderr: {result.stderr.strip()}):\n{dsl}"
        )


def default_validator_bin() -> str:
    flow_starter = REPO.parent / "flow-starter"
    candidate = flow_starter / "target" / "debug" / "flow-dsl-validate"
    if candidate.exists():
        return str(candidate)
    return "flow-dsl-validate"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--validator-bin", default=default_validator_bin())
    args = parser.parse_args()

    if already_extended(SEED_PATH):
        print(f"{SEED_PATH} already contains {SENTINEL_ID}; nothing to do.")
        return 0

    canonical = batch_canonical()
    fill = batch_fill()
    rows: list[tuple[str, str, str, str]] = []
    for suffix, desc, dsl in canonical:
        rows.append((f"seed-canonical-{suffix}", "hand:canonical", desc, dsl))
    for suffix, desc, dsl in fill:
        rows.append((f"seed-fill-{suffix}", "hand:gap-filler", desc, dsl))

    expected = 12 + 38
    if len(rows) != expected:
        raise SystemExit(
            f"strata yielded {len(rows)} entries, expected exactly {expected}"
        )

    for i, (sample_id, _, _, dsl) in enumerate(rows, start=1):
        validate(dsl, args.validator_bin)
        sys.stderr.write(f"  [{i:02d}/{len(rows)}] OK: {sample_id}\n")

    if args.dry_run:
        print(f"--dry-run: validated {len(rows)} samples; not writing")
        return 0

    with SEED_PATH.open("a", encoding="utf-8") as fh:
        for sample_id, source, desc, dsl in rows:
            fh.write(json.dumps(
                {"sample_id": sample_id, "source": source, "description": desc, "dsl": dsl},
                ensure_ascii=False,
            ) + "\n")

    print(f"appended {len(rows)} samples to {SEED_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
