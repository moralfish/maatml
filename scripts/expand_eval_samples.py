#!/usr/bin/env python3
"""Expand `eval_samples.jsonl` from the existing 50 entries to a stratified 100.

Appends 50 new {sample_id, source, description, dsl} rows covering the
deliberate-coverage strata documented in the plan:

  15 single-node flows (one per adapter+actionId combo)
  10 two-or-three-node linear chains
  10 branching flows (pass/fail / fan-in)
   8 multi-adapter compositions (shell+ai, shell+cloud_ai, zowe+utility)
   3 conditional / retry / loop patterns
   4 minimal utility smoke nodes

Total: 50 new rows + 50 existing = 100.

Every appended `dsl` field is validated via `flow-dsl-validate` before
the row is written. A single parse failure aborts the whole run with a
non-zero exit so a partial / inconsistent file is never produced.

Usage:
    python flow-ml/scripts/expand_eval_samples.py [--dry-run]
        [--validator-bin PATH]

Idempotent: if the file already contains a `eval-canonical-001` entry
(the first id this script writes), the script exits cleanly without
re-appending.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Iterable

REPO = Path(__file__).resolve().parent.parent
EVAL_PATH = REPO / "models" / "dsl-generator" / "datasets" / "samples" / "eval_samples.jsonl"
SOURCE = "hand:eval-canonical"
SENTINEL_ID = "eval-canonical-001"


# --- DSL builders --------------------------------------------------------
#
# Each builder returns a (description, dsl_text) pair. The dsl_text always
# ends with a newline so concatenation in the JSONL stays clean. We avoid
# nested double-quotes inside string fields - the lexer accepts \" but the
# spec recommends staying simple, and the dsl-generator follows the spec.

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
    body_text = "\n".join(body)
    return f'{node_id}[utility: "{label}"] {{\n{body_text}\n}}\n'


def flow(name: str, *blocks: str, edges: Iterable[str] = ()) -> str:
    head = f'flow "{name}" v1\n\n'
    body_text = "\n".join(blocks)
    edges_text = "\n".join(edges)
    if edges_text:
        return head + body_text + "\n" + edges_text + "\n"
    return head + body_text


# --- Sample assembly -----------------------------------------------------

def stratum_single_node() -> list[tuple[str, str]]:
    """15 single-node flows; one per adapter+actionId combo we currently
    expose. Floor-level competence check."""
    out: list[tuple[str, str]] = []

    # 7 shell actions
    out.append(("install dependencies", flow("install", shell_tool("deps", "Install", "pnpm", "install --frozen-lockfile"))))
    out.append(("run a one-off shell command", flow("run", shell_run("step", "Echo hello", "echo hello"))))
    out.append(("show git status", flow("git-status", shell_tool("status", "Status", "git", "status --short"))))
    out.append(("install npm packages", flow("npm-install", shell_tool("deps", "Install", "npm", "ci"))))
    out.append(("run cargo tests", flow("cargo-test", shell_tool("test", "Test", "cargo", "test --workspace"))))
    out.append(("check kubernetes rollout", flow("kube-status", shell_tool("rollout", "Rollout status", "kubectl", "rollout status deploy/api"))))
    out.append(("ping a healthcheck endpoint", flow("ping", shell_tool("health", "Healthcheck", "curl", "-fsS https://example.com/health"))))

    # zowe
    out.append(("list jobs on the LPAR", flow("zowe-list", zowe_cli("list", "List jobs", "jobs list"))))

    # local ai
    out.append(("interpret a build log via the local model", flow("interpret", ai_local("explain", "Explain log", "spool-interpreter"))))

    # cloud_ai with three providers
    out.append(("ask Claude to summarise a commit", flow("claude-summary", cloud_ai("draft", "Draft", "claude", "claude-opus-4-5", "Summarise the most recent commit in one sentence."))))
    out.append(("use OpenAI to draft a release note", flow("openai-note", cloud_ai("draft", "Draft note", "openai", "gpt-5", "Draft a release-note bullet."))))
    out.append(("use Gemini to label a support ticket", flow("gemini-label", cloud_ai("classify", "Classify ticket", "gemini", "gemini-2.5-pro", "Label the ticket in one word."))))

    # utility
    out.append(("sleep for five seconds", flow("smoke-sleep", utility("wait", "Wait", "sleep", durationMs=5000))))
    out.append(("log an info line", flow("smoke-log", utility("log", "Log", "log", level="info", message="hello"))))
    out.append(("branch on the previous payload", flow("smoke-branch", utility("decide", "Decide", "branch", on="payload.status"))))

    return out


def stratum_linear() -> list[tuple[str, str]]:
    """10 two-or-three-node linear chains."""
    out: list[tuple[str, str]] = []

    out.append((
        "install dependencies then build",
        flow(
            "deps-build",
            shell_tool("deps", "Install", "pnpm", "install --frozen-lockfile"),
            shell_tool("build", "Build", "pnpm", "build"),
            edges=["deps --> build"],
        ),
    ))
    out.append((
        "build then test then deploy",
        flow(
            "build-test-deploy",
            shell_tool("build", "Build", "cargo", "build --release"),
            shell_tool("test", "Test", "cargo", "test --workspace"),
            shell_tool("deploy", "Deploy", "kubectl", "rollout restart deploy/api"),
            edges=["build --> test", "test --> deploy"],
        ),
    ))
    out.append((
        "fetch artefact then unpack",
        flow(
            "fetch-unpack",
            shell_run("fetch", "Fetch", "curl -fsSL https://example.com/a.tar.gz -o /tmp/a.tar.gz"),
            shell_run("unpack", "Unpack", "tar -xzf /tmp/a.tar.gz -C /tmp/"),
            edges=["fetch --> unpack"],
        ),
    ))
    out.append((
        "list zowe jobs then export the result",
        flow(
            "zowe-list-export",
            zowe_cli("list", "List jobs", "jobs list"),
            shell_run("export", "Export", "cp /tmp/jobs.json /tmp/jobs-archive.json"),
            edges=["list --> export"],
        ),
    ))
    out.append((
        "ask Claude for a name then echo it",
        flow(
            "claude-then-echo",
            cloud_ai("name", "Name", "claude", "claude-opus-4-5", "Pick a one-word codename."),
            shell_run("echo", "Echo", "echo done"),
            edges=["name --> echo"],
        ),
    ))
    out.append((
        "interpret spool then log a summary",
        flow(
            "interpret-then-log",
            ai_local("interpret", "Interpret", "spool-interpreter"),
            utility("note", "Note", "log", level="info"),
            edges=["interpret --> note"],
        ),
    ))
    out.append((
        "wait then run a healthcheck",
        flow(
            "wait-then-ping",
            utility("wait", "Wait", "sleep", durationMs=10000),
            shell_tool("ping", "Healthcheck", "curl", "-fsS https://example.com/health"),
            edges=["wait --> ping"],
        ),
    ))
    out.append((
        "submit job then sleep then fetch result",
        flow(
            "submit-wait-fetch",
            zowe_cli("submit", "Submit", "jobs submit local-file", conn="lpar-test"),
            utility("cool", "Cool down", "sleep", durationMs=5000),
            shell_tool("fetch", "Fetch", "curl", "-fsS https://example.com/r.json -o /tmp/r.json"),
            edges=["submit --> cool", "cool --> fetch"],
        ),
    ))
    out.append((
        "run lints then unit tests",
        flow(
            "lints-then-tests",
            shell_tool("lint", "Lint", "pnpm", "run lint"),
            shell_tool("unit", "Unit", "pnpm", "test"),
            edges=["lint --> unit"],
        ),
    ))
    out.append((
        "git status then commit",
        flow(
            "status-then-commit",
            shell_tool("status", "Status", "git", "status --short"),
            shell_tool("commit", "Commit", "git", "commit -am wip"),
            edges=["status --> commit"],
        ),
    ))

    return out


def stratum_branching() -> list[tuple[str, str]]:
    """10 branching flows (pass/fail outcomes, fan-out, fan-in)."""
    out: list[tuple[str, str]] = []

    out.append((
        "test then deploy on pass or notify on fail",
        flow(
            "test-route",
            shell_tool("test", "Test", "cargo", "test --workspace"),
            shell_tool("deploy", "Deploy", "kubectl", "rollout restart deploy/api"),
            shell_tool("notify", "Notify", "curl", "-fsS -X POST https://hooks.example.com/slack"),
            edges=["test.pass --> deploy", "test.fail --> notify"],
        ),
    ))
    out.append((
        "build then deploy or rollback",
        flow(
            "build-deploy-rollback",
            shell_tool("build", "Build", "cargo", "build --release"),
            shell_tool("deploy", "Deploy", "kubectl", "rollout restart deploy/api"),
            shell_tool("rollback", "Rollback", "kubectl", "rollout undo deploy/api"),
            edges=["build.pass --> deploy", "build.fail --> rollback"],
        ),
    ))
    out.append((
        "lint and unit in parallel then commit on both pass",
        flow(
            "parallel-lint-unit",
            shell_tool("lint", "Lint", "pnpm", "run lint"),
            shell_tool("unit", "Unit tests", "pnpm", "test"),
            shell_tool("commit", "Commit", "git", "commit -am wip"),
            edges=["lint.pass --> commit", "unit.pass --> commit"],
        ),
    ))
    out.append((
        "build and test in parallel then publish",
        flow(
            "build-test-publish",
            shell_tool("build", "Build", "cargo", "build --release"),
            shell_tool("test", "Test", "cargo", "test --workspace"),
            shell_tool("publish", "Publish", "cargo", "publish --dry-run"),
            edges=["build.pass --> publish", "test.pass --> publish"],
        ),
    ))
    out.append((
        "submit job; on fail interpret the spool",
        flow(
            "submit-then-interpret",
            zowe_cli("submit", "Submit", "jobs submit local-file", conn="lpar-prod"),
            ai_local("interpret", "Interpret", "spool-interpreter"),
            edges=["submit.fail --> interpret"],
        ),
    ))
    out.append((
        "deploy then verify; rollback on verify fail",
        flow(
            "verify-or-rollback",
            shell_tool("deploy", "Deploy", "kubectl", "rollout restart deploy/api"),
            shell_tool("verify", "Verify", "curl", "-fsS https://example.com/health"),
            shell_tool("rollback", "Rollback", "kubectl", "rollout undo deploy/api"),
            edges=["deploy --> verify", "verify.fail --> rollback"],
        ),
    ))
    out.append((
        "run cargo audit and pnpm audit and merge with Claude",
        flow(
            "security-audit",
            shell_tool("rust", "Rust audit", "cargo", "audit"),
            shell_tool("node", "Node audit", "pnpm", "audit --prod"),
            cloud_ai("merge", "Merge findings", "claude", "claude-opus-4-5", "Merge these advisories into a single CVE list."),
            edges=["rust --> merge", "node --> merge"],
        ),
    ))
    out.append((
        "two-stage zowe then a final shell",
        flow(
            "zowe-fanin",
            zowe_cli("a", "Probe A", "jobs list"),
            zowe_cli("b", "Probe B", "files list ds"),
            shell_run("merge", "Merge", "cat /tmp/a.json /tmp/b.json > /tmp/c.json"),
            edges=["a --> merge", "b --> merge"],
        ),
    ))
    out.append((
        "test; on fail ask Claude for a triage",
        flow(
            "test-triage",
            shell_tool("test", "Test", "cargo", "test --workspace"),
            cloud_ai("triage", "Triage", "claude", "claude-opus-4-5", "Summarise the failure in two sentences."),
            edges=["test.fail --> triage"],
        ),
    ))
    out.append((
        "build; if pass deploy; if fail also notify",
        flow(
            "always-and-fail",
            shell_tool("build", "Build", "pnpm", "build"),
            shell_tool("deploy", "Deploy", "kubectl", "rollout restart deploy/api"),
            shell_tool("notify", "Notify", "curl", "-fsS -X POST https://hooks.example.com/slack"),
            edges=["build.pass --> deploy", "build.fail --> notify"],
        ),
    ))

    return out


def stratum_multi_adapter() -> list[tuple[str, str]]:
    """8 multi-adapter compositions."""
    out: list[tuple[str, str]] = []

    out.append((
        "shell build then cloud_ai release notes",
        flow(
            "build-then-notes",
            shell_tool("build", "Build", "pnpm", "build"),
            cloud_ai("notes", "Notes", "claude", "claude-opus-4-5", "Draft a release note."),
            edges=["build.pass --> notes"],
        ),
    ))
    out.append((
        "zowe submit then ai interpret",
        flow(
            "submit-interpret",
            zowe_cli("submit", "Submit", "jobs submit local-file", conn="lpar-test"),
            ai_local("interpret", "Interpret", "spool-interpreter"),
            edges=["submit.fail --> interpret"],
        ),
    ))
    out.append((
        "shell test then utility log then cloud_ai triage",
        flow(
            "test-log-triage",
            shell_tool("test", "Test", "cargo", "test --workspace"),
            utility("log", "Log", "log", level="warn"),
            cloud_ai("triage", "Triage", "claude", "claude-opus-4-5", "Summarise the test failure."),
            edges=["test.fail --> log", "log --> triage"],
        ),
    ))
    out.append((
        "zowe list then shell archive then utility sleep",
        flow(
            "zowe-archive-cool",
            zowe_cli("list", "List jobs", "jobs list"),
            shell_run("archive", "Archive", "tar -czf /tmp/jobs.tgz /tmp/jobs.json"),
            utility("cool", "Cool", "sleep", durationMs=2000),
            edges=["list --> archive", "archive --> cool"],
        ),
    ))
    out.append((
        "ai interpret then shell run on pass",
        flow(
            "interpret-then-fix",
            ai_local("interpret", "Interpret", "spool-interpreter"),
            shell_run("fix", "Apply suggested fix", "true"),
            edges=["interpret.pass --> fix"],
        ),
    ))
    out.append((
        "cloud_ai then utility log then shell exit",
        flow(
            "cloud-log-exit",
            cloud_ai("draft", "Draft", "openai", "gpt-5", "Pick a yes or no."),
            utility("note", "Note", "log", level="info"),
            shell_run("done", "Done", "echo done"),
            edges=["draft --> note", "note --> done"],
        ),
    ))
    out.append((
        "shell + zowe + cloud_ai pipeline",
        flow(
            "tri-adapter",
            shell_tool("ping", "Healthcheck", "curl", "-fsS https://example.com/health"),
            zowe_cli("list", "List jobs", "jobs list", conn="lpar-test"),
            cloud_ai("summary", "Summary", "claude", "claude-opus-4-5", "Summarise both outputs."),
            edges=["ping.pass --> list", "list.pass --> summary"],
        ),
    ))
    out.append((
        "shell + ai + utility three-node fork",
        flow(
            "fork-shell-ai-util",
            shell_run("seed", "Seed", "echo seed"),
            ai_local("explain", "Explain", "spool-interpreter"),
            utility("done", "Done", "log", level="info"),
            edges=["seed --> explain", "seed --> done"],
        ),
    ))

    return out


def stratum_conditional() -> list[tuple[str, str]]:
    """3 conditional / retry / loop patterns."""
    out: list[tuple[str, str]] = []

    out.append((
        "fetch with self-edge retry",
        flow(
            "retry-fetch",
            shell_tool("fetch", "Fetch", "curl", "-fsSL https://example.com/a.tar.gz"),
            edges=["fetch.fail --> fetch"],
        ),
    ))
    out.append((
        "deploy then verify; on verify fail rollback then alert",
        flow(
            "deploy-verify-rollback-alert",
            shell_tool("deploy", "Deploy", "kubectl", "rollout restart deploy/api"),
            shell_tool("verify", "Verify", "curl", "-fsS https://example.com/health"),
            shell_tool("rollback", "Rollback", "kubectl", "rollout undo deploy/api"),
            shell_tool("alert", "Alert", "curl", "-fsS -X POST https://hooks.example.com/pagerduty"),
            edges=[
                "deploy --> verify",
                "verify.fail --> rollback",
                "rollback --> alert",
            ],
        ),
    ))
    out.append((
        "branch on payload then converge",
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

    return out


def stratum_smoke() -> list[tuple[str, str]]:
    """4 minimal one-utility-node smoke nodes (anti-overcomplication probe)."""
    out: list[tuple[str, str]] = []

    out.append(("just sleep five seconds", flow("smoke", utility("wait", "Wait", "sleep", durationMs=5000))))
    out.append(("log a single line and stop", flow("smoke-log-only", utility("log", "Log", "log", level="info", message="ok"))))
    out.append(("a placeholder that does nothing", flow("smoke-noop", utility("noop", "No-op", "noop"))))
    out.append(("wait one second", flow("smoke-1s", utility("wait", "Wait", "sleep", durationMs=1000))))

    return out


# --- Driver --------------------------------------------------------------

def collect_new() -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    out.extend(stratum_single_node())
    out.extend(stratum_linear())
    out.extend(stratum_branching())
    out.extend(stratum_multi_adapter())
    out.extend(stratum_conditional())
    out.extend(stratum_smoke())
    expected = 50
    if len(out) != expected:
        raise SystemExit(
            f"strata yielded {len(out)} entries, expected exactly {expected}"
        )
    return out


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
    # Prefer the workspace target dir so a developer who has already run
    # `cargo build` does not need to re-build under a different profile.
    flow_starter = REPO.parent / "flow-studio"
    candidate = flow_starter / "target" / "debug" / "flow-dsl-validate"
    if candidate.exists():
        return str(candidate)
    # Fall back to expecting it on PATH.
    return "flow-dsl-validate"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--validator-bin", default=default_validator_bin())
    args = parser.parse_args()

    if already_extended(EVAL_PATH):
        print(f"{EVAL_PATH} already contains {SENTINEL_ID}; nothing to do.")
        return 0

    new_rows = collect_new()

    # Validate every sample before touching the file. One bad sample
    # aborts the whole run.
    for i, (desc, dsl) in enumerate(new_rows, start=1):
        validate(dsl, args.validator_bin)
        sys.stderr.write(f"  [{i:02d}/50] OK: {desc[:60]}\n")

    if args.dry_run:
        print(f"--dry-run: validated {len(new_rows)} samples; not writing")
        return 0

    # Append in canonical JSON-line form. We keep newlines escaped inside
    # the dsl string field; json.dumps does that correctly.
    with EVAL_PATH.open("a", encoding="utf-8") as fh:
        for i, (description, dsl) in enumerate(new_rows, start=1):
            row = {
                "sample_id": f"eval-canonical-{i:03d}",
                "source": SOURCE,
                "description": description,
                "dsl": dsl,
            }
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"appended {len(new_rows)} canonical eval samples to {EVAL_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
