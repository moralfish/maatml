"""Build the Support Ticket Triage seed corpus deterministically.

Renders parametric ticket templates across the five categories and derives the
gold triage from the routing contract, so every label is correct by
construction. Each sample is gated by the triage validator before it is written.

No API calls. Reproducible given the seed.

Usage:
    python examples/support-ticket-triage/scripts/build_seeds.py
    python examples/support-ticket-triage/scripts/build_seeds.py --target 800
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

EXAMPLE_ROOT = Path(__file__).resolve().parents[1]
REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(EXAMPLE_ROOT))

from maatml.utils.io import stable_hash  # noqa: E402
from triage_plugin.constants import MAX_SUMMARY_WORDS, ROUTING  # noqa: E402
from triage_plugin.validator import validate_triage  # noqa: E402

DATASETS = EXAMPLE_ROOT / "datasets"
SCHEMA_PATH = DATASETS / "schema.json"
SEEDS_PATH = DATASETS / "samples" / "seed_samples.jsonl"
BENCH_PATH = DATASETS / "samples" / "test_prompt_set.jsonl"

# (subject, body, summary, priority) per category. Braced slots come from POOLS.
TEMPLATES: dict[str, list[tuple[str, str, str, str]]] = {
    "billing": [
        ("Charged twice for {month}",
         "We were billed {amount} twice on the same day for our {plan} plan. "
         "Invoice {invoice} shows both charges. Please refund the duplicate.",
         "Duplicate {plan} charge in {month}; refund requested", "p2"),
        ("Card declined at renewal",
         "Our company card was declined at renewal and the workspace is locked. "
         "We updated the card but {product} still will not open.",
         "Renewal decline locked workspace despite updated card", "p1"),
        ("Invoice {invoice} does not match usage",
         "Invoice {invoice} bills {amount} but our usage report for {month} "
         "shows far less. Can someone reconcile this before we pay?",
         "Invoice {invoice} disputed against {month} usage", "p3"),
        ("Need to change billing cycle",
         "We are on monthly billing for the {plan} plan and finance wants to "
         "move to annual starting {month}. What is the process?",
         "Requests switch to annual billing on {plan} plan", "p4"),
    ],
    "access": [
        ("Cannot sign in after SSO change",
         "Since IT rotated our SSO certificate nobody on the {team_name} team "
         "can sign in to {product}. We get an assertion error every time.",
         "SSO certificate rotation blocks {team_name} team sign-in", "p1"),
        ("Locked out after too many attempts",
         "I mistyped my password and now my account is locked. I need access "
         "to {product} before the {month} review.",
         "Account locked after failed attempts; needs unlock", "p2"),
        ("Add {count} seats to the workspace",
         "We hired {count} people starting {month} and need them added to the "
         "{team_name} workspace with the standard role.",
         "Requests {count} new seats for {team_name} workspace", "p3"),
        ("Remove access for a departing colleague",
         "A member of the {team_name} team left on {month}. Please revoke "
         "their access to {product} and transfer their projects.",
         "Offboarding: revoke {product} access for departed member", "p3"),
    ],
    "bug": [
        ("{product} returns 500 on export",
         "Exporting any report from {product} returns a 500. It started after "
         "the {month} release and reproduces on every browser we tried.",
         "{product} export returns 500 since {month} release", "p1"),
        ("Dashboard numbers do not refresh",
         "The dashboard in {product} keeps showing {month} data even after a "
         "hard refresh. The API returns the right values, so it looks like a "
         "caching issue in the UI.",
         "{product} dashboard serves stale {month} data", "p2"),
        ("Search misses recent records",
         "Records created in the last hour do not appear in {product} search "
         "until the next day. Indexing seems to lag badly.",
         "{product} search indexing lags for recent records", "p2"),
        ("Timezone off by one hour",
         "Every timestamp in {product} shows one hour earlier than it should "
         "for users in the {team_name} office since the clocks changed.",
         "{product} timestamps off by one hour for {team_name} office", "p3"),
    ],
    "how_to": [
        ("How do I export {month} data?",
         "I need to pull all {product} records for {month} into a spreadsheet "
         "for the finance review. What is the supported way to do this?",
         "Asks how to export {month} data from {product}", "p4"),
        ("Best way to set up roles",
         "We are onboarding the {team_name} team onto {product} and want the "
         "recommended role setup for {count} people.",
         "Asks recommended role setup for {count}-person team", "p4"),
        ("Can I automate the {plan} plan reports?",
         "Is there an API or scheduled job that can send the {plan} plan "
         "report every month without someone clicking export?",
         "Asks about automating recurring {plan} plan reports", "p4"),
        ("Where are the audit logs?",
         "Our security review needs {product} audit logs for {month}. I "
         "cannot find them in the admin area. Where do they live?",
         "Asks where to find {product} audit logs for {month}", "p3"),
    ],
    "other": [
        ("Feedback on the new layout",
         "The {product} redesign shipped in {month} is much clearer, but the "
         "{team_name} team misses the old compact list view. Any plans to "
         "bring it back?",
         "Feedback on {month} redesign; requests compact view", "p4"),
        ("Partnership enquiry",
         "We are evaluating {product} for a joint offering with our own tools "
         "and would like to talk to someone about a partnership.",
         "Partnership enquiry about {product} integration", "p4"),
        ("Status page did not update",
         "During the incident on {month} your status page still showed all "
         "green while {product} was clearly down for us.",
         "Status page stayed green during {month} incident", "p3"),
        ("Request for a case study",
         "Our {team_name} team has been using {product} for a year and we are "
         "happy to be featured in a case study if you are interested.",
         "Offers to participate in a customer case study", "p4"),
    ],
}

POOLS: dict[str, list[str]] = {
    "month": ["January", "February", "March", "April", "May", "June", "July",
              "August", "September", "October", "November", "December"],
    "amount": ["$49", "$99", "$149", "$240", "$480", "$1,200"],
    "plan": ["Starter", "Pro", "Team", "Business", "Enterprise"],
    "invoice": ["INV-1042", "INV-2318", "INV-3907", "INV-4455", "INV-5120"],
    "product": ["the console", "the mobile app", "the web app", "the admin portal"],
    "team_name": ["support", "finance", "engineering", "sales", "operations"],
    "count": ["three", "five", "eight", "twelve", "twenty"],
}


def _build_sample(rng: random.Random, category: str, index: int, namespace: str) -> dict:
    template_index = rng.randrange(len(TEMPLATES[category]))
    subject, body, summary_t, priority = TEMPLATES[category][template_index]
    slots = {key: rng.choice(values) for key, values in POOLS.items()}
    request = f"Subject: {subject.format(**slots)}\n{body.format(**slots)}"
    summary = " ".join(summary_t.format(**slots).split()[:MAX_SUMMARY_WORDS])
    return {
        "sample_id": f"{namespace}-{category}-{stable_hash(category, index)[:8]}",
        "source": "synthetic:template",
        # Split group key: category plus template is the near-duplicate unit.
        "family": f"{category}:{template_index}",
        "category": category,
        "request": request,
        "expected_output": {
            "priority": priority,
            "category": category,
            # Gold routing comes from the contract the validator enforces.
            "team": ROUTING[category],
            "summary": summary,
        },
    }


def _validate(sample: dict) -> tuple[bool, str]:
    result = validate_triage(
        json.dumps(sample["expected_output"]),
        schema_path=SCHEMA_PATH,
        user_prompt=sample["request"],
    )
    if result.ok:
        return True, ""
    return False, "; ".join(f"L{e.layer}.{e.code}" for e in result.errors[:3])


def _generate(n: int, seed: int, namespace: str) -> list[dict]:
    rng = random.Random(seed)
    categories = sorted(TEMPLATES)
    rows: list[dict] = []
    seen: set[str] = set()
    index = 0
    while len(rows) < n:
        index += 1
        sample = _build_sample(rng, categories[index % len(categories)], index, namespace)
        if sample["sample_id"] in seen:
            continue
        ok, err = _validate(sample)
        if not ok:
            print(f"  [reject] {sample['category']}: {err}")
            continue
        seen.add(sample["sample_id"])
        rows.append(sample)
    return rows


def _write(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build the triage seed corpus.")
    parser.add_argument("--target", type=int, default=600)
    parser.add_argument("--benchmark-n", type=int, default=120)
    parser.add_argument("--seed", type=int, default=20260725)
    parser.add_argument("--out", default=str(SEEDS_PATH))
    parser.add_argument("--benchmark-out", default=str(BENCH_PATH))
    args = parser.parse_args(argv)

    seeds = _generate(args.target, args.seed, "tkt")
    _write(Path(args.out), seeds)
    print(f"wrote {len(seeds)} seed rows -> {args.out}")

    if args.benchmark_n > 0:
        # Benchmark rows are pinned to test, so they take their own family
        # namespace to stay disjoint from the training splits.
        bench = _generate(args.benchmark_n, args.seed + 1, "bench")
        for row in bench:
            row["family"] = f"bench:{row['family']}"
        _write(Path(args.benchmark_out), bench)
        print(f"wrote {len(bench)} benchmark rows -> {args.benchmark_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
