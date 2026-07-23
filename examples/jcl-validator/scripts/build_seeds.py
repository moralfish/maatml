"""Build the JCL Validator seed corpus deterministically.

Reuses the template + defect-injection primitives in
`jcl_plugin.generator` to produce a balanced corpus across all eight
categories (valid + 7 error codes), then gates every sample through the
6-layer `validate_jcl_result` check before writing it to
`datasets/samples/seed_samples.jsonl` under this example folder.

No API calls. Reproducible given the seed. Run anytime to regenerate
or extend the corpus.

Usage:
    python examples/jcl-validator/scripts/build_seeds.py
    python examples/jcl-validator/scripts/build_seeds.py --target 800
    python examples/jcl-validator/scripts/build_seeds.py --append
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

from jcl_plugin.generator import (  # noqa: E402
    INJECTORS,
    _load_templates,
    _render_template,
    DEFAULT_TEMPLATE_DIR,
)
from jcl_plugin.schemas import ErrorCategory  # noqa: E402
from jcl_plugin.validator import validate_jcl_result  # noqa: E402
from maatml.utils.io import stable_hash  # noqa: E402


MODEL_DIR = EXAMPLE_ROOT
DATASETS = MODEL_DIR / "datasets"
SCHEMA_PATH = DATASETS / "jcl_validation_schema.json"
CONTRACTS_PATH = DATASETS / "node_contracts.json"
SEEDS_PATH = DATASETS / "samples" / "seed_samples.jsonl"


# Category quotas for a ~1000-sample corpus. Sized so each error code has
# enough support to avoid mode collapse, with too few samples per minor
# code, classifiers / generators collapse to a dominant label. 100+
# samples per error code is a practical floor; valid stays moderate at
# 200 so we don't bias the
# model back toward false-clean predictions.
DEFAULT_QUOTAS: dict[str, int] = {
    "valid": 200,
    ErrorCategory.missing_dd.value: 120,
    ErrorCategory.invalid_job_card.value: 115,
    ErrorCategory.invalid_exec_statement.value: 115,
    ErrorCategory.invalid_dataset_reference_structure.value: 110,
    ErrorCategory.unresolved_symbolic_parameter.value: 115,
    ErrorCategory.continuation_error.value: 110,
    ErrorCategory.other.value: 115,
}


SEVERITY_FOR_CATEGORY: dict[str, str] = {
    ErrorCategory.missing_dd.value: "error",
    ErrorCategory.invalid_job_card.value: "error",
    ErrorCategory.invalid_exec_statement.value: "error",
    ErrorCategory.invalid_dataset_reference_structure.value: "error",
    ErrorCategory.unresolved_symbolic_parameter.value: "error",
    ErrorCategory.continuation_error.value: "error",
    ErrorCategory.other.value: "warning",
}


CATEGORY_MESSAGES: dict[str, str] = {
    ErrorCategory.missing_dd.value: "DD statement missing the 'DD' keyword.",
    ErrorCategory.invalid_job_card.value: "JOB card is malformed (account, class, or parameter syntax).",
    ErrorCategory.invalid_exec_statement.value: "EXEC statement missing 'PGM=' or using an unknown keyword.",
    ErrorCategory.invalid_dataset_reference_structure.value: "Dataset name violates the qualifier-dot-qualifier structure.",
    ErrorCategory.unresolved_symbolic_parameter.value: "Symbolic parameter is referenced but never set.",
    ErrorCategory.continuation_error.value: "Continuation line must begin with `//` in columns 1-2.",
    ErrorCategory.other.value: "Statement label must begin in column 1 with `//`.",
}


def _build_valid_sample(
    rng: random.Random, templates: list[tuple[str, str]], idx: int
) -> dict | None:
    template_id, raw = rng.choice(templates)
    rendered = _render_template(rng, raw).rstrip() + "\n"
    sample_id = f"syn-valid-{stable_hash(template_id, idx, 'valid')[:8]}"
    confidence = round(rng.uniform(0.90, 0.97), 2)
    return {
        "sample_id": sample_id,
        "source": f"synthetic:{template_id}",
        "family": template_id,
        "category": "valid",
        "request": rendered,
        "expected_validation_result": {
            "valid": True,
            "errors": [],
            "confidence": confidence,
        },
    }


def _build_error_sample(
    rng: random.Random,
    templates: list[tuple[str, str]],
    category: ErrorCategory,
    idx: int,
) -> dict | None:
    template_id, raw = rng.choice(templates)
    rendered = _render_template(rng, raw)
    lines = rendered.splitlines()

    injector = INJECTORS[category]
    result = injector.inject(rng, lines)
    if result is None:
        return None
    new_lines, error_line, error_column, suggestion = result

    request = "\n".join(new_lines).rstrip() + "\n"
    error_obj: dict = {
        "line": int(error_line),
        "severity": SEVERITY_FOR_CATEGORY[category.value],
        "code": category.value,
        "message": CATEGORY_MESSAGES[category.value],
        "suggestion": suggestion,
    }
    if error_column is not None:
        error_obj["column"] = int(error_column)

    confidence = round(rng.uniform(0.82, 0.95), 2)
    sample_id = f"syn-{category.value}-{stable_hash(template_id, idx, category.value)[:8]}"
    return {
        "sample_id": sample_id,
        "source": f"synthetic:{template_id}",
        "family": template_id,
        "category": category.value,
        "request": request,
        "expected_validation_result": {
            "valid": False,
            "errors": [error_obj],
            "confidence": confidence,
        },
    }


def _validate(sample: dict) -> tuple[bool, str]:
    raw = json.dumps(sample["expected_validation_result"])
    result = validate_jcl_result(
        raw,
        schema_path=SCHEMA_PATH,
        contracts_path=CONTRACTS_PATH,
        user_prompt=sample["request"],
    )
    if result.ok:
        return True, ""
    errs = "; ".join(f"L{e.layer}.{e.code}" for e in result.errors[:3])
    return False, errs


def _read_existing_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    seen: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        sid = row.get("sample_id")
        if sid:
            seen.add(sid)
    return seen


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build the JCL Validator seed corpus.")
    parser.add_argument("--target", type=int, default=1000, help="Total samples (default 1000)")
    parser.add_argument("--seed", type=int, default=20260510)
    parser.add_argument(
        "--append",
        action="store_true",
        help="Keep existing rows in seed_samples.jsonl; append new ones with fresh ids",
    )
    parser.add_argument(
        "--out",
        default=str(SEEDS_PATH),
        help="Override the output JSONL path",
    )
    args = parser.parse_args(argv)

    rng = random.Random(args.seed)
    templates = _load_templates(DEFAULT_TEMPLATE_DIR)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    existing_rows: list[dict] = []
    if args.append and out_path.exists():
        for line in out_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
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
    idx = 0
    for category, n in quotas.items():
        produced = 0
        attempts = 0
        max_attempts = n * 20
        while produced < n and attempts < max_attempts:
            attempts += 1
            idx += 1
            if category == "valid":
                sample = _build_valid_sample(rng, templates, idx)
            else:
                sample = _build_error_sample(rng, templates, ErrorCategory(category), idx)
            if sample is None:
                continue
            if sample["sample_id"] in seen_ids:
                continue
            ok, err = _validate(sample)
            if not ok:
                rejected += 1
                if rejected <= 3:
                    print(f"  [reject] {category}: {err}")
                continue
            accepted.append(sample)
            seen_ids.add(sample["sample_id"])
            produced += 1
        print(f"  {category}: produced={produced}/{n} attempts={attempts}")

    if args.append:
        rows_to_write = existing_rows + accepted
    else:
        rows_to_write = accepted

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
