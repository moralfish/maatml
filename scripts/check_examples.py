"""Cheap per-example invariants: splits, gate wiring, and gold self-consistency.

Runs no training and needs no GPU. For each example it prepares the corpus and
then scores the test split using each row's own gold as the prediction, which
gives the full set of metric keys a real evaluation would emit.

Checks:
  1. every split is non-empty (whole-group hashing with too few families
     silently leaves val empty, and training then skips evaluation)
  2. every declared gate names a metric the plugin actually emits
  3. the corpus's own gold passes the validator's contract layers

Full lifecycle verification stays local; see scripts/train_all.py and
scripts/evaluate_all.py.

Usage:
    python scripts/check_examples.py
    python scripts/check_examples.py --only vision jcl-validator
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from maatml.config import get_dataset_cfg, load_model_def  # noqa: E402
from maatml.evaluation.harness import (  # noqa: E402
    RowEval,
    effective_gates,
    _resolve_metrics,
    resolve_validator,
)
from maatml.registry import FORMATS, discover_plugins  # noqa: E402
from maatml.utils.io import iter_jsonl  # noqa: E402

# Reported by the harness rather than by a model's metrics plugin.
_HARNESS_METRICS = frozenset({"output_nonempty_rate"})


def _model_dirs(only: list[str] | None) -> list[Path]:
    root = ROOT / "examples"
    dirs = [d for d in sorted(root.iterdir()) if (d / "model.yml").is_file()]
    if only:
        wanted = set(only)
        dirs = [d for d in dirs if d.name in wanted]
    return dirs


def _gold_row_evals(model_def, rows: list[dict]) -> list[RowEval]:
    """Score each row against its own gold, so no model is needed."""
    cfg = get_dataset_cfg(model_def)
    target_field = str(cfg.get("target_field") or "target")
    request_field = str(cfg.get("request_field") or cfg.get("raw_field") or "request")
    validator_name = (model_def.evaluation or {}).get("validator")
    validate = resolve_validator(str(validator_name)) if validator_name else None
    schema = model_def.resolve(cfg["schema"]) if isinstance(cfg.get("schema"), str) else None
    contracts = (
        model_def.resolve(cfg["contracts"]) if isinstance(cfg.get("contracts"), str) else None
    )

    out: list[RowEval] = []
    for row in rows:
        gold = row.get(target_field)
        text = gold if isinstance(gold, str) else json.dumps(gold)
        result = validate(
            text,
            schema_path=schema,
            contracts_path=contracts,
            user_prompt=row.get(request_field),
        )
        out.append(RowEval(row=row, gen_text=text, result=result))
    return out


def check(model_dir: Path) -> list[str]:
    errors: list[str] = []
    model_def = load_model_def(model_dir)
    name = model_dir.name

    fmt = str(get_dataset_cfg(model_def).get("format", "jsonl_seed"))
    summary = FORMATS.require(fmt)(model_def) or {}
    counts = summary.get("split_counts") or {}
    for split in ("train", "val", "test"):
        if not counts.get(split):
            errors.append(f"{name}: {split} split is empty ({counts})")
    if errors:
        return errors

    test_rows = list(iter_jsonl(model_def.prepared_dir / "test.jsonl"))
    gates = effective_gates(model_def)
    if not gates:
        errors.append(f"{name}: no evaluation.gates declared")

    metrics_fns = _resolve_metrics((model_def.evaluation or {}).get("metrics"))
    if not metrics_fns:
        errors.append(f"{name}: evaluation.metrics does not resolve")
        return errors

    row_evals = _gold_row_evals(model_def, test_rows)
    emitted = set(_HARNESS_METRICS)
    for fn in metrics_fns:
        emitted |= set(fn(row_evals))
    missing = sorted(set(gates) - emitted)
    if missing:
        errors.append(f"{name}: gates name metrics never emitted: {missing}")

    failed_gold = sum(1 for r in row_evals if not r.result.ok)
    if failed_gold:
        errors.append(
            f"{name}: {failed_gold}/{len(row_evals)} gold rows fail the validator"
        )
    return errors


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", nargs="+", help="Subset by folder name")
    args = parser.parse_args(argv)

    discover_plugins()
    failures: list[str] = []
    for model_dir in _model_dirs(args.only):
        try:
            errors = check(model_dir)
        except Exception as exc:  # noqa: BLE001  report, do not abort the sweep
            errors = [f"{model_dir.name}: {type(exc).__name__}: {exc}"]
        if errors:
            failures.extend(errors)
            for err in errors:
                print(f"FAIL {err}")
        else:
            print(f"ok   {model_dir.name}")
    if failures:
        print(f"\n{len(failures)} problem(s)")
        return 1
    print("\nall examples pass the cheap invariants")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
