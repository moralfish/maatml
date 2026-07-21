"""Build a deterministic vision-describer seed corpus.

Generates prediction-like vision JSON (with confidence/coordinate noise),
linearizes it, and pairs it with a template caption. Every row is gated by
the task validator before writing.

Usage:
    python examples/vision-describer/scripts/build_seeds.py
    python examples/vision-describer/scripts/build_seeds.py --target 400
    python examples/vision-describer/scripts/build_seeds.py --benchmark-only
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

EXAMPLE_ROOT = Path(__file__).resolve().parents[1]
REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(EXAMPLE_ROOT))

from vision_describer_plugin.constants import SCENE_LABELS  # noqa: E402
from vision_describer_plugin.generator import make_sample_row  # noqa: E402
from vision_describer_plugin.validator import validate_vision_describer  # noqa: E402

DATASETS = EXAMPLE_ROOT / "datasets"
SCHEMA_PATH = DATASETS / "schema.json"
CONTRACTS_PATH = DATASETS / "node_contracts.json"
SEEDS_PATH = DATASETS / "samples" / "seed_samples.jsonl"
BENCH_PATH = DATASETS / "samples" / "benchmark_samples.jsonl"

POSE_STYLES = ["both_up", "left_up", "right_up", "lowered", "upright"]


def _gate(row: dict) -> bool:
    result = validate_vision_describer(
        json.dumps(row["expected_description"], ensure_ascii=False),
        schema_path=SCHEMA_PATH,
        contracts_path=CONTRACTS_PATH,
        user_prompt=row["request"],
    )
    return result.ok


def build_corpus(target: int, seed: int) -> list[dict]:
    rows: list[dict] = []
    idx = 0
    attempts = 0
    max_attempts = target * 20
    while len(rows) < target and attempts < max_attempts:
        attempts += 1
        scene = SCENE_LABELS[idx % len(SCENE_LABELS)]
        pose_style = POSE_STYLES[idx % len(POSE_STYLES)]
        n_dets = idx % 4  # 0..3
        row = make_sample_row(
            idx,
            seed=seed,
            scene=scene,
            n_dets=n_dets,
            pose_style=pose_style,
        )
        idx += 1
        if _gate(row):
            rows.append(row)
    if len(rows) < target:
        raise RuntimeError(
            f"Only accepted {len(rows)}/{target} rows after {attempts} attempts"
        )
    return rows


def build_benchmark(seed: int = 7) -> list[dict]:
    """One fixed anchor per scene × a couple of pose/object variants."""
    rows: list[dict] = []
    i = 0
    for scene in SCENE_LABELS:
        for pose_style, n_dets in (("both_up", 2), ("lowered", 0), ("upright", 1)):
            row = make_sample_row(
                10_000 + i,
                seed=seed,
                scene=scene,
                n_dets=n_dets,
                pose_style=pose_style,
            )
            i += 1
            row["source"] = "benchmark:vision_describer"
            row["family"] = f"bench:{scene}"
            if _gate(row):
                rows.append(row)
    return rows


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", type=int, default=120, help="Seed corpus size")
    parser.add_argument("--seed", type=int, default=4242)
    parser.add_argument(
        "--benchmark-only",
        action="store_true",
        help="Only regenerate benchmark_samples.jsonl",
    )
    parser.add_argument(
        "--append",
        action="store_true",
        help="Append to existing seed_samples.jsonl instead of overwriting",
    )
    args = parser.parse_args()

    bench = build_benchmark(seed=args.seed)
    _write_jsonl(BENCH_PATH, bench)
    print(f"wrote {len(bench)} benchmark rows → {BENCH_PATH}")

    if args.benchmark_only:
        return

    rows = build_corpus(args.target, args.seed)
    if args.append and SEEDS_PATH.is_file():
        existing = [
            json.loads(line)
            for line in SEEDS_PATH.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        seen = {r["sample_id"] for r in existing}
        added = [r for r in rows if r["sample_id"] not in seen]
        rows = existing + added
        print(f"append: +{len(added)} new rows (total {len(rows)})")

    _write_jsonl(SEEDS_PATH, rows)
    print(f"wrote {len(rows)} seed rows → {SEEDS_PATH}")


if __name__ == "__main__":
    main()
