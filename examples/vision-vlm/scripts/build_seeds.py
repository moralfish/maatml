#!/usr/bin/env python3
"""Deterministic seed corpus builder for examples/vision-vlm.

Usage:
  python examples/vision-vlm/scripts/build_seeds.py --target 16
  python examples/vision-vlm/scripts/build_seeds.py --target 300
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from vlm_plugin.datagen import build_described_row  # noqa: E402
from vlm_plugin.validator import validate_vision_vlm  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--target", type=int, default=16)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--size", type=int, default=320)
    p.add_argument(
        "--out",
        type=Path,
        default=ROOT / "datasets" / "samples" / "seed_samples.jsonl",
    )
    p.add_argument(
        "--benchmark",
        type=Path,
        default=ROOT / "datasets" / "samples" / "benchmark_samples.jsonl",
    )
    p.add_argument("--benchmark-n", type=int, default=8)
    args = p.parse_args()

    images_dir = ROOT / "datasets" / "samples" / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    schema = ROOT / "datasets" / "schema.json"

    rows = []
    rejected = 0
    i = 0
    while len(rows) < args.target:
        row = build_described_row(
            i,
            base_seed=args.seed,
            size=args.size,
            image_rel="datasets/samples/images/{id}.png",
            images_dir=images_dir,
        )
        i += 1
        vr = validate_vision_vlm(
            json.dumps(row["expected_output"]),
            schema_path=schema,
        )
        if not vr.ok:
            rejected += 1
            continue
        rows.append(row)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    bench = rows[: args.benchmark_n]
    for b in bench:
        b["source"] = "benchmark:fixed"
    with args.benchmark.open("w", encoding="utf-8") as fh:
        for row in bench:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(
        f"wrote {len(rows)} seeds → {args.out} "
        f"(rejected={rejected}); benchmark={args.benchmark_n} → {args.benchmark}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
