"""Data preparation: synthesize / load corpora, split into train/val/test JSONLs.

All three `prepare_*` functions take a `ModelDefinition` (loaded from
`models/<name>/model.yml`) and write splits into the model's
`output/prepared/` folder by default.

The data section of `model.yml` carries everything that used to live in the
old `configs/<name>/data.yaml` files:

  data:
    seed: 7331
    schema: datasets/schema.json          # optional; informational
    prompt_spec: datasets/prompt_spec.json # optional; informational
    seed_samples: datasets/samples/seed_samples.jsonl  # spool / dsl
    template_dir: datasets/templates       # jcl
    raw_field: description | raw_spool
    split_ratios: [0.6, 0.2, 0.2]
    augment:                               # dsl only (rule-based)
      target_count: 1500
      seed: 42
      out: datasets/samples/augmented_samples.jsonl
    n_per_class: { ... }                   # jcl only (synthetic generation)
    n_valid: 2000                          # jcl only

All paths inside `data:` are resolved relative to the model folder.
"""
from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Iterable, Optional

from rich.console import Console

from ..config import ModelDefinition
from .sanitizer import sanitize_jcl, sanitize_spool
from .schemas import (
    FlowGraphProposal,
    FlowGraphSample,
    JclSample,
    JclValidationResult,
    SpoolInterpretation,
    SpoolSample,
    Split,
)
from ..utils.io import iter_jsonl, stable_hash, write_jsonl

console = Console()


def _split_from_hash(sample_id: str, ratios: tuple[float, float, float]) -> Split:
    digest = stable_hash(sample_id)
    bucket = int(digest[:12], 16) / float(1 << 48)
    train_r, val_r, _ = ratios
    if bucket < train_r:
        return Split.train
    if bucket < train_r + val_r:
        return Split.val
    return Split.test


def _write_split(rows: list[dict], out_dir: Path, split: Split) -> Path:
    return write_jsonl(out_dir / f"{split.value}.jsonl", rows)


def _write_dataset_card(
    out_dir: Path,
    *,
    title: str,
    counts: dict[str, int],
    split_counts: dict[str, int],
    extra_lines: Iterable[str] = (),
) -> Path:
    lines = [f"# {title}", "", "## Counts by split", ""]
    for split in (Split.train, Split.val, Split.test):
        lines.append(f"- {split.value}: {split_counts.get(split.value, 0)}")
    lines.extend(["", "## Counts by label", ""])
    for label, n in sorted(counts.items()):
        lines.append(f"- {label}: {n}")
    lines.extend(["", *extra_lines])
    path = out_dir / "dataset_card.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _resolve_ratios(data_cfg: dict) -> tuple[float, float, float]:
    ratios = tuple(data_cfg.get("split_ratios", [0.8, 0.1, 0.1]))
    if len(ratios) != 3 or abs(sum(ratios) - 1.0) > 1e-6:
        raise ValueError(f"split_ratios must sum to 1.0; got {ratios}")
    return ratios  # type: ignore[return-value]


def prepare_jcl(model_def: ModelDefinition, out_dir: Optional[Path] = None) -> dict:
    """Build train/val/test splits for JCL Validator (generative SFT).

    Hand-authored samples in `seed_samples.jsonl`. Each row carries
    `{sample_id, source, category, request, expected_validation_result}` where
    `request` is sanitized JCL text and `expected_validation_result` is the
    gold `JclValidationResult` JSON. Splits 80/10/10 by hash.

    The legacy classifier-corpus generator from `synthetic/jcl_generator.py`
    is no longer wired into prepare; it lives on as a developer tool for
    seeding hand-authored few-shot pools when needed.
    """
    cfg = model_def.data
    out = Path(out_dir) if out_dir else model_def.prepared_dir
    out.mkdir(parents=True, exist_ok=True)
    ratios = _resolve_ratios(cfg)

    if "seed_samples" not in cfg:
        raise ValueError("model.yml `data:` must declare `seed_samples`")

    by_split: dict[Split, list[dict]] = {Split.train: [], Split.val: [], Split.test: []}
    category_counts: Counter[str] = Counter()
    source_counts: Counter[str] = Counter()

    seed_path = model_def.resolve(cfg["seed_samples"])
    for raw in iter_jsonl(seed_path):
        sanitized = sanitize_jcl(raw["request"])
        split = _split_from_hash(raw["sample_id"], ratios)
        sample = JclSample(
            sample_id=raw["sample_id"],
            source=raw["source"],
            category=raw["category"],
            request=sanitized,
            expected_validation_result=JclValidationResult.model_validate(
                raw["expected_validation_result"]
            ),
            split=split,
        )
        by_split[split].append(sample.model_dump(mode="json"))
        category_counts[sample.category] += 1
        source_counts[sample.source] += 1

    benchmark_path = cfg.get("benchmark_samples")
    if benchmark_path:
        bench = model_def.resolve(benchmark_path)
        for raw in iter_jsonl(bench):
            if "expected_validation_result" not in raw:
                continue
            sanitized = sanitize_jcl(raw["request"])
            sample = JclSample(
                sample_id=raw["sample_id"],
                source=raw["source"],
                category=raw["category"],
                request=sanitized,
                expected_validation_result=JclValidationResult.model_validate(
                    raw["expected_validation_result"]
                ),
                split=Split.test,
            )
            by_split[Split.test].append(sample.model_dump(mode="json"))
            category_counts[sample.category] += 1
            source_counts[sample.source] += 1

    paths = {split.value: str(_write_split(rows, out, split)) for split, rows in by_split.items()}
    split_counts = {split.value: len(rows) for split, rows in by_split.items()}
    card = _write_dataset_card(
        out,
        title=f"{model_def.model_id} dataset",
        counts=dict(category_counts),
        split_counts=split_counts,
        extra_lines=[
            f"Seed source: {seed_path}",
            f"Benchmark source: {model_def.resolve(benchmark_path) if benchmark_path else 'none'}",
            f"Sources: {dict(source_counts)}",
        ],
    )

    summary = {
        "out_dir": str(out),
        "splits": paths,
        "card": str(card),
        "category_counts": dict(category_counts),
        "source_counts": dict(source_counts),
        "split_counts": split_counts,
    }
    console.print(
        f"[green]JCL prepare complete[/]: {summary['split_counts']} "
        f"(categories: {len(category_counts)})"
    )
    return summary


def prepare_spool(model_def: ModelDefinition, out_dir: Optional[Path] = None) -> dict:
    """Build train/val/test splits for the Spool Interpreter (generative SFT).

    Hand-authored + Claude-generated samples in `seed_samples.jsonl`. Each
    row carries `{sample_id, source, category, request, expected_interpretation}`
    where `request` is sanitized spool text and `expected_interpretation` is
    the gold `SpoolInterpretation` JSON. Splits 80/10/10 by hash.
    """
    cfg = model_def.data
    out = Path(out_dir) if out_dir else model_def.prepared_dir
    out.mkdir(parents=True, exist_ok=True)
    ratios = _resolve_ratios(cfg)

    if "seed_samples" not in cfg:
        raise ValueError("model.yml `data:` must declare `seed_samples`")

    by_split: dict[Split, list[dict]] = {Split.train: [], Split.val: [], Split.test: []}
    category_counts: Counter[str] = Counter()
    source_counts: Counter[str] = Counter()

    seed_path = model_def.resolve(cfg["seed_samples"])
    for raw in iter_jsonl(seed_path):
        sanitized = sanitize_spool(raw["request"])
        split = _split_from_hash(raw["sample_id"], ratios)
        sample = SpoolSample(
            sample_id=raw["sample_id"],
            source=raw["source"],
            category=raw["category"],
            request=sanitized,
            expected_interpretation=SpoolInterpretation.model_validate(
                raw["expected_interpretation"]
            ),
            split=split,
        )
        by_split[split].append(sample.model_dump(mode="json"))
        category_counts[sample.category] += 1
        source_counts[sample.source] += 1

    benchmark_path = cfg.get("benchmark_samples")
    if benchmark_path:
        bench = model_def.resolve(benchmark_path)
        for raw in iter_jsonl(bench):
            if "expected_interpretation" not in raw:
                continue
            sanitized = sanitize_spool(raw["request"])
            sample = SpoolSample(
                sample_id=raw["sample_id"],
                source=raw["source"],
                category=raw["category"],
                request=sanitized,
                expected_interpretation=SpoolInterpretation.model_validate(
                    raw["expected_interpretation"]
                ),
                split=Split.test,
            )
            by_split[Split.test].append(sample.model_dump(mode="json"))
            category_counts[sample.category] += 1
            source_counts[sample.source] += 1

    paths = {split.value: str(_write_split(rows, out, split)) for split, rows in by_split.items()}
    split_counts = {split.value: len(rows) for split, rows in by_split.items()}
    card = _write_dataset_card(
        out,
        title=f"{model_def.model_id} dataset",
        counts=dict(category_counts),
        split_counts=split_counts,
        extra_lines=[
            f"Seed source: {seed_path}",
            f"Benchmark source: {model_def.resolve(benchmark_path) if benchmark_path else 'none'}",
            f"Sources: {dict(source_counts)}",
        ],
    )

    summary = {
        "out_dir": str(out),
        "splits": paths,
        "card": str(card),
        "category_counts": dict(category_counts),
        "source_counts": dict(source_counts),
        "split_counts": split_counts,
    }
    console.print(
        f"[green]Spool prepare complete[/]: {summary['split_counts']} "
        f"(categories: {len(category_counts)})"
    )
    return summary


def prepare_flow_graph(model_def: ModelDefinition, out_dir: Optional[Path] = None) -> dict:
    """Build train/val/test splits for FlowGraphGenerator.

    Hand-authored + Claude-converted samples in `seed_samples.jsonl`. Each row
    has `{sample_id, source, category, request, expected_graph}`. Optional
    `benchmark_samples` (the 8 doc-canonical eval prompts) are always routed
    to the test split so the regression set stays fixed across retrains.

    No augmentation step — pure SFT, the safety categories require carefully
    designed gold graphs that augmentation can't synthesise.
    """
    cfg = model_def.data
    out = Path(out_dir) if out_dir else model_def.prepared_dir
    out.mkdir(parents=True, exist_ok=True)
    ratios = _resolve_ratios(cfg)

    if "seed_samples" not in cfg:
        raise ValueError("model.yml `data:` must declare `seed_samples`")

    by_split: dict[Split, list[dict]] = {Split.train: [], Split.val: [], Split.test: []}
    category_counts: Counter[str] = Counter()
    source_counts: Counter[str] = Counter()

    seed_path = model_def.resolve(cfg["seed_samples"])
    for raw in iter_jsonl(seed_path):
        split = _split_from_hash(raw["sample_id"], ratios)
        sample = FlowGraphSample(
            sample_id=raw["sample_id"],
            source=raw["source"],
            category=raw["category"],
            request=raw["request"],
            expected_graph=FlowGraphProposal.model_validate(raw["expected_graph"]),
            split=split,
        )
        by_split[split].append(sample.model_dump(mode="json"))
        category_counts[sample.category] += 1
        source_counts[sample.source] += 1

    benchmark_path = cfg.get("benchmark_samples")
    if benchmark_path:
        bench = model_def.resolve(benchmark_path)
        for raw in iter_jsonl(bench):
            # Benchmark prompts ship without gold graphs; eval scores against
            # the validator + expected_behavior text, not against a target.
            # Only include if the benchmark file carries `expected_graph`.
            if "expected_graph" not in raw:
                continue
            sample = FlowGraphSample(
                sample_id=raw["sample_id"],
                source=raw["source"],
                category=raw["category"],
                request=raw["request"],
                expected_graph=FlowGraphProposal.model_validate(raw["expected_graph"]),
                split=Split.test,
            )
            by_split[Split.test].append(sample.model_dump(mode="json"))
            category_counts[sample.category] += 1
            source_counts[sample.source] += 1

    paths = {split.value: str(_write_split(rows, out, split)) for split, rows in by_split.items()}
    split_counts = {split.value: len(rows) for split, rows in by_split.items()}
    card = _write_dataset_card(
        out,
        title=f"{model_def.model_id} dataset",
        counts=dict(category_counts),
        split_counts=split_counts,
        extra_lines=[
            f"Seed source: {seed_path}",
            f"Benchmark source: {model_def.resolve(benchmark_path) if benchmark_path else 'none'}",
            f"Sources: {dict(source_counts)}",
        ],
    )

    summary = {
        "out_dir": str(out),
        "splits": paths,
        "card": str(card),
        "category_counts": dict(category_counts),
        "source_counts": dict(source_counts),
        "split_counts": split_counts,
    }
    console.print(
        f"[green]FlowGraph prepare complete[/]: {summary['split_counts']} "
        f"(categories: {len(category_counts)})"
    )
    return summary


def prepare_dataset() -> None:
    console.print(
        "Use prepare_jcl(model_def, out_dir), prepare_spool(model_def, out_dir), "
        "prepare_dsl(model_def, out_dir), or prepare_agent(model_def, out_dir)"
    )
