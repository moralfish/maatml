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
from .schemas import AgentSample, DslSample, ErrorCategory, SpoolSample, Split
from .synthetic.dsl_generator import generate_augmented_jsonl
from .synthetic.jcl_generator import generate_corpus
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
    """Synthesize the JCL Validator corpus from `model_def.data`."""
    cfg = model_def.data
    out = Path(out_dir) if out_dir else model_def.prepared_dir
    out.mkdir(parents=True, exist_ok=True)

    template_dir = model_def.resolve(cfg["template_dir"])
    n_per_class = {ErrorCategory(k): int(v) for k, v in cfg["n_per_class"].items()}
    ratios = _resolve_ratios(cfg)

    by_split: dict[Split, list[dict]] = {Split.train: [], Split.val: [], Split.test: []}
    label_counts: Counter[str] = Counter()

    for sample in generate_corpus(
        seed=int(cfg["seed"]),
        n_per_class=n_per_class,
        n_valid=int(cfg.get("n_valid", 0)),
        template_dir=template_dir,
        split_ratios=ratios,
    ):
        sanitized = sanitize_jcl(sample.sanitized_jcl)
        sample = sample.model_copy(update={"sanitized_jcl": sanitized})
        by_split[sample.split].append(sample.model_dump(mode="json"))
        label_counts[(sample.error_category or ErrorCategory.none).value] += 1

    paths = {split.value: str(_write_split(rows, out, split)) for split, rows in by_split.items()}
    split_counts = {split.value: len(rows) for split, rows in by_split.items()}
    card = _write_dataset_card(
        out,
        title=f"{model_def.model_id} dataset",
        counts=dict(label_counts),
        split_counts=split_counts,
        extra_lines=[
            f"Source: synthetic generator (seed={cfg['seed']})",
            f"Templates: {template_dir}",
        ],
    )

    summary = {
        "out_dir": str(out),
        "splits": paths,
        "card": str(card),
        "label_counts": dict(label_counts),
        "split_counts": split_counts,
    }
    console.print(f"[green]JCL prepare complete[/]: {summary['split_counts']}")
    return summary


def prepare_spool(model_def: ModelDefinition, out_dir: Optional[Path] = None) -> dict:
    """Build train/val/test splits for the Spool Interpreter from `model_def.data`."""
    cfg = model_def.data
    out = Path(out_dir) if out_dir else model_def.prepared_dir
    out.mkdir(parents=True, exist_ok=True)
    raw_field = cfg.get("raw_field", "raw_spool")
    ratios = _resolve_ratios(cfg)

    # `seed_samples` is the canonical single source for spool/dsl.  The legacy
    # `sources:` list is still supported for back-compat with old YAMLs.
    sources: list[str] = []
    if "seed_samples" in cfg:
        sources.append(cfg["seed_samples"])
    sources.extend(cfg.get("sources", []))
    if not sources:
        raise ValueError("model.yml `data:` must declare `seed_samples` or `sources`")

    by_split: dict[Split, list[dict]] = {Split.train: [], Split.val: [], Split.test: []}
    label_counts: Counter[str] = Counter()

    for source in sources:
        src_path = model_def.resolve(source)
        for raw in iter_jsonl(src_path):
            sanitized = sanitize_spool(raw[raw_field])
            split = _split_from_hash(raw["sample_id"], ratios)
            sample = SpoolSample(
                sample_id=raw["sample_id"],
                source=raw["source"],
                sanitized_spool=sanitized,
                status=raw["status"],
                return_code=raw.get("return_code"),
                failure_category=raw["failure_category"],
                root_cause=raw["root_cause"],
                suggested_fix=raw["suggested_fix"],
                split=split,
            )
            by_split[split].append(sample.model_dump(mode="json"))
            label_counts[sample.failure_category.value] += 1

    paths = {split.value: str(_write_split(rows, out, split)) for split, rows in by_split.items()}
    split_counts = {split.value: len(rows) for split, rows in by_split.items()}
    card = _write_dataset_card(
        out,
        title=f"{model_def.model_id} dataset",
        counts=dict(label_counts),
        split_counts=split_counts,
        extra_lines=[f"Sources: {sources}"],
    )

    summary = {
        "out_dir": str(out),
        "splits": paths,
        "card": str(card),
        "label_counts": dict(label_counts),
        "split_counts": split_counts,
    }
    console.print(f"[green]Spool prepare complete[/]: {summary['split_counts']}")
    return summary


def prepare_dsl(model_def: ModelDefinition, out_dir: Optional[Path] = None) -> dict:
    """Build train/val/test splits for the DSL Generator from `model_def.data`.

    If `data.augment` is present, the rule-based augmenter (
    `flow_ml.data.synthetic.dsl_generator.generate_augmented_jsonl`) is run
    on `data.seed_samples` first; the resulting JSONL replaces the seed file
    as the input to splitting (the augmented file already contains the seeds).
    """
    cfg = model_def.data
    out = Path(out_dir) if out_dir else model_def.prepared_dir
    out.mkdir(parents=True, exist_ok=True)
    raw_field = cfg.get("raw_field", "description")
    ratios = _resolve_ratios(cfg)

    seed_source: Path
    if "seed_samples" in cfg:
        seed_source = model_def.resolve(cfg["seed_samples"])
    elif cfg.get("sources"):
        seed_source = model_def.resolve(cfg["sources"][0])
    else:
        raise ValueError("model.yml `data:` must declare `seed_samples` or `sources`")

    # --- augmentation step (optional) ---
    aug_cfg = cfg.get("augment")
    sources: list[Path] = [seed_source]
    if aug_cfg:
        aug_out = model_def.resolve(aug_cfg["out"])
        target = int(aug_cfg.get("target_count", 1500))
        aug_seed = int(aug_cfg.get("seed", 42))
        console.print(
            f"[cyan]Augmenting[/] {seed_source.name} -> {aug_out.name} "
            f"(target={target}, seed={aug_seed})"
        )
        n = generate_augmented_jsonl(
            seed_path=seed_source,
            out_path=aug_out,
            target_count=target,
            seed=aug_seed,
            include_seeds=True,
        )
        console.print(f"  wrote {n} samples to {aug_out}")
        # The augmented file already includes seeds; use it instead of the seed file.
        sources = [aug_out]

    # Extra sources (legacy `sources:` entries past the first) are appended.
    for extra in cfg.get("sources", [])[1:]:
        sources.append(model_def.resolve(extra))

    by_split: dict[Split, list[dict]] = {Split.train: [], Split.val: [], Split.test: []}
    source_counts: Counter[str] = Counter()

    for src_path in sources:
        for raw in iter_jsonl(src_path):
            split = _split_from_hash(raw["sample_id"], ratios)
            sample = DslSample(
                sample_id=raw["sample_id"],
                source=raw["source"],
                description=raw[raw_field],
                dsl=raw["dsl"],
                split=split,
            )
            by_split[split].append(sample.model_dump(mode="json"))
            source_counts[sample.source] += 1

    paths = {split.value: str(_write_split(rows, out, split)) for split, rows in by_split.items()}
    split_counts = {split.value: len(rows) for split, rows in by_split.items()}
    card = _write_dataset_card(
        out,
        title=f"{model_def.model_id} dataset",
        counts=dict(source_counts),
        split_counts=split_counts,
        extra_lines=[f"Sources: {[str(s) for s in sources]}"],
    )

    summary = {
        "out_dir": str(out),
        "splits": paths,
        "card": str(card),
        "source_counts": dict(source_counts),
        "split_counts": split_counts,
    }
    console.print(f"[green]DSL prepare complete[/]: {summary['split_counts']}")
    return summary


def prepare_agent(model_def: ModelDefinition, out_dir: Optional[Path] = None) -> dict:
    """Build train/val/test splits for the Agent Planner.

    Hand-authored `seed_samples` are hash-split for training and validation.
    Optional `benchmark_samples` are always assigned to the test split so the
    local-agent benchmark stays fixed across retrains and candidate models.
    """
    cfg = model_def.data
    out = Path(out_dir) if out_dir else model_def.prepared_dir
    out.mkdir(parents=True, exist_ok=True)
    ratios = _resolve_ratios(cfg)

    if "seed_samples" not in cfg:
        raise ValueError("model.yml `data:` must declare `seed_samples`")

    by_split: dict[Split, list[dict]] = {Split.train: [], Split.val: [], Split.test: []}
    source_counts: Counter[str] = Counter()

    seed_path = model_def.resolve(cfg["seed_samples"])
    for raw in iter_jsonl(seed_path):
        split = _split_from_hash(raw["sample_id"], ratios)
        sample = AgentSample(
            sample_id=raw["sample_id"],
            source=raw["source"],
            request=raw["request"],
            context=raw.get("context", ""),
            expected_intent=raw["expected_intent"],
            agent_plan=raw["agent_plan"],
            split=split,
        )
        by_split[split].append(sample.model_dump(mode="json"))
        source_counts[sample.source] += 1

    benchmark_path = cfg.get("benchmark_samples")
    if benchmark_path:
        bench = model_def.resolve(benchmark_path)
        for raw in iter_jsonl(bench):
            sample = AgentSample(
                sample_id=raw["sample_id"],
                source=raw["source"],
                request=raw["request"],
                context=raw.get("context", ""),
                expected_intent=raw["expected_intent"],
                agent_plan=raw["agent_plan"],
                split=Split.test,
            )
            by_split[Split.test].append(sample.model_dump(mode="json"))
            source_counts[sample.source] += 1

    paths = {split.value: str(_write_split(rows, out, split)) for split, rows in by_split.items()}
    split_counts = {split.value: len(rows) for split, rows in by_split.items()}
    card = _write_dataset_card(
        out,
        title=f"{model_def.model_id} dataset",
        counts=dict(source_counts),
        split_counts=split_counts,
        extra_lines=[
            f"Seed source: {seed_path}",
            f"Benchmark source: {model_def.resolve(benchmark_path) if benchmark_path else 'none'}",
        ],
    )

    summary = {
        "out_dir": str(out),
        "splits": paths,
        "card": str(card),
        "source_counts": dict(source_counts),
        "split_counts": split_counts,
    }
    console.print(f"[green]Agent prepare complete[/]: {summary['split_counts']}")
    return summary


def prepare_dataset() -> None:
    console.print(
        "Use prepare_jcl(model_def, out_dir), prepare_spool(model_def, out_dir), "
        "prepare_dsl(model_def, out_dir), or prepare_agent(model_def, out_dir)"
    )
