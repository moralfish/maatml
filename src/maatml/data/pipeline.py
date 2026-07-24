"""Data preparation: load seed corpora, split into train/val/test JSONLs.

The generic ``prepare`` entry point reads knobs from ``dataset:`` (falling
back to ``data:``), optionally sanitizes by tag (via the ``SANITIZERS``
registry), splits by group key (``dataset.group_by`` when set, else
``family`` → ``source`` → ``sample_id``), pins benchmarks to test, and
writes splits + a dataset card.
"""
from __future__ import annotations

import json
import warnings
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable, Optional

from rich.console import Console

from ..config import ModelDefinition, get_dataset_cfg
from ..registry import SANITIZERS, register_format
from .schemas import Split
from ..utils.io import iter_jsonl, stable_hash, write_jsonl

console = Console()


def _split_from_hash(key: str, ratios: tuple[float, float, float]) -> Split:
    digest = stable_hash(key)
    bucket = int(digest[:12], 16) / float(1 << 48)
    train_r, val_r, _ = ratios
    if bucket < train_r:
        return Split.train
    if bucket < train_r + val_r:
        return Split.val
    return Split.test


_warned_group_fallback: set[str] = set()


def _group_key(row: dict, group_by: Optional[str] = None) -> str:
    """Stable group key for leakage-safe splitting.

    When ``dataset.group_by`` is set (e.g. ``family``), that field is tried
    first. Otherwise, or if the preferred field is missing, fall back to
    ``family`` → ``source`` → ``sample_id``. Falling back past ``family``
    emits a one-shot warning so operators notice missing family tags.
    """
    preferred = (group_by or "").strip() or None
    if preferred:
        val = row.get(preferred)
        if isinstance(val, str) and val.strip():
            return f"{preferred}:{val.strip()}"
        if val is not None and not isinstance(val, str):
            return f"{preferred}:{val}"

    family = row.get("family")
    if isinstance(family, str) and family.strip():
        return f"family:{family.strip()}"
    source = row.get("source")
    if isinstance(source, str) and source.strip():
        key = f"source:{source.strip()}"
        if key not in _warned_group_fallback:
            _warned_group_fallback.add(key)
            warnings.warn(
                f"samples with source={source!r} have no family; splitting by source "
                f"(re-run seed builders to stamp family)",
                stacklevel=3,
            )
        return key
    sid = str(row.get("sample_id") or "")
    key = f"sample_id:{sid}"
    if "sample_id" not in _warned_group_fallback:
        _warned_group_fallback.add("sample_id")
        warnings.warn(
            "samples missing family/source; splitting by sample_id",
            stacklevel=3,
        )
    return key


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


def _resolve_ratios(cfg: dict) -> tuple[float, float, float]:
    ratios = tuple(cfg.get("split_ratios", [0.8, 0.1, 0.1]))
    if len(ratios) != 3 or abs(sum(ratios) - 1.0) > 1e-6:
        raise ValueError(f"split_ratios must sum to 1.0; got {ratios}")
    return ratios  # type: ignore[return-value]


def _apply_sanitize(row: dict, sanitize_tags: list[str], request_field: str) -> dict:
    if not sanitize_tags:
        return row
    out = dict(row)
    text = out.get(request_field)
    if not isinstance(text, str):
        return out
    for tag in sanitize_tags:
        fn = SANITIZERS.get(tag)
        if fn is None:
            raise ValueError(
                f"Unknown sanitize tag {tag!r}; known: {SANITIZERS.names()} "
                f"(register via @register_sanitizer or model.yml plugins:)"
            )
        text = fn(text)
    out[request_field] = text
    return out


# A group holding at least this share of the corpus cannot be split: hashing
# it once sends (nearly) every row to the same split, leaving val/test empty.
_DEGENERATE_GROUP_SHARE = 0.9


def _row_identity(row: dict) -> str:
    """Per-row group key: ``sample_id`` when present, else a content hash."""
    sid = row.get("sample_id")
    if isinstance(sid, str) and sid.strip():
        return sid.strip()
    if sid is not None and not isinstance(sid, str):
        return str(sid)
    return stable_hash(json.dumps(row, sort_keys=True, default=str))[:16]


def _group_rows(
    rows: list[dict], *, group_by: Optional[str] = None
) -> tuple[dict[str, list[dict]], Optional[str]]:
    """Group rows by key, falling back to per-row keys for a degenerate group.

    Corpora produced by ``maatml datagen`` / ``maatml ingest`` share one
    ``source`` and carry no ``family``, so every row hashed to the same group
    and the whole corpus landed in a single split (empty val/test, a green
    prepare, and nothing to evaluate on). When one group covers ~the whole
    corpus its members are split per row instead, loudly.

    Returns ``(groups, degenerate_key)``.
    """
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        groups[_group_key(row, group_by=group_by)].append(row)
    if len(rows) < 2:
        return groups, None

    dominant_key, members = max(groups.items(), key=lambda kv: len(kv[1]))
    if len(members) < 2 or len(members) / len(rows) < _DEGENERATE_GROUP_SHARE:
        return groups, None

    message = (
        f"group key {dominant_key!r} covers {len(members)}/{len(rows)} rows: "
        "splitting those rows individually so val/test are not empty. "
        "Group-level leakage protection does NOT apply to them; stamp a "
        "real family/group_by field (seed builders do) to restore it."
    )
    warnings.warn(message, RuntimeWarning, stacklevel=3)
    console.print(f"[yellow]warning[/] {message}")
    del groups[dominant_key]
    for row in members:
        groups[f"{dominant_key}#row:{_row_identity(row)}"].append(row)
    return groups, dominant_key


def _assign_group_splits(
    rows: list[dict],
    ratios: tuple[float, float, float],
    *,
    group_by: Optional[str] = None,
) -> tuple[dict[Split, list[dict]], dict[str, Split], Optional[str]]:
    """Hash each group once; every member inherits that split.

    Returns ``(by_split, group_assignment, degenerate_key)`` so the caller can
    check benchmark rows against the split each group landed in.
    """
    groups, degenerate_key = _group_rows(rows, group_by=group_by)

    by_split: dict[Split, list[dict]] = {
        Split.train: [],
        Split.val: [],
        Split.test: [],
    }
    assignment: dict[str, Split] = {}
    for gkey, members in groups.items():
        split = _split_from_hash(gkey, ratios)
        assignment[gkey] = split
        for row in members:
            tagged = dict(row)
            tagged["split"] = split.value
            by_split[split].append(tagged)
    return by_split, assignment, degenerate_key


def _warn_on_empty_splits(by_split: dict[Split, list[dict]], *, n_groups: int) -> None:
    """Say so when a split came out empty instead of reporting a clean prepare.

    Whole-group hashing with only a handful of groups can leave val (or test)
    with nothing in it. Training then skips evaluation entirely and still
    reports success, so the operator needs to hear about it here.
    """
    empty = [split.value for split, rows in by_split.items() if not rows]
    if not empty or not any(by_split.values()):
        return
    message = (
        f"split(s) {', '.join(sorted(empty))} are empty: {n_groups} group(s) "
        "cannot fill the configured split_ratios. Add more groups (family / "
        "dataset.group_by values) or widen split_ratios; training will skip "
        "evaluation on an empty val split."
    )
    warnings.warn(message, RuntimeWarning, stacklevel=3)
    console.print(f"[yellow]warning[/] {message}")


def _check_benchmark_leakage(
    benchmark_rows: list[dict],
    group_assignment: dict[str, Split],
    *,
    group_by: Optional[str] = None,
) -> None:
    """Refuse to pin a benchmark row whose group already trains.

    Benchmark rows are pinned to test unconditionally. If one shares a group
    key with rows the hash sent to train or val, the same family is on both
    sides of the split and the benchmark number is inflated.
    """
    leaked: dict[str, str] = {}
    for row in benchmark_rows:
        key = _group_key(row, group_by=group_by)
        split = group_assignment.get(key)
        if split in (Split.train, Split.val):
            leaked.setdefault(key, split.value)
    if leaked:
        detail = ", ".join(f"{key} (in {split})" for key, split in sorted(leaked.items()))
        raise ValueError(
            "benchmark_samples share group keys with the training splits: "
            f"{detail}. A benchmark is pinned to test, so those groups would sit "
            "on both sides of the split. Re-tag the benchmark rows (family / "
            "dataset.group_by) or remove them from the seed corpus."
        )


def prepare_rows(
    model_def: ModelDefinition,
    seed_rows: list[dict],
    *,
    out_dir: Optional[Path] = None,
    seed_label: str = "",
    benchmark_rows: Optional[list[dict]] = None,
    benchmark_label: Optional[str] = None,
    sanitize_applied: Optional[list[str]] = None,
) -> dict:
    """Split already-loaded rows and write train/val/test + dataset card.

    Shared by ``jsonl_seed`` and format adapters (alpaca / sharegpt).

    ``sanitize_applied`` is the list of sanitizer tags actually run on the rows
    (the card reports these, not the declared config, so it never claims a
    sanitizer ran when it did not).
    """
    cfg = get_dataset_cfg(model_def)
    out = Path(out_dir) if out_dir else model_def.prepared_dir
    out.mkdir(parents=True, exist_ok=True)
    ratios = _resolve_ratios(cfg)
    group_by = cfg.get("group_by")
    if group_by is not None:
        group_by = str(group_by).strip() or None

    by_split, group_assignment, degenerate_key = _assign_group_splits(
        seed_rows, ratios, group_by=group_by
    )
    category_counts: Counter[str] = Counter()
    source_counts: Counter[str] = Counter()
    family_counts: Counter[str] = Counter()
    for rows in by_split.values():
        for row in rows:
            category_counts[str(row.get("category") or "unknown")] += 1
            source_counts[str(row.get("source") or "unknown")] += 1
            if row.get("family"):
                family_counts[str(row["family"])] += 1

    if benchmark_rows:
        _check_benchmark_leakage(benchmark_rows, group_assignment, group_by=group_by)
        for row in benchmark_rows:
            tagged = dict(row)
            tagged["split"] = Split.test.value
            by_split[Split.test].append(tagged)
            category_counts[str(tagged.get("category") or "unknown")] += 1
            source_counts[str(tagged.get("source") or "unknown")] += 1
            if tagged.get("family"):
                family_counts[str(tagged["family"])] += 1

    # After benchmark pinning: test may be non-empty only because of it.
    _warn_on_empty_splits(by_split, n_groups=len(group_assignment))

    paths = {
        split.value: str(_write_split(rows, out, split)) for split, rows in by_split.items()
    }
    split_counts = {split.value: len(rows) for split, rows in by_split.items()}
    card = _write_dataset_card(
        out,
        title=f"{model_def.identity} dataset",
        counts=dict(category_counts),
        split_counts=split_counts,
        extra_lines=[
            f"Seed source: {seed_label or 'in-memory'}",
            f"Benchmark source: {benchmark_label or 'none'}",
            f"group_by: {group_by or '(default family→source→sample_id)'}",
            f"Sources: {dict(source_counts)}",
            f"Families: {dict(family_counts) if family_counts else '{}'}",
            f"Sanitize: {list(sanitize_applied) if sanitize_applied else 'none'}",
            f"Degenerate group split per row: {degenerate_key or 'none'}",
        ],
    )

    summary = {
        "out_dir": str(out),
        "splits": paths,
        "card": str(card),
        "category_counts": dict(category_counts),
        "source_counts": dict(source_counts),
        "family_counts": dict(family_counts),
        "split_counts": split_counts,
        "degenerate_group": degenerate_key,
    }
    console.print(
        f"[green]prepare complete[/] ({model_def.identity}): {summary['split_counts']} "
        f"(categories: {len(category_counts)})"
    )
    return summary


@register_format("jsonl_seed")
def prepare(model_def: ModelDefinition, out_dir: Optional[Path] = None) -> dict:
    """Generic JSONL-seed prepare: sanitize → group-split → write card.

    Reads ``seed_samples`` / ``benchmark_samples`` / ``split_ratios`` /
    ``sanitize`` / ``request_field`` from ``get_dataset_cfg(model_def)``.
    """
    cfg = get_dataset_cfg(model_def)

    if "seed_samples" not in cfg:
        raise ValueError("model.yml `dataset:`/`data:` must declare `seed_samples`")

    request_field = cfg.get("request_field") or cfg.get("raw_field") or "request"
    sanitize_tags = list(cfg.get("sanitize") or [])

    seed_path = model_def.resolve(cfg["seed_samples"])
    seed_rows: list[dict] = []
    for raw in iter_jsonl(seed_path):
        seed_rows.append(_apply_sanitize(raw, sanitize_tags, request_field))

    bench_rows: list[dict] = []
    benchmark_path = cfg.get("benchmark_samples")
    bench_label = None
    if benchmark_path:
        bench = model_def.resolve(benchmark_path)
        bench_label = str(bench)
        for raw in iter_jsonl(bench):
            bench_rows.append(_apply_sanitize(raw, sanitize_tags, request_field))

    return prepare_rows(
        model_def,
        seed_rows,
        out_dir=out_dir,
        seed_label=str(seed_path),
        benchmark_rows=bench_rows or None,
        benchmark_label=bench_label,
        sanitize_applied=sanitize_tags,
    )
