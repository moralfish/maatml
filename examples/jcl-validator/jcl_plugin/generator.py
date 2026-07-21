from __future__ import annotations

import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterator, Optional

from maatml.data.schemas import Split
from maatml.utils.io import stable_hash

from .schemas import ErrorCategory, JclSample


EXAMPLE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TEMPLATE_DIR = EXAMPLE_ROOT / "datasets" / "templates"

JOBNAMES = ["MYJOB001", "ACCTBTCH", "RPTGEN01", "DLYLOAD2", "ETLEXTR3", "RECONJOB"]
ACCTS = ["123456", "555000", "789012", "AB12CD"]
OWNERS = ["ACCTING", "RPTTEAM", "ETLOPS", "DBA", "OPSCREW"]
PGMS = ["IEFBR14", "IEBGENER", "IDCAMS", "SORT", "ICETOOL", "IKJEFT01"]
PGM2S = ["IEBCOPY", "IEHLIST", "IDCAMS", "SORT"]
DSNS = [
    "USER.MY.INPUT",
    "PROD.DAILY.LOAD",
    "ETL.STG.RAW",
    "RPT.OUTPUT.YEAR",
    "TEST.WORK.FILE",
    "ACCT.MASTER.IDX",
    "DBA.UTIL.LIB",
]
LIBS = ["MY.LOAD.LIB", "PROD.LINKLIB", "DEV.PGM.LIB"]
MEMS = ["MAIN", "DRIVER", "REPORT", "EXTRACT"]
PROCS = ["MYPROC", "ETLPROC", "RPTPROC"]


PLACEHOLDER_POOLS: dict[str, list[str]] = {
    "JOBNAME": JOBNAMES,
    "ACCT": ACCTS,
    "OWNER": OWNERS,
    "PGM": PGMS,
    "PGM2": PGM2S,
    "DSN1": DSNS,
    "DSN2": DSNS,
    "LIB": LIBS,
    "MEM": MEMS,
    "PROC": PROCS,
}

PLACEHOLDER_RE = re.compile(r"\{\{([A-Z][A-Z0-9_]*)\}\}")


@dataclass(frozen=True)
class Defect:
    category: ErrorCategory
    inject: Callable[[random.Random, list[str]], Optional[tuple[list[str], int, Optional[int], str]]]


def _render_template(rng: random.Random, text: str) -> str:
    seen: dict[str, str] = {}

    def _pick(m: re.Match[str]) -> str:
        name = m.group(1)
        if name in seen:
            return seen[name]
        pool = PLACEHOLDER_POOLS.get(name)
        value = rng.choice(pool) if pool else name
        seen[name] = value
        return value

    return PLACEHOLDER_RE.sub(_pick, text)


def _find_indexes(lines: list[str], predicate: Callable[[str], bool]) -> list[int]:
    return [i for i, ln in enumerate(lines) if predicate(ln)]


def _is_dd_line(ln: str) -> bool:
    return bool(re.match(r"//[A-Z@#$][A-Z0-9@#$]{0,7}\s+DD\b", ln))


def _is_exec_line(ln: str) -> bool:
    return bool(re.match(r"//[A-Z@#$][A-Z0-9@#$]{0,7}\s+EXEC\b", ln))


def _is_continuation_target(lines: list[str], i: int) -> bool:
    if i == 0:
        return False
    prev = lines[i - 1].rstrip()
    if not prev.endswith(","):
        return False
    return lines[i].startswith("//") and re.match(r"//\s+\S", lines[i]) is not None


def inject_missing_dd(rng: random.Random, lines: list[str]) -> Optional[tuple[list[str], int, Optional[int], str]]:
    candidates = [i for i in _find_indexes(lines, _is_dd_line) if "SYSPRINT" not in lines[i]]
    if not candidates:
        return None
    idx = rng.choice(candidates)
    original = lines[idx]
    m = re.match(r"(//[A-Z@#$][A-Z0-9@#$]{0,7}\s+)DD\s+(.*)", original)
    if not m:
        return None
    new_line = f"{m.group(1)}   {m.group(2)}"
    new_lines = lines.copy()
    new_lines[idx] = new_line
    label_match = re.match(r"//([A-Z@#$][A-Z0-9@#$]{0,7})", original)
    ddname = label_match.group(1) if label_match else "DD"
    return new_lines, idx + 1, 1, f"Add 'DD' keyword for {ddname}"


def inject_invalid_job_card(rng: random.Random, lines: list[str]) -> Optional[tuple[list[str], int, Optional[int], str]]:
    job_idx = next((i for i, ln in enumerate(lines) if " JOB " in ln), None)
    if job_idx is None:
        return None
    variant = rng.choice(["drop_class", "malform_parens", "blank_account"])
    original = lines[job_idx]
    if variant == "drop_class":
        new_line = re.sub(r",CLASS=[A-Z0-9]", "", original, count=1)
    elif variant == "malform_parens":
        new_line = original.replace("(", "", 1)
    else:
        new_line = re.sub(r"JOB\s+\([^)]*\)", "JOB ()", original, count=1)
    if new_line == original:
        return None
    new_lines = lines.copy()
    new_lines[job_idx] = new_line
    return new_lines, job_idx + 1, 1, "Fix the JOB card (account, class, parameter syntax)"


def inject_unresolved_symbolic_parameter(rng: random.Random, lines: list[str]) -> Optional[tuple[list[str], int, Optional[int], str]]:
    candidates = [i for i, ln in enumerate(lines) if "DSN=" in ln and "&" not in ln]
    if not candidates:
        return None
    idx = rng.choice(candidates)
    original = lines[idx]
    new_line = re.sub(r"DSN=[A-Z0-9.]+", "DSN=&UNDEF", original, count=1)
    if new_line == original:
        return None
    col = new_line.find("&UNDEF") + 1
    new_lines = lines.copy()
    new_lines[idx] = new_line
    return new_lines, idx + 1, col, "Define &UNDEF via // SET or remove the symbolic reference"


def inject_continuation_error(rng: random.Random, lines: list[str]) -> Optional[tuple[list[str], int, Optional[int], str]]:
    candidates = [i for i in range(1, len(lines)) if _is_continuation_target(lines, i)]
    if not candidates:
        return None
    idx = rng.choice(candidates)
    original = lines[idx]
    new_line = "  " + original[2:] if original.startswith("//") else original
    if new_line == original:
        return None
    new_lines = lines.copy()
    new_lines[idx] = new_line
    return new_lines, idx + 1, 1, "Continuation lines must begin with // in cols 1-2"


def inject_invalid_exec_statement(rng: random.Random, lines: list[str]) -> Optional[tuple[list[str], int, Optional[int], str]]:
    candidates = _find_indexes(lines, _is_exec_line)
    if not candidates:
        return None
    idx = rng.choice(candidates)
    original = lines[idx]
    variant = rng.choice(["wrong_keyword", "drop_pgm"])
    if variant == "wrong_keyword" and "PGM=" in original:
        new_line = original.replace("PGM=", "PROG=", 1)
        col = new_line.find("PROG=") + 1
        suggestion = "Use PGM= or PROC= on EXEC statements"
    elif "PGM=" in original:
        new_line = original.replace("PGM=", "", 1)
        col = original.find("PGM=") + 1
        suggestion = "EXEC requires PGM= or PROC="
    else:
        return None
    if new_line == original:
        return None
    new_lines = lines.copy()
    new_lines[idx] = new_line
    return new_lines, idx + 1, col, suggestion


def inject_invalid_dataset_reference_structure(rng: random.Random, lines: list[str]) -> Optional[tuple[list[str], int, Optional[int], str]]:
    candidates = [i for i, ln in enumerate(lines) if re.search(r"DSN=[A-Z0-9.]+", ln)]
    if not candidates:
        return None
    idx = rng.choice(candidates)
    original = lines[idx]
    bad = rng.choice(["BAD..NAME..1", "1BAD.NAME", "BAD.NAME.", "BAD@@NAME"])
    new_line = re.sub(r"DSN=[A-Z0-9.]+", f"DSN={bad}", original, count=1)
    col = new_line.find("DSN=") + 5
    new_lines = lines.copy()
    new_lines[idx] = new_line
    return new_lines, idx + 1, col, "Dataset names must be 1-8 char qualifiers separated by dots, starting with a letter"


def inject_other(rng: random.Random, lines: list[str]) -> Optional[tuple[list[str], int, Optional[int], str]]:
    candidates = [i for i, ln in enumerate(lines) if ln.startswith("//") and " " in ln[2:]]
    if not candidates:
        return None
    idx = rng.choice(candidates)
    original = lines[idx]
    new_line = " " + original[1:]
    new_lines = lines.copy()
    new_lines[idx] = new_line
    return new_lines, idx + 1, 1, "Statement labels must begin in column 1 with //"


INJECTORS: dict[ErrorCategory, Defect] = {
    ErrorCategory.missing_dd: Defect(ErrorCategory.missing_dd, inject_missing_dd),
    ErrorCategory.invalid_job_card: Defect(ErrorCategory.invalid_job_card, inject_invalid_job_card),
    ErrorCategory.unresolved_symbolic_parameter: Defect(ErrorCategory.unresolved_symbolic_parameter, inject_unresolved_symbolic_parameter),
    ErrorCategory.continuation_error: Defect(ErrorCategory.continuation_error, inject_continuation_error),
    ErrorCategory.invalid_exec_statement: Defect(ErrorCategory.invalid_exec_statement, inject_invalid_exec_statement),
    ErrorCategory.invalid_dataset_reference_structure: Defect(ErrorCategory.invalid_dataset_reference_structure, inject_invalid_dataset_reference_structure),
    ErrorCategory.other: Defect(ErrorCategory.other, inject_other),
}


def _load_templates(template_dir: Path) -> list[tuple[str, str]]:
    paths = sorted(template_dir.glob("*.jcl"))
    if not paths:
        raise FileNotFoundError(f"No .jcl templates found in {template_dir}")
    return [(p.stem, p.read_text(encoding="utf-8")) for p in paths]


def generate_one(
    rng: random.Random,
    templates: list[tuple[str, str]],
    category: Optional[ErrorCategory],
    *,
    seed: int,
    idx: int,
    split: Split,
) -> Optional[JclSample]:
    template_id, raw = rng.choice(templates)
    rendered = _render_template(rng, raw)
    lines = rendered.splitlines()

    if category is None or category is ErrorCategory.none:
        sid = f"syn-valid-{stable_hash(template_id, seed, idx)[:8]}"
        return JclSample(  # type: ignore[call-arg]
            sample_id=sid,
            source=f"synthetic:{template_id}",
            sanitized_jcl="\n".join(lines) + "\n",
            is_valid=True,
            error_category=ErrorCategory.none,
            error_line=None,
            error_column=None,
            suggestion=None,
            split=split,
        )

    injector = INJECTORS[category]
    result = injector.inject(rng, lines)
    if result is None:
        return None
    new_lines, error_line, error_column, suggestion = result
    sid = f"syn-{category.value}-{stable_hash(template_id, seed, idx)[:8]}"
    return JclSample(  # type: ignore[call-arg]
        sample_id=sid,
        source=f"synthetic:{template_id}",
        sanitized_jcl="\n".join(new_lines) + "\n",
        is_valid=False,
        error_category=category,
        error_line=error_line,
        error_column=error_column,
        suggestion=suggestion,
        split=split,
    )


def _split_for(rng: random.Random, ratios: tuple[float, float, float]) -> Split:
    train_r, val_r, _ = ratios
    r = rng.random()
    if r < train_r:
        return Split.train
    if r < train_r + val_r:
        return Split.val
    return Split.test


def generate_corpus(
    *,
    seed: int,
    n_per_class: dict[ErrorCategory, int],
    n_valid: int,
    template_dir: str | Path | None = None,
    split_ratios: tuple[float, float, float] = (0.8, 0.1, 0.1),
) -> Iterator[JclSample]:
    rng = random.Random(seed)
    templates = _load_templates(Path(template_dir) if template_dir else DEFAULT_TEMPLATE_DIR)
    counter = 0

    for category, n in n_per_class.items():
        produced = 0
        attempts = 0
        max_attempts = n * 10
        while produced < n and attempts < max_attempts:
            sample = generate_one(
                rng,
                templates,
                category,
                seed=seed,
                idx=counter,
                split=_split_for(rng, split_ratios),
            )
            counter += 1
            attempts += 1
            if sample is not None:
                produced += 1
                yield sample

    for _ in range(n_valid):
        sample = generate_one(
            rng,
            templates,
            None,
            seed=seed,
            idx=counter,
            split=_split_for(rng, split_ratios),
        )
        counter += 1
        if sample is not None:
            yield sample
