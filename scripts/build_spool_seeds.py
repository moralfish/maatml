"""Build the Spool Interpreter seed corpus deterministically.

Produces a balanced corpus across the 13 categories defined in
`models/spool-interpreter/datasets/node_contracts.json`:

    completed
    dataset_resolution_failure        allocation_failure
    permission_or_security_failure    jcl_syntax_failure
    utility_parameter_failure         execution_abend
    scheduler_or_environment_issue    other
    smart_restart_resource_unavailable    smart_restart_configuration
    smart_restart_application_logic       smart_restart_input_syntax

Each category has 2-4 spool message templates with parametric slots
(dataset names, return codes, system codes, line numbers, …). Every
generated sample is gated by the 6-layer `validate_spool_result` check
before being written.

No API calls. Run anytime to regenerate or extend the corpus.

Usage:
    python scripts/build_spool_seeds.py             # default 500 samples
    python scripts/build_spool_seeds.py --target 800
    python scripts/build_spool_seeds.py --append    # keep existing rows
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from flow_ml.validation import validate_spool_result  # noqa: E402


MODEL_DIR = REPO / "models" / "spool-interpreter"
DATASETS = MODEL_DIR / "datasets"
SCHEMA_PATH = DATASETS / "spool_interpretation_schema.json"
CONTRACTS_PATH = DATASETS / "node_contracts.json"
SEEDS_PATH = DATASETS / "samples" / "seed_samples.jsonl"


# Pools used by parametric templates.  All values are sanitized.
JOBNAMES = ["MYJOB001", "ACCTBTCH", "RPTGEN01", "DLYLOAD2", "ETLEXTR3", "RECONJOB", "DBA0001", "BKUPJOB"]
JOBIDS = ["JOB04211", "JOB05172", "JOB06033", "JOB07788", "JOB08120", "JOB09445"]
STEPNAMES = ["STEP01", "STEP02", "LOAD", "SORT", "EXTRACT", "BUILD", "REPORT", "BACKUP"]
DSNS = [
    "USER.MY.INPUT", "PROD.DAILY.LOAD", "ETL.STG.RAW", "RPT.OUTPUT.YEAR",
    "TEST.WORK.FILE", "ACCT.MASTER.IDX", "DBA.UTIL.LIB", "PROD.SORT.IN",
    "PROD.SORT.OUT", "USER.LOAD.LIB", "DEV.PGM.LIB", "ARCHIVE.MONTHLY",
]
USERS = ["TSOU01", "BATCH02", "OPSADM", "DBA01", "RPTRUN"]
ABEND_CODES = ["S0C7", "S0C4", "S322", "S806", "S913", "S013", "S04E"]
RC_OK = ["0000", "0004"]
RC_WARN = ["0008", "000C"]
RC_ERR = ["0010", "0012", "0016", "0020", "012C"]


def _pick(rng: random.Random, pool: list[str]) -> str:
    return rng.choice(pool)


def _slot(text: str, **subs: str) -> str:
    for k, v in subs.items():
        text = text.replace("{" + k + "}", v)
    return text


def _hash(*parts: object) -> str:
    h = hashlib.sha256("|".join(str(p) for p in parts).encode()).hexdigest()
    return h[:8]


# ---------------------------------------------------------------------------
# Per-category builders. Each returns (request, expected_interpretation).
# ---------------------------------------------------------------------------


def build_completed(rng: random.Random) -> tuple[str, dict]:
    job = _pick(rng, JOBNAMES)
    jobid = _pick(rng, JOBIDS)
    step = _pick(rng, STEPNAMES)
    rc = _pick(rng, RC_OK + RC_WARN)
    request = (
        f"$HASP373 {job}    STARTED - INIT 1   - CLASS A - SYS SYS1\n"
        f"IEF403I {job} - STARTED - TIME=09.{rng.randint(10, 59)}.{rng.randint(10, 59)}\n"
        f"IEF142I {job} {step}        - STEP WAS EXECUTED - COND CODE {rc[-4:]}\n"
        f"IEF373I STEP/{step:<8}/START 2026{rng.randint(100, 360):03d}.0915\n"
        f"IEF374I STEP/{step:<8}/STOP  2026{rng.randint(100, 360):03d}.0917\n"
        f"$HASP395 {job}    ENDED - RC={rc}\n"
    )
    severity = "successfully" if rc == "0000" else "with warnings"
    interp = {
        "summary": f"Job {job} completed {severity} (RC={rc}).",
        "status": "completed",
        "returnCode": rc,
        "rootCause": "Step finished within expected return-code envelope; no failure observed.",
        "suggestedFix": "No action required.",
        "failureCategory": None if rc in RC_OK else "other",
        "confidence": round(rng.uniform(0.92, 0.98), 2),
    }
    return request, interp


def build_dataset_resolution_failure(rng: random.Random) -> tuple[str, dict]:
    job = _pick(rng, JOBNAMES)
    step = _pick(rng, STEPNAMES)
    dsn = _pick(rng, DSNS)
    rc = _pick(rng, RC_ERR)
    request = (
        f"$HASP373 {job}    STARTED\n"
        f"IEF212I {job} {step} {dsn}        - DATA SET NOT FOUND\n"
        f"IEF272I {job} {step} - STEP WAS NOT EXECUTED.\n"
        f"IEF142I {job} {step}        - STEP WAS NOT EXECUTED - COND CODE {rc[-4:]}\n"
        f"$HASP395 {job}    ENDED - RC={rc}\n"
    )
    interp = {
        "summary": f"Job {job} step {step} failed: dataset {dsn} not found in catalog.",
        "status": "failed",
        "returnCode": rc,
        "rootCause": (
            f"The catalog has no entry for {dsn}; either it was deleted, "
            f"never created, or referenced under the wrong qualifier."
        ),
        "suggestedFix": (
            f"Confirm the dataset name; if intentional, allocate it before "
            f"this step or restore from backup."
        ),
        "failureCategory": "dataset_resolution_failure",
        "confidence": round(rng.uniform(0.88, 0.96), 2),
    }
    return request, interp


def build_allocation_failure(rng: random.Random) -> tuple[str, dict]:
    job = _pick(rng, JOBNAMES)
    step = _pick(rng, STEPNAMES)
    dsn = _pick(rng, DSNS)
    rc = _pick(rng, RC_ERR)
    variant = rng.choice(["space", "volume", "unit"])
    if variant == "space":
        msg = f"IEF257I {job} {step} - SPACE REQUESTED NOT AVAILABLE FOR {dsn}"
        root = (
            f"Requested SPACE for {dsn} exceeds free extents on the target "
            f"volume; the allocator could not satisfy the primary quantity."
        )
        fix = "Reduce SPACE=, target a larger volume, or release stale allocations."
    elif variant == "volume":
        msg = f"IEF863I VOLUME UNAVAILABLE FOR {dsn}"
        root = "Required volume is offline or has been varied off the system."
        fix = "Vary the volume online or re-route the dataset to an available volume."
    else:
        msg = f"IEF210I {job} {step} - UNIT NOT AVAILABLE FOR {dsn}"
        root = "Unit was not available at allocation time (busy or offline)."
        fix = "Wait for the unit to drain or use a generic unit name (SYSDA)."
    request = (
        f"$HASP373 {job}    STARTED\n"
        f"{msg}\n"
        f"IEF272I {job} {step} - STEP WAS NOT EXECUTED.\n"
        f"$HASP395 {job}    ENDED - RC={rc}\n"
    )
    interp = {
        "summary": f"Job {job} step {step} failed: allocation could not be satisfied for {dsn}.",
        "status": "failed",
        "returnCode": rc,
        "rootCause": root,
        "suggestedFix": fix,
        "failureCategory": "allocation_failure",
        "confidence": round(rng.uniform(0.86, 0.94), 2),
    }
    return request, interp


def build_permission_or_security_failure(rng: random.Random) -> tuple[str, dict]:
    job = _pick(rng, JOBNAMES)
    step = _pick(rng, STEPNAMES)
    dsn = _pick(rng, DSNS)
    user = _pick(rng, USERS)
    rc = _pick(rng, RC_ERR)
    request = (
        f"$HASP373 {job}    STARTED USER={user}\n"
        f"ICH408I USER({user}) GROUP() NAME()\n"
        f"  {dsn} CL(DATASET)\n"
        f"  INSUFFICIENT ACCESS AUTHORITY\n"
        f"  ACCESS INTENT(UPDATE) ACCESS ALLOWED(NONE)\n"
        f"IEF272I {job} {step} - STEP WAS NOT EXECUTED.\n"
        f"$HASP395 {job}    ENDED - RC={rc}\n"
    )
    interp = {
        "summary": f"Job {job} step {step} blocked by RACF: {user} has no UPDATE access to {dsn}.",
        "status": "failed",
        "returnCode": rc,
        "rootCause": (
            f"RACF profile for {dsn} grants {user} no UPDATE authority; "
            f"the access-intent check failed before allocation."
        ),
        "suggestedFix": (
            f"Grant {user} UPDATE access via PERMIT, run under an authorised "
            f"id, or change the step to use a dataset {user} can write."
        ),
        "failureCategory": "permission_or_security_failure",
        "confidence": round(rng.uniform(0.90, 0.97), 2),
    }
    return request, interp


def build_jcl_syntax_failure(rng: random.Random) -> tuple[str, dict]:
    job = _pick(rng, JOBNAMES)
    step = _pick(rng, STEPNAMES)
    rc = _pick(rng, ["000C", "0010", "0012"])
    bad_line = rng.randint(2, 8)
    variant = rng.choice(["unbalanced_paren", "unknown_keyword", "missing_pgm"])
    if variant == "unbalanced_paren":
        diag = "IEFC630I  UNBALANCED PARENTHESES"
        root = "Unbalanced parentheses in a parameter list; the JCL converter could not tokenise the statement."
        fix = "Match opening and closing parentheses around the offending parameter."
    elif variant == "unknown_keyword":
        diag = "IEFC662I  UNIDENTIFIED OPERATION FIELD"
        root = "An EXEC or DD operand uses a keyword the JCL converter does not recognise."
        fix = "Replace the unknown keyword with a documented one (PGM=, PROC=, DSN=, DISP=, …)."
    else:
        diag = "IEFC662I  PGM= OR PROC= MUST BE SPECIFIED"
        root = "EXEC card is missing the mandatory PGM= or PROC= operand."
        fix = "Add PGM=<program> or PROC=<proc-name> to the EXEC statement."
    request = (
        f"$HASP373 {job}    STARTED\n"
        f"STMT NO. MESSAGE\n"
        f"   {bad_line:<5} {diag}\n"
        f"IEF453I {job} - JOB FAILED - JCL ERROR - TIME=09.{rng.randint(10, 59)}.{rng.randint(10, 59)}\n"
        f"$HASP395 {job}    ENDED - RC={rc}\n"
    )
    interp = {
        "summary": f"Job {job} failed JCL conversion at statement {bad_line}.",
        "status": "failed",
        "returnCode": rc,
        "rootCause": root,
        "suggestedFix": fix,
        "failureCategory": "jcl_syntax_failure",
        "confidence": round(rng.uniform(0.87, 0.95), 2),
    }
    return request, interp


def build_utility_parameter_failure(rng: random.Random) -> tuple[str, dict]:
    job = _pick(rng, JOBNAMES)
    step = _pick(rng, STEPNAMES)
    rc = _pick(rng, ["0010", "0012", "0016"])
    variant = rng.choice(["sort", "iebgener", "idcams"])
    if variant == "sort":
        diag = (
            "ICE140I 0 SORT FIELDS=(1,8,CH,A) WAS SPECIFIED INVALIDLY\n"
            "ICE000I 1   SORT TERMINATED - INVALID CONTROL STATEMENT"
        )
        root = "DFSORT control statement specifies SORT FIELDS with offsets that exceed the input record length."
        fix = "Recheck record length; ensure the (start, length) pairs fit within the LRECL."
    elif variant == "iebgener":
        diag = (
            "IEB369I IEBGENER STARTED ON {date}\n"
            "IEB344I CONTROL CARD ERROR ON SYSIN\n"
            "IEB347I IEBGENER ENDED - INVALID CONTROL CARD"
        ).replace("{date}", "2026.121")
        root = "IEBGENER SYSIN control card has an unrecognised keyword or malformed continuation."
        fix = "Review the GENERATE/MEMBER cards on SYSIN; conform to the IEBGENER manual reference."
    else:
        diag = (
            "IDC0002I IDCAMS PROCESSING WAS COMPLETED WITH MAXIMUM CONDITION CODE 12\n"
            "IDC3210I  ** OPEN ERROR ON SYSIN"
        )
        root = "IDCAMS rejected the SYSIN deck due to a parameter contradiction (e.g. CLUSTER + REUSE)."
        fix = "Reconcile mutually exclusive parameters and resubmit."
    request = (
        f"$HASP373 {job}    STARTED\n"
        f"{diag}\n"
        f"IEF142I {job} {step}        - STEP WAS EXECUTED - COND CODE {rc[-4:]}\n"
        f"$HASP395 {job}    ENDED - RC={rc}\n"
    )
    interp = {
        "summary": f"Job {job} step {step} failed because the utility rejected its control statements.",
        "status": "failed",
        "returnCode": rc,
        "rootCause": root,
        "suggestedFix": fix,
        "failureCategory": "utility_parameter_failure",
        "confidence": round(rng.uniform(0.84, 0.93), 2),
    }
    return request, interp


def build_execution_abend(rng: random.Random) -> tuple[str, dict]:
    job = _pick(rng, JOBNAMES)
    step = _pick(rng, STEPNAMES)
    abend = _pick(rng, ABEND_CODES)
    rc = None  # abends don't expose a normal RC
    cause_map = {
        "S0C7": "Data exception (numeric field used as character or vice versa).",
        "S0C4": "Storage protect / addressing exception (out-of-bounds memory access).",
        "S322": "Time exceeded — job CPU time limit reached before completion.",
        "S806": "Module not found in any library on STEPLIB / JOBLIB / linklist.",
        "S913": "Insufficient access authority to a dataset opened during the step.",
        "S013": "Open of a sequential dataset failed (LRECL/BLKSIZE mismatch).",
        "S04E": "DB2 thread abend — typically resource unavailable or rolled-back transaction.",
    }
    fix_map = {
        "S0C7": "Validate input data with a NUMERIC test before the COMPUTE / arithmetic call.",
        "S0C4": "Trace pointer use; ensure arrays and tables are bounded; rebuild with debugging symbols if needed.",
        "S322": "Increase JOB or STEP TIME=, or split the workload into smaller chunks.",
        "S806": "Add the required load library to STEPLIB; confirm the program name has not been retired.",
        "S913": "Permit the executing user via RACF or run under an authorised id.",
        "S013": "Match LRECL/BLKSIZE between JCL and dataset attributes.",
        "S04E": "Inspect DB2 master log; verify required objects are accessible and within bind plan.",
    }
    request = (
        f"$HASP373 {job}    STARTED\n"
        f"IEA995I SYMPTOM DUMP OUTPUT\n"
        f"  SYSTEM COMPLETION CODE={abend}\n"
        f"  TIME=09.{rng.randint(10, 59)}.{rng.randint(10, 59)} JOBNAME={job} STEPNAME={step}\n"
        f"IEF472I {job} {step} - COMPLETION CODE - SYSTEM={abend}\n"
        f"$HASP395 {job}    ENDED\n"
    )
    interp = {
        "summary": f"Job {job} step {step} abended with system code {abend}.",
        "status": "abended",
        "returnCode": rc,
        "rootCause": cause_map[abend],
        "suggestedFix": fix_map[abend],
        "failureCategory": "execution_abend",
        "confidence": round(rng.uniform(0.85, 0.95), 2),
    }
    return request, interp


def build_scheduler_or_environment_issue(rng: random.Random) -> tuple[str, dict]:
    job = _pick(rng, JOBNAMES)
    rc = _pick(rng, RC_ERR)
    variant = rng.choice(["init_class", "submit_held", "smfid"])
    if variant == "init_class":
        diag = (
            f"$HASP110 {job}    NO ELIGIBLE INITIATORS FOR CLASS Z\n"
            f"$HASP190 {job}    HELD AT JOB QUEUE EXIT"
        )
        root = "No initiators are configured for the requested CLASS; the job sat on the input queue."
        fix = "Submit under a class with active initiators or ask the scheduler to start one for class Z."
    elif variant == "submit_held":
        diag = (
            f"$HASP373 {job}    STARTED\n"
            f"IAT5210 JOB {job} HAS BEEN PURGED BY SCHEDULER (HOLD EXPIRED)"
        )
        root = "Scheduler purged the job after its hold window expired without release."
        fix = "Resubmit the job and ensure the operations team releases it within the SLA window."
    else:
        diag = (
            f"$HASP373 {job}    STARTED\n"
            f"IEF196I SMF EXIT FAILED - JOB CANCELLED\n"
            f"IEF453I {job} - JOB FAILED - JCL ERROR"
        )
        root = "SMF environment exit signalled an installation policy violation and cancelled the job before step start."
        fix = "Review the SMF exit configuration with systems programming; resubmit once cleared."
    request = (
        f"{diag}\n"
        f"$HASP395 {job}    ENDED - RC={rc}\n"
    )
    interp = {
        "summary": f"Job {job} failed before any step ran due to a scheduler/environment condition.",
        "status": "failed",
        "returnCode": rc,
        "rootCause": root,
        "suggestedFix": fix,
        "failureCategory": "scheduler_or_environment_issue",
        "confidence": round(rng.uniform(0.82, 0.92), 2),
    }
    return request, interp


def build_other(rng: random.Random) -> tuple[str, dict]:
    job = _pick(rng, JOBNAMES)
    step = _pick(rng, STEPNAMES)
    rc = _pick(rng, RC_ERR + RC_WARN)
    diag_msg = rng.choice([
        "Application returned a non-zero condition code without a recognised diagnostic.",
        "Step was flushed by a prior step's COND= and reported an unusual code.",
        "Externally-orchestrated dependency (FTP, MQ, …) reported a soft failure that propagated here.",
    ])
    request = (
        f"$HASP373 {job}    STARTED\n"
        f"IEF142I {job} {step}        - STEP WAS EXECUTED - COND CODE {rc[-4:]}\n"
        f"$HASP395 {job}    ENDED - RC={rc}\n"
    )
    interp = {
        "summary": f"Job {job} step {step} ended with an inconclusive failure (RC={rc}).",
        "status": "failed",
        "returnCode": rc,
        "rootCause": diag_msg,
        "suggestedFix": "Open the step's joblog and any application-side logs to identify the underlying error.",
        "failureCategory": "other",
        "confidence": round(rng.uniform(0.70, 0.85), 2),
    }
    return request, interp


# Smart Restart families — taken from flow-studio's Smart Restart taxonomy.


def build_smart_restart_resource_unavailable(rng: random.Random) -> tuple[str, dict]:
    job = _pick(rng, JOBNAMES)
    step = _pick(rng, STEPNAMES)
    rc = _pick(rng, RC_ERR)
    res = rng.choice(["CICS region", "DB2 subsystem", "MQ queue manager", "external FTP server"])
    request = (
        f"$HASP373 {job}    STARTED\n"
        f"SMART-RESTART  EVAL  STEP={step}\n"
        f"SMART-RESTART  CATEGORY=resource_unavailable\n"
        f"SMART-RESTART  DETAIL={res} not reachable; retry advised.\n"
        f"IEF142I {job} {step}        - STEP WAS NOT EXECUTED - COND CODE {rc[-4:]}\n"
        f"$HASP395 {job}    ENDED - RC={rc}\n"
    )
    interp = {
        "summary": f"Smart Restart flagged step {step} as blocked by {res}.",
        "status": "failed",
        "returnCode": rc,
        "rootCause": f"Required external resource ({res}) was unavailable at step start; not a code defect.",
        "suggestedFix": f"Confirm {res} health, then resubmit; Smart Restart can re-drive the step automatically.",
        "failureCategory": "smart_restart_resource_unavailable",
        "confidence": round(rng.uniform(0.85, 0.93), 2),
    }
    return request, interp


def build_smart_restart_configuration(rng: random.Random) -> tuple[str, dict]:
    job = _pick(rng, JOBNAMES)
    step = _pick(rng, STEPNAMES)
    rc = _pick(rng, RC_ERR)
    knob = rng.choice(["MAXCC", "DSN ALIAS", "STEPLIB PROTECT", "SYMBOLIC PARM"])
    request = (
        f"$HASP373 {job}    STARTED\n"
        f"SMART-RESTART  EVAL  STEP={step}\n"
        f"SMART-RESTART  CATEGORY=configuration\n"
        f"SMART-RESTART  DETAIL={knob} setting in the deployed JCL contradicts the runtime catalog.\n"
        f"IEF142I {job} {step}        - STEP WAS NOT EXECUTED - COND CODE {rc[-4:]}\n"
        f"$HASP395 {job}    ENDED - RC={rc}\n"
    )
    interp = {
        "summary": f"Smart Restart traced step {step} failure to a configuration mismatch ({knob}).",
        "status": "failed",
        "returnCode": rc,
        "rootCause": f"The {knob} value baked into the JCL no longer matches the active catalog/runtime profile.",
        "suggestedFix": f"Update the JCL or catalog so {knob} matches; redeploy and resubmit.",
        "failureCategory": "smart_restart_configuration",
        "confidence": round(rng.uniform(0.84, 0.93), 2),
    }
    return request, interp


def build_smart_restart_application_logic(rng: random.Random) -> tuple[str, dict]:
    job = _pick(rng, JOBNAMES)
    step = _pick(rng, STEPNAMES)
    rc = _pick(rng, RC_ERR)
    detail = rng.choice([
        "Cursor returned zero rows where the program asserts at least one.",
        "COMPUTE result truncated past the receiving field's PIC clause.",
        "Conditional branch reached an UNREACHABLE paragraph and EXIT'd with non-zero RC.",
    ])
    request = (
        f"$HASP373 {job}    STARTED\n"
        f"SMART-RESTART  EVAL  STEP={step}\n"
        f"SMART-RESTART  CATEGORY=application_logic\n"
        f"SMART-RESTART  DETAIL={detail}\n"
        f"IEF142I {job} {step}        - STEP WAS EXECUTED - COND CODE {rc[-4:]}\n"
        f"$HASP395 {job}    ENDED - RC={rc}\n"
    )
    interp = {
        "summary": f"Smart Restart attributed step {step} failure to application logic.",
        "status": "failed",
        "returnCode": rc,
        "rootCause": detail,
        "suggestedFix": "Code change is required; Smart Restart will not auto-retry. Hand off to the application team.",
        "failureCategory": "smart_restart_application_logic",
        "confidence": round(rng.uniform(0.80, 0.91), 2),
    }
    return request, interp


def build_smart_restart_input_syntax(rng: random.Random) -> tuple[str, dict]:
    job = _pick(rng, JOBNAMES)
    step = _pick(rng, STEPNAMES)
    rc = _pick(rng, RC_ERR)
    src = rng.choice(["control card", "SYSIN deck", "input file header", "PARM string"])
    request = (
        f"$HASP373 {job}    STARTED\n"
        f"SMART-RESTART  EVAL  STEP={step}\n"
        f"SMART-RESTART  CATEGORY=input_syntax\n"
        f"SMART-RESTART  DETAIL={src} did not match the parser grammar; first divergent token highlighted.\n"
        f"IEF142I {job} {step}        - STEP WAS EXECUTED - COND CODE {rc[-4:]}\n"
        f"$HASP395 {job}    ENDED - RC={rc}\n"
    )
    interp = {
        "summary": f"Smart Restart blamed step {step} on malformed {src}.",
        "status": "failed",
        "returnCode": rc,
        "rootCause": f"The supplied {src} violates the program's expected grammar.",
        "suggestedFix": f"Fix the {src}, validate against the spec, and resubmit.",
        "failureCategory": "smart_restart_input_syntax",
        "confidence": round(rng.uniform(0.82, 0.92), 2),
    }
    return request, interp


CATEGORY_BUILDERS = {
    "completed": build_completed,
    "dataset_resolution_failure": build_dataset_resolution_failure,
    "allocation_failure": build_allocation_failure,
    "permission_or_security_failure": build_permission_or_security_failure,
    "jcl_syntax_failure": build_jcl_syntax_failure,
    "utility_parameter_failure": build_utility_parameter_failure,
    "execution_abend": build_execution_abend,
    "scheduler_or_environment_issue": build_scheduler_or_environment_issue,
    "other": build_other,
    "smart_restart_resource_unavailable": build_smart_restart_resource_unavailable,
    "smart_restart_configuration": build_smart_restart_configuration,
    "smart_restart_application_logic": build_smart_restart_application_logic,
    "smart_restart_input_syntax": build_smart_restart_input_syntax,
}


def _enrich_v2_fields(
    rng: random.Random,
    category: str,
    interp: dict,
    related_docs_catalog: dict[str, list[str]],
) -> None:
    """Stamp v2-only fields (`explanation`, `relatedDocs`) onto an interp
    record in place.

    Narrative is composed deterministically from fields already present
    on the interp so it stays faithful to the synthetic spool: a 2-3
    sentence walkthrough that opens with status/step, recaps the root
    cause, and closes with the operator-facing fix. This avoids needing
    a separate template per category (13 builders + maintenance burden).
    """
    summary = interp.get("summary", "")
    root = interp.get("rootCause", "")
    fix = interp.get("suggestedFix", "")
    status = interp.get("status", "completed")

    if status == "completed":
        interp["explanation"] = (
            f"Execution completed under the expected return-code envelope. "
            f"{summary} No remediation is required; downstream steps may "
            f"proceed."
        )
    else:
        # 2-3 sentence narrative: situation → cause → fix.
        opener = rng.choice([
            f"The job entered execution and progressed until the failure surfaced.",
            f"Execution started normally and ran up to the point of failure.",
            f"The step began processing and then halted on the condition described below.",
        ])
        interp["explanation"] = (
            f"{opener} {root} {fix}"
        )

    pool = related_docs_catalog.get(category, [])
    if not pool:
        interp["relatedDocs"] = []
    else:
        # Pick 1-3 doc keys per sample for stable training signal.
        k = min(len(pool), rng.randint(1, 3))
        interp["relatedDocs"] = rng.sample(pool, k)


# Quotas tuned so common production failures dominate, Smart Restart
# subcategories get steady representation, and `completed` baselines are
# heavy enough to teach the "no failureCategory needed" pattern.
DEFAULT_QUOTAS: dict[str, int] = {
    "completed": 90,
    "dataset_resolution_failure": 50,
    "allocation_failure": 35,
    "permission_or_security_failure": 35,
    "jcl_syntax_failure": 35,
    "utility_parameter_failure": 35,
    "execution_abend": 50,
    "scheduler_or_environment_issue": 30,
    "other": 30,
    "smart_restart_resource_unavailable": 30,
    "smart_restart_configuration": 30,
    "smart_restart_application_logic": 25,
    "smart_restart_input_syntax": 25,
}


def _validate(sample: dict) -> tuple[bool, str]:
    raw = json.dumps(sample["expected_interpretation"])
    result = validate_spool_result(
        raw,
        schema_path=SCHEMA_PATH,
        contracts_path=CONTRACTS_PATH,
        user_prompt=sample["request"],
    )
    if result.ok:
        return True, ""
    errs = "; ".join(f"L{e.layer}.{e.code}" for e in result.errors[:3])
    return False, errs


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build the Spool Interpreter seed corpus.")
    parser.add_argument("--target", type=int, default=500)
    parser.add_argument("--seed", type=int, default=20260510)
    parser.add_argument("--append", action="store_true")
    parser.add_argument("--out", default=str(SEEDS_PATH))
    args = parser.parse_args(argv)

    rng = random.Random(args.seed)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    contracts = json.loads(CONTRACTS_PATH.read_text(encoding="utf-8"))
    related_docs_catalog = contracts.get("related_docs_catalog", {})
    if not related_docs_catalog:
        print(
            "warning: related_docs_catalog not found in node_contracts.json; "
            "relatedDocs will be empty for every sample"
        )

    existing_rows: list[dict] = []
    if args.append and out_path.exists():
        for line in out_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
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
        builder = CATEGORY_BUILDERS[category]
        produced = 0
        attempts = 0
        max_attempts = n * 20
        while produced < n and attempts < max_attempts:
            attempts += 1
            idx += 1
            request, interp = builder(rng)
            sid = f"syn-{category}-{_hash(category, idx, args.seed)}"
            if sid in seen_ids:
                continue
            _enrich_v2_fields(rng, category, interp, related_docs_catalog)
            sample = {
                "sample_id": sid,
                "source": "synthetic:template",
                "category": category,
                "request": request,
                "expected_interpretation": interp,
            }
            ok, err = _validate(sample)
            if not ok:
                rejected += 1
                if rejected <= 3:
                    print(f"  [reject] {category}: {err}")
                continue
            accepted.append(sample)
            seen_ids.add(sid)
            produced += 1
        print(f"  {category}: produced={produced}/{n} attempts={attempts}")

    rows_to_write = (existing_rows + accepted) if args.append else accepted
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
