"""Append Smart/RESTART abend seed samples to seed_samples.jsonl.

Adds ~25 hand-curated spool snippets covering each of the four
smart_restart_* failure categories defined in
flow-starter/docs/smart-restart/messages.md, plus one or two examples in
the existing categories that absorb specific DCA codes
(dataset_resolution_failure, permission_or_security_failure, execution_abend).

Idempotent: every sample id begins with seed-smart-restart-, so re-runs
detect the existing rows and skip the append. Each generated raw_spool is
constructed from real DCA wording from the manual; nothing is invented.

Usage:
    python3 flow-ml/scripts/expand_spool_smart_restart_samples.py
"""

from __future__ import annotations

import json
import pathlib
import sys

REPO = pathlib.Path(__file__).resolve().parents[1]
SEED_FILE = REPO / "models/spool-interpreter/datasets/samples/seed_samples.jsonl"
SENTINEL_PREFIX = "seed-smart-restart-"


def make_samples() -> list[dict]:
    """Build the static list of Smart/RESTART seed samples.

    Each sample uses the manual's exact DCA wording so the model learns
    to recognize the codes verbatim. raw_spool is short on purpose: the
    sanitized form fits in well under the 2048-token input cap.
    """
    samples: list[dict] = []

    def add(
        code: str,
        category: str,
        status: str,
        return_code: str | None,
        spool_lines: list[str],
        root_cause: str,
        suggested_fix: str,
    ) -> None:
        sample_id = f"{SENTINEL_PREFIX}{code.lower()}"
        spool = "JOB12345 STARTED\n" + "\n".join(spool_lines) + "\nJOB12345 ENDED\n"
        samples.append(
            {
                "sample_id": sample_id,
                "source": f"manual:smart-rrsaf:{code}",
                "raw_spool": spool,
                "status": status,
                "return_code": return_code,
                "failure_category": category,
                "root_cause": root_cause,
                "suggested_fix": suggested_fix,
            }
        )

    # ---- smart_restart_resource_unavailable (transient, retryable) --------

    add(
        code="DCA002E",
        category="smart_restart_resource_unavailable",
        status="failed",
        return_code="DCA002E",
        spool_lines=[
            "DCA002E Smart/RRSAF Will Wait 5 Minutes before Retrying Connection to DSN1",
        ],
        root_cause="Db2 subsystem DSN1 is not operational; Smart/RRSAF is waiting before retrying the connection",
        suggested_fix="Wait for the Db2 subsystem to come online; Smart/RRSAF retries automatically up to its RETRY count",
    )
    add(
        code="DCA019E",
        category="smart_restart_resource_unavailable",
        status="failed",
        return_code="DCA019E",
        spool_lines=[
            "DCA002E Smart/RRSAF Will Wait 5 Minutes before Retrying Connection to DSN1",
            "DCA019E Exhausted Connection Retries to DSN1 - Initialization Failed",
        ],
        root_cause="Smart/RRSAF exhausted connection retries to Db2 subsystem DSN1",
        suggested_fix="Resubmit when the Db2 subsystem is fully operational; Smart/RESTART resumes from the last checkpoint",
    )
    add(
        code="DCA021E",
        category="smart_restart_resource_unavailable",
        status="failed",
        return_code="DCA021E",
        spool_lines=[
            "DCA021E Db2 Subsystem DSN1 Has Terminated since the Last SQL Call",
        ],
        root_cause="Db2 subsystem DSN1 terminated between two SQL calls in the running step",
        suggested_fix="Resubmit the job step when the Db2 subsystem is back online",
    )
    add(
        code="DCA063E",
        category="smart_restart_resource_unavailable",
        status="failed",
        return_code="DCA063E",
        spool_lines=[
            "DCA063E Db2 Subsystem Named DSN1 Is Not Active",
        ],
        root_cause="Db2 subsystem DSN1 is not active at job start",
        suggested_fix="Resubmit when the Db2 subsystem is fully operational",
    )
    add(
        code="DCA066E",
        category="smart_restart_resource_unavailable",
        status="failed",
        return_code="DCA066E",
        spool_lines=[
            "DCA066E Connection Failed Because Db2 Subsystem DSN1 Is Terminating",
        ],
        root_cause="Smart/RRSAF connection rejected because Db2 subsystem DSN1 is in the process of terminating",
        suggested_fix="Wait for the Db2 subsystem to come fully down then back up before resubmitting",
    )
    add(
        code="DCA069E",
        category="smart_restart_resource_unavailable",
        status="failed",
        return_code="DCA069E",
        spool_lines=[
            "DCA069E Maximum Connections Reached. Attempt Db2 Connection Later.",
        ],
        root_cause="Db2 connection ceiling reached; no thread available for the application plan",
        suggested_fix="Resubmit later; if recurrent, raise the Db2 max-connections limit",
    )
    add(
        code="U2001-DCA007",
        category="smart_restart_resource_unavailable",
        status="abended",
        return_code="U2001",
        spool_lines=[
            "DCA021E Db2 Subsystem DSN1 Has Terminated since the Last SQL Call",
            "IEA995I SYMPTOM DUMP OUTPUT",
            "SYSTEM COMPLETION CODE=U2001  REASON CODE=00DCA007",
        ],
        root_cause="Smart/RRSAF abended with U2001 reason 00DCA007 because Db2 terminated mid-job",
        suggested_fix="Resubmit when Db2 is back; Smart/RESTART resumes from the last checkpoint",
    )
    add(
        code="U2001-DCA020",
        category="smart_restart_resource_unavailable",
        status="abended",
        return_code="U2001",
        spool_lines=[
            "IEA995I SYMPTOM DUMP OUTPUT",
            "SYSTEM COMPLETION CODE=U2001  REASON CODE=00DCA020",
        ],
        root_cause="Smart/RRSAF abended with U2001 reason 00DCA020 because Db2 was forcibly terminated by -STOP Db2 FORCE",
        suggested_fix="Resubmit after the operations team has restarted Db2",
    )

    # ---- smart_restart_configuration (deploy or installation issue) ------

    add(
        code="DCA001E",
        category="smart_restart_configuration",
        status="failed",
        return_code="DCA001E",
        spool_lines=[
            "DCA001E Unable to Load Smart/RRSAF - Terminating",
        ],
        root_cause="DCAHLI could not locate the SQLBATCH load module on the program fetch search path",
        suggested_fix="Add the SQLBATCH library to STEPLIB or JOBLIB, or include it in the MVS link list",
    )
    add(
        code="DCA046E",
        category="smart_restart_configuration",
        status="failed",
        return_code="DCA046E",
        spool_lines=[
            "DCA046E Smart/RRSAF Exception Manager Disabled. DXM Couldn't Be Loaded",
        ],
        root_cause="DXM exception manager module is not present on the program fetch search path",
        suggested_fix="Place the DXM load module in a library on STEPLIB or JOBLIB or in the MVS link list",
    )
    add(
        code="CA064E",
        category="smart_restart_configuration",
        status="failed",
        return_code="CA064E",
        spool_lines=[
            "CA064E Db2 Subsystem Named DSNX Does Not Exist",
        ],
        root_cause="No MVS subsystem is defined with the name DSNX; Smart/RRSAF cannot connect",
        suggested_fix="Correct the SYSTEM keyword or the default Db2 subsystem in the Smart/RRSAF profile",
    )
    add(
        code="DCA067E",
        category="smart_restart_configuration",
        status="failed",
        return_code="DCA067E",
        spool_lines=[
            "DCA067E Application Plan Named PAYROLL  Is Not Valid",
        ],
        root_cause="Application plan PAYROLL is not bound or has been freed",
        suggested_fix="BIND the plan and rerun the job step",
    )
    add(
        code="DCA090E",
        category="smart_restart_configuration",
        status="failed",
        return_code="DCA090E",
        spool_lines=[
            "DCA090E Failed To Locate Smart/RRSAF Main Module DCA$MAIN - Terminating",
        ],
        root_cause="The DCA$MAIN load module is not on the program fetch search path",
        suggested_fix="Add the Smart/RRSAF load library to STEPLIB or JOBLIB or to the MVS link list",
    )
    add(
        code="DCA093E",
        category="smart_restart_configuration",
        status="failed",
        return_code="DCA093E",
        spool_lines=[
            "DCA093E Couldn't Load Profile Defaults Module DCA$TPD - Terminating",
        ],
        root_cause="The DCA$TPD profile defaults table has not been tailored, assembled, and link-edited",
        suggested_fix="Run the Smart/RRSAF Administration Facility to build DCA$TPD and place it on the search path",
    )
    add(
        code="U2001-DCA004",
        category="smart_restart_configuration",
        status="abended",
        return_code="U2001",
        spool_lines=[
            "DCA062E Error on Db2 Plan PAYROLL: Return Code 0008 Reason Code 00F30025",
            "IEA995I SYMPTOM DUMP OUTPUT",
            "SYSTEM COMPLETION CODE=U2001  REASON CODE=00DCA004",
        ],
        root_cause="Smart/RRSAF abended with U2001 reason 00DCA004 because thread creation for plan PAYROLL failed",
        suggested_fix="Look up the Db2 return and reason code in Db2 Messages and Codes; rebind PAYROLL if it is no longer current",
    )

    # ---- smart_restart_application_logic (app code bug) ------------------

    add(
        code="DCA013E",
        category="smart_restart_application_logic",
        status="failed",
        return_code="DCA013E",
        spool_lines=[
            "DCA013E Smart/RRSAF Failed to Reconnect following Job Restart",
        ],
        root_cause="Smart/RRSAF could not reconnect to Db2 after a checkpoint restart attempt",
        suggested_fix="Collect SYSUDUMP and review the Smart/RESTART profile and application restart logic",
    )
    add(
        code="DCA020E",
        category="smart_restart_application_logic",
        status="failed",
        return_code="DCA020E",
        spool_lines=[
            "DCA001E Unable to Load Smart/RRSAF - Terminating",
            "DCA020E SQL Call Issued after Smart/RRSAF Initialization Failed",
        ],
        root_cause="Application issued an SQL call after Smart/RRSAF initialization had already failed",
        suggested_fix="Add logic to terminate the application after a Smart/RRSAF init failure SQLCODE",
    )
    add(
        code="DCA022E",
        category="smart_restart_application_logic",
        status="abended",
        return_code="U2001",
        spool_lines=[
            "DCA022E Smart/RRSAF Abending - No Environment Exists for SQL Calls",
            "IEA995I SYMPTOM DUMP OUTPUT",
            "SYSTEM COMPLETION CODE=U2001  REASON CODE=00DCA006",
        ],
        root_cause="Application repeatedly issued SQL calls without a valid Smart/RRSAF environment; abended after five attempts",
        suggested_fix="Fix application logic so SQL calls stop after Smart/RRSAF initialization fails",
    )
    add(
        code="DCA024E",
        category="smart_restart_application_logic",
        status="failed",
        return_code="DCA024E",
        spool_lines=[
            "DCA021E Db2 Subsystem DSN1 Has Terminated since the Last SQL Call",
            "DCA024E SQL Call Was Issued after Db2 Terminated",
        ],
        root_cause="Application issued a fresh SQL call after being notified that Db2 had terminated",
        suggested_fix="Add logic to terminate the application gracefully on Db2 termination notification",
    )
    add(
        code="DCA048E",
        category="smart_restart_application_logic",
        status="failed",
        return_code="DCA048E",
        spool_lines=[
            "DCA048E Smart/RRSAF Epilog Won't Be Called because of Invalid Savearea Backchain",
        ],
        root_cause="OS linkage convention violation detected; Smart/RRSAF task termination will not be called",
        suggested_fix="Inspect the calling sequence and issue an explicit SQL COMMIT or ROLLBACK from the application",
    )

    # ---- smart_restart_input_syntax (RAINPUT / SQLBATCH parameter mistakes)

    add(
        code="DCA003E",
        category="smart_restart_input_syntax",
        status="failed",
        return_code="DCA003E",
        spool_lines=[
            "DCA097I Smart/RRSAF Control Statement Follows:",
            "DCA099I PLAN PAYROLL",
            "DCA003E Smart/RRSAF Input Lacks Open Paren in the Form Keyword(value)",
        ],
        root_cause="PLAN keyword in the Smart/RRSAF input is missing its open parenthesis",
        suggested_fix="Rewrite as PLAN(PAYROLL) and rerun the job step",
    )
    add(
        code="DCA006E",
        category="smart_restart_input_syntax",
        status="failed",
        return_code="DCA006E",
        spool_lines=[
            "DCA097I Smart/RRSAF Control Statement Follows:",
            "DCA099I PLN(PAYROLL)",
            "DCA006E Smart/RRSAF Found Unrecognized Keyword PLN",
        ],
        root_cause="PLN is not a recognized Smart/RRSAF keyword; the intended keyword is PLAN",
        suggested_fix="Correct the keyword to PLAN(PAYROLL) and rerun",
    )
    add(
        code="DCA042E",
        category="smart_restart_input_syntax",
        status="failed",
        return_code="DCA042E",
        spool_lines=[
            "DCA097I Smart/RRSAF Control Statement Follows:",
            "DCA099I PROGRAM(1BADNAME)",
            "DCA042E The PROGRAM Name 1BADNAME is Invalid",
        ],
        root_cause="Program name 1BADNAME starts with a digit; program names must start with an alphabetic character",
        suggested_fix="Rename the program to start with A through Z or correct the PROGRAM keyword value",
    )
    add(
        code="DCA005E",
        category="smart_restart_input_syntax",
        status="failed",
        return_code="DCA005E",
        spool_lines=[
            "DCA097I Smart/RRSAF Control Statement Follows:",
            "DCA099I RETRY(MAX)",
            "DCA005E Smart/RRSAF Requires an Integer RETRY(MAX)",
        ],
        root_cause="RETRY parameter received the non-numeric value MAX; an integer between 0 and 255 is required",
        suggested_fix="Specify an integer in range, for example RETRY(5)",
    )
    add(
        code="DCA044E",
        category="smart_restart_input_syntax",
        status="failed",
        return_code="DCA044E",
        spool_lines=[
            "DCA097I Smart/RRSAF Control Statement Follows:",
            "DCA099I DETECT(MAYBE)",
            "DCA044E Smart/RRSAF DETECT Option Must Be YES, NO, ON, or OFF",
        ],
        root_cause="DETECT option received an invalid value MAYBE",
        suggested_fix="Specify DETECT(YES), DETECT(NO), DETECT(ON), or DETECT(OFF)",
    )

    # ---- existing categories that absorb specific DCA codes -------------

    add(
        code="DCA009E",
        category="dataset_resolution_failure",
        status="failed",
        return_code="DCA009E",
        spool_lines=[
            "DCA009E SQLBATCH File Open Failure - Terminating",
            "IEC130I SQLBATCH DD STATEMENT MISSING",
        ],
        root_cause="SQLBATCH DD statement is missing or points at a dataset that cannot be opened",
        suggested_fix="Add or correct the SQLBATCH DD statement in the JCL and rerun",
    )
    add(
        code="DCA065E",
        category="permission_or_security_failure",
        status="failed",
        return_code="DCA065E",
        spool_lines=[
            "DCA065E ID Is Not Authorized To Use Db2",
        ],
        root_cause="Job authorization ID lacks RACF permission to use Db2",
        suggested_fix="Request Db2 access from the Db2 system administrator for the job authorization ID",
    )
    add(
        code="DCA068E",
        category="permission_or_security_failure",
        status="failed",
        return_code="DCA068E",
        spool_lines=[
            "DCA068E Your Db2 Connection ID Is Not Authorized To Use Plan PAYROLL",
        ],
        root_cause="Connection ID lacks Execute authority on Db2 plan PAYROLL",
        suggested_fix="GRANT EXECUTE ON PLAN PAYROLL to the connection ID and rerun",
    )
    add(
        code="DCA095E-803",
        category="execution_abend",
        status="failed",
        return_code="SQLCODE=-803",
        spool_lines=[
            "DCA095E Failing SQL Statement = INSERT : SQLCODE = -00803",
        ],
        root_cause="INSERT failed with SQLCODE -803 (duplicate key on a unique index)",
        suggested_fix="Inspect the input row for a key collision; deduplicate or use MERGE/UPSERT",
    )

    return samples


def main() -> int:
    if not SEED_FILE.exists():
        print(f"seed file missing: {SEED_FILE}", file=sys.stderr)
        return 1

    existing_lines = SEED_FILE.read_text(encoding="utf-8").splitlines()
    existing_ids = set()
    for raw in existing_lines:
        if not raw.strip():
            continue
        try:
            obj = json.loads(raw)
        except json.JSONDecodeError:
            continue
        existing_ids.add(obj.get("sample_id"))

    if any(sid and sid.startswith(SENTINEL_PREFIX) for sid in existing_ids):
        print(
            "Smart/RESTART seed samples already present (sentinel prefix detected); "
            "no append needed."
        )
        return 0

    new_samples = make_samples()
    appended = []
    with SEED_FILE.open("a", encoding="utf-8") as f:
        for sample in new_samples:
            if sample["sample_id"] in existing_ids:
                continue
            f.write(json.dumps(sample, ensure_ascii=False))
            f.write("\n")
            appended.append(sample["sample_id"])

    print(f"appended {len(appended)} Smart/RESTART seed samples to {SEED_FILE}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
