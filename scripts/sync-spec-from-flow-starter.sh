#!/usr/bin/env bash
# Copy the compiled DSL spec from the flow-starter repo into the
# dsl-generator's prompt_spec.json `system` field.
#
# Why a copy and not an import or symlink: flow-ml ships standalone (e.g.
# on a training box without flow-starter checked out), so the model
# package must carry the spec text it was trained against. We also stamp
# the source git-sha into a `_provenance` block on the JSON so a quick
# `git log -1 <sha>` from the flow-starter side identifies which exact
# adapter catalog the model was grounded against.
#
# Usage:
#   bash flow-ml/scripts/sync-spec-from-flow-starter.sh
#   FLOW_STARTER=/path/to/flow-starter bash flow-ml/scripts/sync-spec-from-flow-starter.sh
#
# Defaults to ../flow-starter relative to the flow-ml repo root.
#
# Idempotent: re-running with no changes upstream produces no diff. Forces
# a fresh build of `docs/dsl/spec_compiled.md` first via
# `cargo build -p flow-application` so the sync uses the latest committed
# adapter catalog, not a stale cached file.

set -euo pipefail

FLOW_ML_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
FLOW_STARTER="${FLOW_STARTER:-$(cd "$FLOW_ML_ROOT/../flow-starter" && pwd)}"

if [[ ! -d "$FLOW_STARTER" ]]; then
  echo "flow-starter checkout not found at $FLOW_STARTER" >&2
  echo "  set FLOW_STARTER=/path/to/flow-starter and re-run" >&2
  exit 2
fi

SPEC_FILE="$FLOW_STARTER/docs/dsl/spec_compiled.md"
PROMPT_SPEC="$FLOW_ML_ROOT/models/dsl-generator/datasets/prompt_spec.json"

# 1. Refresh the compiled spec from sources. `cargo build -p flow-application`
#    triggers the build.rs that re-runs the markdown renderer; we touch
#    its inputs first to force re-execution even when nothing else
#    changed.
echo "refreshing $SPEC_FILE via cargo build -p flow-application ..."
( cd "$FLOW_STARTER" && touch crates/flow-application/build.rs && cargo build -p flow-application --quiet )

if [[ ! -f "$SPEC_FILE" ]]; then
  echo "spec_compiled.md missing at $SPEC_FILE after build" >&2
  exit 1
fi

# 2. Read the source git sha so we can stamp it into the prompt_spec.
SOURCE_SHA="$(cd "$FLOW_STARTER" && git rev-parse HEAD 2>/dev/null || echo "unknown")"

# 3. Splice the new system field + provenance into prompt_spec.json. We
#    use `python3 -c` rather than `jq` because Python's stdlib is more
#    likely to be present on a clean training-box install (a Mac or
#    Linux box that has flow-ml's pip deps but may not have jq).
python3 - "$SPEC_FILE" "$PROMPT_SPEC" "$SOURCE_SHA" <<'PY'
import json
import pathlib
import sys
from datetime import datetime, timezone

spec_path = pathlib.Path(sys.argv[1])
prompt_spec_path = pathlib.Path(sys.argv[2])
source_sha = sys.argv[3]

system = spec_path.read_text(encoding="utf-8")
spec = json.loads(prompt_spec_path.read_text(encoding="utf-8"))
spec["system"] = system
spec["_provenance"] = {
    "source": "flow-starter:docs/dsl/spec_compiled.md",
    "git_sha": source_sha,
    "synced_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
}

# Stable key ordering keeps the JSONL diff readable; the runtime is
# tolerant of key order so this is purely cosmetic.
prompt_spec_path.write_text(
    json.dumps(spec, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
    encoding="utf-8",
)
print(f"wrote {prompt_spec_path} ({len(system)} chars in `system`, sha={source_sha[:12]})")
PY
