#!/bin/sh
set -eu
target=${1:?target path required}
python3 - "$target" <<'PY'
import json
import sys

path = sys.argv[1]
payload = json.load(open(path, encoding="utf-8"))
payload["benchmark"]["quiescence_admission"] = "disabled"
payload["benchmark"]["rejected_windows"] = "not-recorded"
payload["benchmark"]["external_cpu_estimate"] = "not-recorded"
with open(path, "w", encoding="utf-8") as handle:
    json.dump(payload, handle, indent=2)
    handle.write("\n")
PY
