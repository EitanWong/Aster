#!/bin/sh
set -eu
target=${1:?target path required}
python3 - "$target" <<'PY'
import json
import sys

path = sys.argv[1]
payload = json.load(open(path, encoding="utf-8"))
payload["benchmark"]["telemetry"] = "disabled"
payload["benchmark"]["thermal_power"] = "unprobed"
payload["benchmark"]["process_sampling"] = "none"
with open(path, "w", encoding="utf-8") as handle:
    json.dump(payload, handle, indent=2)
    handle.write("\n")
PY
