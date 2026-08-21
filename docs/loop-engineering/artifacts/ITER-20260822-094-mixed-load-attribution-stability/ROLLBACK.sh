#!/bin/sh
set -eu
target=${1:?usage: ROLLBACK.sh TARGET}
cp "$(dirname "$0")/BASELINE_FILE" "$target"
