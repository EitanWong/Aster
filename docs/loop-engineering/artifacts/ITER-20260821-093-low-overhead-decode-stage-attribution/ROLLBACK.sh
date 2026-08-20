#!/bin/sh
set -eu

artifact_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
target=${1:?usage: ROLLBACK.sh TARGET_COPY}
cp "$artifact_dir/BASELINE_FILE" "$target"
