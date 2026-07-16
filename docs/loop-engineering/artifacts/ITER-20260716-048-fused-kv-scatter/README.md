# Iteration 048 Reproduction Artifact

This artifact compares three fused K/V scatter designs:

1. pure MLX combined storage with one scatter;
2. vllm-metal `reshape_and_cache` at commit `4c18ee0`;
3. an Aster-layout standalone C++ Primitive.

The first and third designs were rejected. No binary or Aster runtime change is
archived.

## Setup

```bash
export REPO_ROOT="$(git rev-parse --show-toplevel)"
export ARTIFACT="$REPO_ROOT/docs/loop-engineering/artifacts/ITER-20260716-048-fused-kv-scatter"
export VENV_DIR="$(mktemp -d /tmp/aster-iter048-venv.XXXXXX)"
export REFERENCE_VENV_DIR="$(mktemp -d /tmp/aster-iter048-ref-venv.XXXXXX)"
export REFERENCE_COPY="$(mktemp -d /tmp/aster-iter048-vllm-metal.XXXXXX)"
export ASTER_KV_SCATTER_BUILD_DIR="$(mktemp -d /tmp/aster-iter048-build.XXXXXX)"
export PYTHON="$VENV_DIR/bin/python"
export REFERENCE_PYTHON="$REFERENCE_VENV_DIR/bin/python"

test "$(uv --version | awk '{print $2}')" = "0.11.15"
uv --no-config venv --python 3.13.12 "$VENV_DIR"
uv --no-config venv --python 3.13.12 "$REFERENCE_VENV_DIR"
test "$("$PYTHON" --version)" = "Python 3.13.12"
test "$("$REFERENCE_PYTHON" --version)" = "Python 3.13.12"

env -u UV_INDEX -u UV_EXTRA_INDEX_URL -u UV_INDEX_URL \
  uv --no-config pip sync --python "$PYTHON" --require-hashes --strict \
  --only-binary :all: --default-index https://pypi.org/simple \
  "$ARTIFACT/requirements.lock"
env -u UV_INDEX -u UV_EXTRA_INDEX_URL -u UV_INDEX_URL \
  uv --no-config pip sync --python "$REFERENCE_PYTHON" --require-hashes --strict \
  --only-binary :all: --default-index https://pypi.org/simple \
  "$ARTIFACT/reference-vllm-metal/requirements.lock"
```

The `.txt` files record direct intent. The generated `.lock` files include the
transitive `mlx-metal` dependency and distribution hashes; only the lock files
are installed.

The uv floor is 0.11.15 because it fixes
[GHSA-4gg8-gxpx-9rph](https://github.com/astral-sh/uv/security/advisories/GHSA-4gg8-gxpx-9rph).
The recorded run used the immutable
[0.11.15 release](https://github.com/astral-sh/uv/releases/tag/0.11.15) after
verifying its checksum and GitHub artifact attestation.

## Pure MLX

```bash
"$PYTHON" "$ARTIFACT/pure-mlx/bench.py" \
  --smoke-only --run-id 10 \
  --output "$ARTIFACT/pure-mlx/results/smoke.json"

for run_id in 11 12 13 14 15; do
  index=$((run_id - 10))
  "$PYTHON" "$ARTIFACT/pure-mlx/bench.py" \
    --warmups 30 --iterations 200 --run-id "$run_id" \
    --output "$ARTIFACT/pure-mlx/results/main-run-$index.json"
done

"$PYTHON" "$ARTIFACT/pure-mlx/aggregate.py" \
  --resamples 5000 \
  --output "$ARTIFACT/pure-mlx/results/bootstrap.json" \
  "$ARTIFACT/pure-mlx/results/main-run-1.json" \
  "$ARTIFACT/pure-mlx/results/main-run-2.json" \
  "$ARTIFACT/pure-mlx/results/main-run-3.json" \
  "$ARTIFACT/pure-mlx/results/main-run-4.json" \
  "$ARTIFACT/pure-mlx/results/main-run-5.json"
```

The real combined-storage path established no gain and showed directional
16-token batch-1 and 64-token batch-2 regressions. Pre-stacking K/V also failed
to establish a stable 3% ceiling, so changing Aster's pool layout is not
justified.

## Reference Primitive

Export the exact vllm-metal commit into an isolated copy. The benchmark checks
the exported source against `manifest.json` before importing or building it:

```bash
test "$(git -C "$REPO_ROOT/examples/vllm-metal" rev-parse HEAD)" = \
  "4c18ee0e6e3ce2b594ab114d0a53ca24eafb1d58"
test -z "$(git -C "$REPO_ROOT/examples/vllm-metal" status \
  --porcelain --untracked-files=no)"
git -C "$REPO_ROOT/examples/vllm-metal" archive \
  4c18ee0e6e3ce2b594ab114d0a53ca24eafb1d58 \
  vllm_metal tests/test_reshape_and_cache.py | tar -x -C "$REFERENCE_COPY"

PYTHONPATH="$REFERENCE_COPY" \
VLLM_METAL_BUILD_FROM_SOURCE=1 \
VLLM_METAL_REFERENCE_ROOT="$REFERENCE_COPY/vllm_metal" \
"$REFERENCE_PYTHON" "$ARTIFACT/reference-vllm-metal/bench.py" \
  --smoke-only --run-id 20 \
  --output "$ARTIFACT/reference-vllm-metal/results/smoke.json"

for run_id in 21 22 23 24 25; do
  index=$((run_id - 20))
  PYTHONPATH="$REFERENCE_COPY" \
  VLLM_METAL_BUILD_FROM_SOURCE=1 \
  VLLM_METAL_REFERENCE_ROOT="$REFERENCE_COPY/vllm_metal" \
  "$REFERENCE_PYTHON" "$ARTIFACT/reference-vllm-metal/bench.py" \
    --warmups 30 --iterations 200 --run-id "$run_id" \
    --output "$ARTIFACT/reference-vllm-metal/results/main-run-$index.json"
done

"$REFERENCE_PYTHON" "$ARTIFACT/reference-vllm-metal/aggregate.py" \
  --resamples 5000 \
  --output "$ARTIFACT/reference-vllm-metal/results/bootstrap.json" \
  "$ARTIFACT/reference-vllm-metal/results/main-run-1.json" \
  "$ARTIFACT/reference-vllm-metal/results/main-run-2.json" \
  "$ARTIFACT/reference-vllm-metal/results/main-run-3.json" \
  "$ARTIFACT/reference-vllm-metal/results/main-run-4.json" \
  "$ARTIFACT/reference-vllm-metal/results/main-run-5.json"
```

The pinned reference passed FP16/BF16/FP32 and negative-slot parity. Its
1/4/8/16/64/128 token cells cleared the 3% interval gate, proving that the
mechanism works for vllm-metal's token-contiguous slot-mapping layout.

## Aster Layout

```bash
chmod 700 "$ASTER_KV_SCATTER_BUILD_DIR"
"$VENV_DIR/bin/cmake" \
  -S "$ARTIFACT/aster-layout" \
  -B "$ASTER_KV_SCATTER_BUILD_DIR" \
  -DPython_EXECUTABLE="$PYTHON" \
  -DCMAKE_BUILD_TYPE=Release
"$VENV_DIR/bin/cmake" \
  --build "$ASTER_KV_SCATTER_BUILD_DIR" --config Release -j 4

export ASTER_ROOT="$REPO_ROOT"
export VLLM_METAL_REFERENCE_ROOT="$REPO_ROOT/examples/vllm-metal/vllm_metal"

"$PYTHON" "$ARTIFACT/aster-layout/bench.py" \
  --smoke-only --run-id 30 \
  --output "$ARTIFACT/aster-layout/results/smoke.json"

for run_id in 31 32 33 34 35; do
  index=$((run_id - 30))
  "$PYTHON" "$ARTIFACT/aster-layout/bench.py" \
    --warmups 30 --iterations 200 --run-id "$run_id" \
    --output "$ARTIFACT/aster-layout/results/main-run-$index.json"
done

"$PYTHON" "$ARTIFACT/aster-layout/aggregate.py" \
  --resamples 5000 \
  --output "$ARTIFACT/aster-layout/results/main-bootstrap.json" \
  "$ARTIFACT/aster-layout/results/main-run-1.json" \
  "$ARTIFACT/aster-layout/results/main-run-2.json" \
  "$ARTIFACT/aster-layout/results/main-run-3.json" \
  "$ARTIFACT/aster-layout/results/main-run-4.json" \
  "$ARTIFACT/aster-layout/results/main-run-5.json"
```

Run the independent confirmation and stress matrix:

```bash
for run_id in 41 42 43 44 45; do
  index=$((run_id - 40))
  "$PYTHON" "$ARTIFACT/aster-layout/bench.py" \
    --warmups 30 --iterations 200 --run-id "$run_id" \
    --output "$ARTIFACT/aster-layout/results/confirm-run-$index.json"
done

"$PYTHON" "$ARTIFACT/aster-layout/aggregate.py" \
  --resamples 5000 \
  --output "$ARTIFACT/aster-layout/results/confirm-bootstrap.json" \
  "$ARTIFACT/aster-layout/results/confirm-run-1.json" \
  "$ARTIFACT/aster-layout/results/confirm-run-2.json" \
  "$ARTIFACT/aster-layout/results/confirm-run-3.json" \
  "$ARTIFACT/aster-layout/results/confirm-run-4.json" \
  "$ARTIFACT/aster-layout/results/confirm-run-5.json"

"$PYTHON" "$ARTIFACT/aster-layout/bench.py" \
  --warmups 30 --iterations 1000 --run-id 51 \
  --output "$ARTIFACT/aster-layout/results/stress.json"
```

No cell established a >=3% gain in both the main and confirmation groups. The
confirmation instead found stable regressions for 64-token batch 4 and 8. The
1000-iteration matrix retained exact post-loop parity, zero swap growth, no
thermal warning, and a maximum MLX peak of `52,428,824 B`.

## Integrity

`manifest.json` binds every executable artifact source, both hash-locked
dependency sets, Aster's compared adapter, the exact vllm-metal commit, and its
three audited source files. Every raw record carries that manifest hash. The
aggregators re-hash the current sources, verify the nested repository commit and
clean tracked state, reject duplicate cells/processes, enforce 30/200 sample
shape, recompute every summary and delta from raw samples, and require exact
post-benchmark pool parity.

Process effects are computed as candidate/baseline ratios inside each process
before the five process effects are aggregated. Bootstrap intervals resample
processes and use paired moving-block indices inside each selected process;
they remain exploratory stability estimates, not calibrated significance
tests.

The Aster-layout records additionally archive before/after `vm.swapusage` and
`pmset -g therm` command output. The stress record verifies complete-pool parity
after all 1,000 writes per method in every cell. Power remains unmeasured
because `powermetrics` requires elevated privileges.
