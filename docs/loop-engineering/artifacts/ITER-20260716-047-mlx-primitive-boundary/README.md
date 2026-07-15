# Iteration 047 Reproduction Artifact

This directory preserves the rejected MLX C++ Primitive experiment from
Iteration 047. It is evidence and a reproducible benchmark, not production
Aster code.

## Scope

- Apple Silicon Metal only.
- Decode query length 1.
- FP16 `Hq=16`, `Hkv=8`, `D=V=128` benchmark specialization.
- MLX 0.32.0 with nanobind 2.13.0.
- Build output stays outside the repository.
- Aggregation requires Git to identify commit
  `22865cd0e290acdfe02e0b845eb680eef7fc0a76` as the last change to
  `aster/inference/metal_paged_attention.py` and requires that source to retain
  SHA-256
  `b7b4bea2ead78057d4d4759d99fc1de62f674a6ba3dd603b0d7233bc8bbd8796`.

## Setup

```bash
export REPO_ROOT="$(git rev-parse --show-toplevel)"
export ARTIFACT="$REPO_ROOT/docs/loop-engineering/artifacts/ITER-20260716-047-mlx-primitive-boundary"
export VENV_DIR="$(mktemp -d /tmp/aster-iter047-venv.XXXXXX)"
export ASTER_PRIMITIVE_BUILD_DIR="$(mktemp -d /tmp/aster-iter047-build.XXXXXX)"
export PYTHON="$VENV_DIR/bin/python"

uv venv --python 3.13 "$VENV_DIR"
uv pip install --python "$PYTHON" -r "$ARTIFACT/requirements.txt"
```

## Correctness

The smoke matrix covers block boundaries, causal offsets, rotated physical
blocks, invalid inputs, and invalid block propagation. The large matrix checks
every timed 2K/8K batch shape with random data against native MLX.

```bash
"$PYTHON" "$ARTIFACT/bench.py" \
  --aster-root "$REPO_ROOT" \
  --smoke-only --warmups 0 --iterations 1 --run-id 50 \
  --output "$ARTIFACT/results/smoke.json"

"$PYTHON" "$ARTIFACT/bench.py" \
  --aster-root "$REPO_ROOT" \
  --smoke-only --verify-large \
  --batches 1 2 4 8 --tokens 2048 8192 \
  --warmups 0 --iterations 1 --run-id 51 \
  --output "$ARTIFACT/results/large-correctness.json"
```

## Main A/B

Run five independent Python processes:

```bash
for run_id in 61 62 63 64 65; do
  index=$((run_id - 60))
  "$PYTHON" "$ARTIFACT/bench.py" \
    --aster-root "$REPO_ROOT" \
    --batches 1 2 4 8 --tokens 2048 8192 \
    --warmups 30 --iterations 200 --run-id "$run_id" \
    --output "$ARTIFACT/results/main-run-$index.json"
done
```

Aggregate with a paired moving-block bootstrap nested inside independent
process resampling:

```bash
"$PYTHON" "$ARTIFACT/aggregate.py" \
  --resamples 5000 \
  --output "$ARTIFACT/results/main-bootstrap.json" \
  "$ARTIFACT/results/main-run-1.json" \
  "$ARTIFACT/results/main-run-2.json" \
  "$ARTIFACT/results/main-run-3.json" \
  "$ARTIFACT/results/main-run-4.json" \
  "$ARTIFACT/results/main-run-5.json"
```

## Candidate Confirmation

The main matrix produced nominal `>=3%` median gains at 2K/batch-1 and
8K/batch-2 without establishing a `>=3%` interval. Re-run the containing four
cells in five new processes:

```bash
for run_id in 91 92 93 94 95; do
  index=$((run_id - 90))
  "$PYTHON" "$ARTIFACT/bench.py" \
    --aster-root "$REPO_ROOT" \
    --batches 1 2 --tokens 2048 8192 \
    --warmups 30 --iterations 200 --run-id "$run_id" \
    --output "$ARTIFACT/results/confirm-run-$index.json"
done

"$PYTHON" "$ARTIFACT/aggregate.py" \
  --resamples 5000 \
  --output "$ARTIFACT/results/confirm-bootstrap.json" \
  "$ARTIFACT/results/confirm-run-1.json" \
  "$ARTIFACT/results/confirm-run-2.json" \
  "$ARTIFACT/results/confirm-run-3.json" \
  "$ARTIFACT/results/confirm-run-4.json" \
  "$ARTIFACT/results/confirm-run-5.json"
```

## Stress A/B

Run the 32K matrix in five independent processes, then aggregate it:

```bash
for run_id in 71 72 73 74 75; do
  index=$((run_id - 70))
  "$PYTHON" "$ARTIFACT/bench.py" \
    --aster-root "$REPO_ROOT" \
    --batches 1 2 --tokens 32768 \
    --warmups 10 --iterations 50 --run-id "$run_id" \
    --output "$ARTIFACT/results/stress-32k-run-$index.json"
done

"$PYTHON" "$ARTIFACT/aggregate.py" \
  --resamples 5000 \
  --output "$ARTIFACT/results/stress-32k-bootstrap.json" \
  "$ARTIFACT/results/stress-32k-run-1.json" \
  "$ARTIFACT/results/stress-32k-run-2.json" \
  "$ARTIFACT/results/stress-32k-run-3.json" \
  "$ARTIFACT/results/stress-32k-run-4.json" \
  "$ARTIFACT/results/stress-32k-run-5.json"
```

Run the 64K matrix in five independent processes, then aggregate it:

```bash
for run_id in 81 82 83 84 85; do
  index=$((run_id - 80))
  "$PYTHON" "$ARTIFACT/bench.py" \
    --aster-root "$REPO_ROOT" \
    --batches 1 --tokens 65536 \
    --warmups 10 --iterations 30 --run-id "$run_id" \
    --output "$ARTIFACT/results/stress-64k-run-$index.json"
done

"$PYTHON" "$ARTIFACT/aggregate.py" \
  --resamples 5000 \
  --output "$ARTIFACT/results/stress-64k-bootstrap.json" \
  "$ARTIFACT/results/stress-64k-run-1.json" \
  "$ARTIFACT/results/stress-64k-run-2.json" \
  "$ARTIFACT/results/stress-64k-run-3.json" \
  "$ARTIFACT/results/stress-64k-run-4.json" \
  "$ARTIFACT/results/stress-64k-run-5.json"
```

## Results

All JSON files are compact, machine-readable raw records. Each run contains:

- environment, prototype source SHA-256 values, and Aster baseline identity;
- correctness and invalid-input outcomes;
- raw method samples plus median, p95, min, max, and standard deviation;
- MLX peak/active/post-clear memory;
- swap before/after values.

`aggregate.py` rejects fewer than five processes, duplicate run IDs or files,
duplicate process IDs, different environments or Aster baselines, result hashes
that differ from the current archived sources, inconsistent cells or benchmark
configurations, unexpected methods, non-finite/non-positive timings, nonzero
cross-path parity, and incomplete sample sets. Its 95% bootstrap intervals are
exploratory stability estimates, not calibrated significance tests.
