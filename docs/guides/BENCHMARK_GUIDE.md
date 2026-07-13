# Engine Benchmark Guide

## Purpose

The current benchmark path is focused on the text engine itself, not the old
autotune or worker architecture.

Use it to validate:

- single-request latency
- repeated-prefix reuse
- divergent-prefix safety behavior
- mixed concurrency behavior
- long-prompt stability
- decode-step throughput

## Prerequisites

Run this on an Apple Silicon machine with the real runtime installed:

- `mlx`
- `mlx-lm`
- `PyYAML`
- `prometheus-client`

Use the same model weights and config you intend to serve with.

## Direct Engine Benchmark

The benchmark bypasses HTTP and exercises `InferenceEngine` directly.

```bash
python scripts/dev/benchmark_live.py --config configs/config.yaml --workload all --concurrency 2
```

The default benchmark uses the configured runtime kernel. To force the current
stable path:

```bash
python scripts/dev/benchmark_live.py \
  --config configs/config.yaml \
  --workload mixed \
  --concurrency-levels 1,2,4,8 \
  --runtime-kernel manual
```

For a long-context pressure probe, set the repeated prompt length explicitly:

```bash
python scripts/dev/benchmark_live.py \
  --config configs/config.yaml \
  --workload long \
  --concurrency 1 \
  --long-prompt-words 16000 \
  --runtime-kernel manual
```

Optional output file:

```bash
python scripts/dev/benchmark_live.py \
  --config configs/config.yaml \
  --workload all \
  --concurrency 4 \
  --output benchmark_results/engine_benchmark.json
```

## Workloads

### `single`

One interactive request. Use this for baseline latency and TTFT validation.

### `reuse`

Repeated identical prompts with a large shared prefix. Use this to validate:

- prefix reuse hits
- reused prompt tokens
- second-request latency drop

### `reuse-divergent`

Two sequential agent turns with a large shared prefix and different suffixes.
This distinguishes a real LCP hit from a safe skip when the underlying MLX-LM
cache cannot rewind. Inspect `prefix_unsafe_lcp_skips` rather than counting a
shared string prefix as a hit.

### `mixed`

Short and long prompts together. Use this to inspect fairness and decode batch
behavior under local multi-request serving.

### `long`

Long prompt ingestion with generation. Use this to validate prefill chunking,
memory pressure response, and sustained responsiveness.

## What To Record

For each run, capture:

- elapsed wall time
- average latency
- p95 latency
- runtime kernel
- MLX allocator peak memory in GB
- prefill peak and active MLX memory in GB
- aggregate completion tokens/sec
- average generation tokens/sec
- completion tokens per decode step
- total completion tokens
- total prompt tokens
- prefix reuse hits
- prefix tokens reused
- exact/LCP cache hits and unsafe LCP skips
- decode steps
- completed, failed, cancelled, and admission-rejected request counts

Also collect:

- peak memory from MLX or Activity Monitor
- model name and quantization
- exact Aster config
- machine type and RAM size

## Recommended Comparison Set

Run the same workloads against:

1. old Aster MLX baseline, if still available
2. current native engine
3. `engine.runtime_kernel=manual`
4. future `engine.runtime_kernel=batch_generator` once the implementation is
   enabled
5. any local historical result already captured before the rewrite

Be explicit about:

- model identity
- prompt set
- concurrency
- context length
- whether prefix reuse was warm or cold
- exact or divergent prefix workload
- runtime kernel

## Interpreting Results

Healthy signs:

- `reuse` shows meaningful reused-token counts and lower latency on later requests
- `mixed` shows decode batch steps greater than one under concurrency
- `long` completes without uncontrolled failure growth or repeated admission rejects

Potential next bottleneck:

- if decode batch size is greater than one but throughput gains are small, the
  merge-and-extract cost of temporary batch caches is likely the next hot path

## Current Limitation

This benchmark is intentionally honest:

- it does not claim wins by itself
- it prepares the workloads and counters needed to measure the new engine
- final performance conclusions still require live execution on the target MLX setup
