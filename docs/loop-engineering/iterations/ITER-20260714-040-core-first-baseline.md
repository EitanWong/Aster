# Iteration 040: Core-First Runtime Baseline

## Scope

- Starting commit: `6723384`
- No production code change; baseline and priority reset only.
- Machine: macOS 27.0, Apple Silicon arm64, 24 GB unified memory
- Packages: MLX `0.32.0`, mlx-lm `0.31.3`
- Model: Qwen3.5-0.8B 4-bit MLX
- Runtime: Aster manual runtime, prefix cache disabled, greedy sampling

## Decision boundary

The DFlash repositories cloned in `examples/` are reference material only.
The active workstream returns to Aster's core manual runtime: request
lifecycle, prefill/decode scheduling, KV/prefix ownership, batching,
correctness, memory pressure, and service stability. No DFlash speculative
decoding or third-party implementation is being integrated in this iteration.

## Reproducible baseline

```text
rtk .venv/bin/python scripts/dev/benchmark_live.py \
  --config /tmp/aster-qwen08-native.yaml \
  --workload all \
  --concurrency-levels 1,4 \
  --long-prompt-words 1024 \
  --output /tmp/aster-core-baseline-20260714.json
```

The workload used deterministic temperature `0.0`, 128 completion tokens for
single/long requests and the benchmark's fixed mixed/staggered request matrix.
All 12 records completed with zero errors, zero cancellations, and zero swap
growth (`swap_used_bytes` stayed at `1,152,647,168` before and after).

| Workload | Concurrency | Prompt tokens | Elapsed | Completion tok/s | Peak MLX |
| --- | ---: | ---: | ---: | ---: | ---: |
| single | 1 | 14 | `1.042s` | `122.89` | `0.511GB` |
| mixed | 1 | 1,416 | `2.636s` | `109.25` | `1.581GB` |
| staggered | 1 | 1,229 | `2.487s` | `109.39` | `1.492GB` |
| long | 1 | 1,205 | `1.206s` | `106.14` | `1.492GB` |
| mixed | 4 | 1,416 | `2.700s` | `106.66` | `1.581GB` |
| staggered | 4 | 1,229 | `2.588s` | `105.12` | `1.492GB` |
| long | 4 | 4,820 | `5.307s` | `96.48` | `1.626GB` |

Short single-request decode reached `146~148 tok/s` generation throughput;
the concurrency-4 long workload fell to `31.16 tok/s` average generation
throughput because prompt work and request-local decode states share the
manual engine loop. This is a baseline observation, not an optimization
claim. Prefix-cache hit fields are zero by design in this run.

## Next core investigation

Profile the manual runtime's long/concurrent path by prefill versus decode
step, then evaluate one state-isolation or scheduling change at a time. The
candidate must preserve token/hash/finish parity, cancellation, zero swap
growth, and the existing single-request decode baseline before any default
change is retained.
