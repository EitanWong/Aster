# Iteration 043 — Adaptive prefill transient tracker

Date: 2026-07-15

## Objective

Validate the static transient-prefill guard against a real 9B long-context
pressure workload, then close any measured gap without changing decode batching
or cache ownership.

## Start commit

`9e64a98 perf: guard transient prefill memory`

## Baseline investigation

A prefix-off Qwen3.5-9B 4-bit greedy probe with 30,181 prompt tokens completed
in 27 prefill steps, reached 11.378 GB MLX peak memory, and grew swap. The
static guard did not reduce the final tail because the idle fast path expanded
the remaining 3,556 tokens after the initial 1,024-token estimate.

Fixing that bypass alone retained 27 steps and the same 11.378 GB peak. The
remaining gap matched OMLX's reason for combining a static SDPA estimate with
recent measured chunk growth: the static score/output formula under-counted
other MLX command-buffer and activation costs.

## Reference design

`examples/omlx/omlx/scheduler.py` uses the maximum of the recent observed
per-token growth, an EWMA, and a context-aware static estimate, then applies a
safety multiplier. `examples/omlx/omlx/memory_monitor.py` resets/observes MLX
memory around prefill and separates the model baseline from request growth.

## Implementation

- `RequestState` now retains the previous prefill active memory and the maximum
  observed transient bytes per token for its own lifecycle.
- `InferenceEngine` observes each `PrefillChunkResult` after MLX resets its
  per-chunk peak. The first chunk establishes active baseline; later chunks
  record `peak - previous_active` divided by processed tokens.
- The next chunk uses the greater of the static full-attention estimate and
  the observed per-token growth times OMLX's 1.3 safety factor.
- Idle prefill completion re-runs the transient calculation for the full tail,
  so it cannot bypass a previously safe configured chunk budget.

## Correctness and tests

- New tests cover idle-tail bypass prevention, observed-growth chunk clamping,
  and peak-growth accounting.
- Full suite: `455 passed, 9 skipped, 1 warning`.
- The two 9B A/B pairs produced the identical greedy text SHA
  `4683328e2d0d020ba313d6566a6f9c5a8803f98a2c3f7d5a3520d37bbc734279`,
  128 completion tokens, and `length` finish reason.

## Controlled benchmark

Environment: macOS 27 arm64, 24 GB unified memory, MLX 0.32.0, mlx-lm 0.31.3,
Qwen3.5-9B 4-bit, prefix cache disabled, greedy sampling, one request, 30,181
prompt tokens and 128 completion tokens. Each run was a fresh Python process.
The baseline ran from a detached local worktree at `9e64a98`; candidate runs
used this iteration's working tree.

| Metric | Baseline pair | Candidate pair | Median change |
| --- | --- | --- | ---: |
| Greedy text SHA | identical | identical | exact parity |
| Prefill steps | 27, 27 | 28, 28 | +1 step |
| Peak MLX memory | 11.378, 11.378 GB | 11.042, 10.912 GB | -3.53% |
| Prefill model time | 76.231, 75.410 s | 75.253, 76.052 s | -0.22% |
| Total latency | 90.150, 89.356 s | 88.733, 89.750 s | -0.57% |
| Swap delta | 0, -8 MB | 0, 0 | no added swap |

The peak-memory improvement clears the project 3% gate for this long-context
resource scenario. Timing changes are within a narrow range and are not
claimed as a general throughput improvement.

## Decision

Retain the adaptive tracker as a long-context resource optimization. It adds a
single prefill step for the 30K probe while reducing peak MLX memory by 3.53%
at the median, preserving exact greedy output. It remains per-request and has
no shared state, so cancellation and cleanup discard it with `RequestState`.

## Next priority

Profile prefix snapshot lookup and clone cost across long Agent histories.
Compare Aster's full-snapshot store with the structural sharing, pinning, and
radix-index practices in SGLang and Rapid-MLX before changing cache ownership.

