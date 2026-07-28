# Iteration 054: Neutral Repetition Processor Screen

- **Date:** 2026-07-23
- **Start/end reference:** `2cb14052d4a3edbcbd420e8bf8d28cfce4a6bba2`; no production source change
- **Model/runtime:** Qwen3.5-0.8B 4-bit, MLX 0.32.0, MLX-LM 0.31.3

## Problem

`InferenceRequest.repetition_penalty` defaults to `1.0`. Aster passes that value
directly to MLX-LM 0.31.3's `make_logits_processors`, which creates a repetition
processor for every nonzero value. The resulting default processor gathers the
last 20 token logits, performs sign-dependent multiply/divide by one, and
scatters the unchanged values back. It is mathematically neutral but creates a
real MLX graph and forces construction of a token array on every decode row.

The benchmark candidate marked requests whose repetition penalty was exactly
`1.0` and skipped only the leading MLX-LM repetition processor. Non-neutral
penalty requests retained their complete processor list as a control. This
models passing `None` rather than `1.0` at sampler initialization without
changing production source.

## Results

Each measured cell used two fresh processes, 32 adjacent pair warmups, 256
measured steps, alternating call order, and alternating physical runner
assignment.

| Cell | Candidate speed change | Candidate behavior | Exact | Swap growth |
| --- | ---: | --- | :---: | ---: |
| greedy B2 | `+1.47%` median (`+1.36%` to `+1.58%`) | 576 skips/process | yes | 0 |
| greedy B4 | `+1.57%` median (`+1.37%` to `+1.76%`) | 1,152 skips/process | yes | 0 |
| greedy B8 | `+1.50%` median (`+0.89%` to `+2.11%`) | 2,304 skips/process | yes | 0 |
| penalties B4 | `+0.23%` median (`+0.03%` to `+0.43%`) | 0 skips/process | yes | 0 |

A 32-step greedy B4 smoke measured `+1.99%` and 160 skips. It is retained only
as smoke evidence. The real-penalty control's near-zero delta confirms that the
candidate does not remove active penalty graphs.

## Decision

**Reject for production under the current 3% gate.** Removing the neutral
processor produces a consistent small improvement and exact output, but none of
the representative cells reaches the required core gain. No dependency-specific
sentinel or production branch is retained.

The measurement reveals the next bounded candidate: active MLX-LM repetition,
presence, and frequency processors each use only the last 20 tokens, while
Aster currently converts the full prompt plus completion history to an MLX array
on every row. The next iteration will measure a 20-token context bound under
short and long penalty workloads.

## Verification

- Artifact tests: `2 passed`.
- Ruff: passed.
- Eight measured payloads: exact token/text/cache parity and zero swap growth.
- Rollback: remove the Iteration 054 artifact directory; serving code is unchanged.
