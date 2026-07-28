# Iteration 055: Bound Built-in Penalty Processor Context

- **Date:** 2026-07-23
- **Start/end reference:** `2cb14052d4a3edbcbd420e8bf8d28cfce4a6bba2`;
  production change remains uncommitted in the shared worktree
- **Hardware:** Apple M5, 10 CPU cores, 24 GB unified memory
- **Software:** macOS 27.0 (`26A5388g`), Python 3.14.5, MLX 0.32.0,
  MLX-LM 0.31.3
- **Model:** Qwen3.5-0.8B 4-bit

## Problem and Priority

MLX-LM's built-in repetition, presence, and frequency processors read at most
their configured context window, which defaults to 20 tokens. Aster previously
built `prompt_tokens + output_token_ids` and converted that entire history to
an MLX array for every active row and decode step. The processor then sliced the
array back to 20 tokens. This makes processor overhead grow linearly with
conversation length even though the mathematical dependency is constant.

Iteration 054 also showed that the default `repetition_penalty=1.0` creates a
mathematically neutral processor in MLX-LM 0.31.3. Its isolated `+1.47%` to
`+1.57%` result did not pass the production gate, but omitting that processor is
required to represent the no-processor state accurately in this broader
context-ownership fix.

## Hypothesis and Gates

The candidate assigns one explicit context contract at decode initialization:

- `0` when no logits processor exists;
- `20` when only Aster-configured built-in penalty processors exist;
- `None` when a structured, thinking, or other full-history processor exists.

For active built-in penalties, constructing only the preceding 19 tokens and
letting the runner append the current input token should remove an O(history)
host copy and host-to-device array construction. Success required:

- exact token, text, and cache parity in every process;
- no swap growth;
- 18 fresh processes and 9 runner-assignment-balanced replicates;
- a 96.09%-coverage distribution-free median interval above `+3%` for the
  24,601-token core scenario, including both A/B order strata;
- all but at most one replicate meeting the predeclared block-stability gate;
- short B2/B4 balanced and order-stratified interval lower bounds above `-1%`.

## Design and Implementation

Changed production files:

- `aster/inference/model_runner.py`
  - omits neutral repetition penalty construction;
  - passes an explicit 20-token context to all three built-in penalty factories;
  - exposes the context contract on `DecodeInit` and `DecodeWorkItem`;
  - defensively bounds processor input before `mx.array`.
- `aster/inference/request_state.py`
  - carries the initialized processor context contract for each request.
- `aster/inference/engine.py`
  - constructs only the required preceding token slice without concatenating
    the full prompt and completion for bounded processors.
- `tests/test_processor_context.py`
  - covers the neutral case, active penalties, structured full-history fallback,
    first-token de-duplication, generated-token windows, and runner defense.

Custom structured and thinking processors retain the old full-history contract.
Existing manually constructed work items default to `None`, preserving backward
compatibility. The runner applies the bound again so an alternate runtime cannot
accidentally pass an oversized history to a bounded built-in processor.

The relevant local reference is MLX-LM 0.31.3
`mlx_lm.sample_utils.make_logits_processors`: repetition, presence, and
frequency processors each slice to their configured context size. Aster does
not import dependency-private processor types; it owns the context metadata at
the factory call where the processor behavior is known.

## Test-First Evidence

The initial four focused assertions failed before the production fields and
helpers existed. After implementation:

- processor-context tests: `6 passed`;
- artifact tests: `6 passed`;
- affected runner/runtime/structured suite: `127 passed, 1 deselected`;
- full suite excluding the pre-existing snapshot-budget failure:
  `483 passed, 9 skipped, 1 deselected`;
- unfiltered full suite: `483 passed, 9 skipped, 1 failed`;
- Ruff and `git diff --check`: passed.

The sole full-suite failure is unchanged and outside this iteration:
`test_long_context_snapshot_budget_is_capped_for_clone_headroom` expects the
missing `InferenceEngine._snapshot_budget_for_state` from another worktree
change.

## Benchmark Design

The strict harness keeps two independent KV states on one loaded model and
times adjacent baseline/production calls. It alternates A/B first-call order on
every step and swaps physical runner assignment on odd/even processes. Each
odd/even pair is one assignment-balanced replicate. Every payload binds source,
model, config, input, PID, seed, and runner assignment hashes.

Formal long command:

```bash
.venv/bin/python docs/loop-engineering/artifacts/ITER-20260723-055-bounded-penalty-context/strict_matrix.py \
  --runs 18 --batch-size 2 --context-words 8192 --steps 256 \
  --pair-warmup-steps 16 --model-warmup-tokens 8 --prefill-step 1024 \
  --block-size 16 \
  --output-dir docs/loop-engineering/artifacts/ITER-20260723-055-bounded-penalty-context/results/strict-long-window-r18
```

The fixed input produced 24,601 prompt tokens per lane. Each process measured
256 paired decode steps after 16 paired warmups. The same harness ran 18-process
409-token B2 and B4 matrices for short-context no-regression evidence.

## Formal Results

### 24,601-token active-penalty B2

| Metric | Baseline | Production | Change |
| --- | ---: | ---: | ---: |
| Decode tokens/s, process median | `137.151` | `143.924` | `+4.94%` descriptive |
| Decode step p50 | `14.509 ms` | `13.822 ms` | `-4.74%` |
| Decode step p95 | `20.293 ms` | `18.597 ms` | `-8.36%` |
| MLX peak, process median | `2.068 GB` | shared paired process | no attributable increase |
| Swap delta | `0` | `0` | no growth |

Paired speed distributions:

| Unit | Min | Median | p95 | Max |
| --- | ---: | ---: | ---: | ---: |
| 18 process speedups | `+2.841%` | `+5.017%` | `+8.317%` | `+9.159%` |
| 9 balanced replicate speedups | `+4.215%` | `+5.017%` | `+6.971%` | `+7.139%` |

Strict distribution-free intervals:

| Stratum | 96.09%-coverage median interval | 3% lower-bound gate |
| --- | ---: | :---: |
| Runner-balanced | `[+4.752%, +6.719%]` | pass |
| Baseline first | `[+4.335%, +7.064%]` | pass |
| Production first | `[+4.097%, +5.455%]` | pass |

All 9 replicates passed block stability. All 18 processes retained exact
token/text/cache output and zero swap growth. Across timed and warmup rows, the
candidate represented 242,219,808 logical source tokens with 195,840 device
tokens, a `1,236.8x` reduction. The largest row fell from 24,872 logical tokens
to 20 device tokens.

`final-admission.json` recomputes the long strict gate, both short no-regression
gates, source/model signature equality, exactness, swap non-growth, formal
process count, and retained failed evidence. All nine composite gates pass.

### 409-token no-regression matrices

| Cell | Replicate median | Replicate min/p95/max | Balanced interval | Order lower bounds | Result |
| --- | ---: | ---: | ---: | ---: | --- |
| B2 | `+0.157%` | `-1.164% / +1.100% / +1.214%` | `[-0.365%, +0.927%]` | `-0.941% / -0.566%` | no regression |
| B4 | `+0.523%` | `-0.363% / +1.183% / +1.245%` | `[-0.264%, +1.090%]` | `+0.260% / -0.960%` | no regression |

These cells do not pass the `+3%` speed gate and are not claimed as speedups.
They establish that the long-context specialization does not exceed the
project's 1% short-context regression allowance. All 36 processes retained
exact parity. Swap never grew; one B4 process reclaimed 8 MiB.

## Retained Failed Evidence

The first formal 64-step, 18-process matrix is retained under
`results/strict-ultra-long-r18`. Its runner-balanced interval cleared the 3%
floor at `[+3.674%, +8.237%]`, but the baseline-first and production-first lower
bounds were `+0.937%` and `+2.759%`, and only 6/9 replicates met block stability.
It failed admission. A predeclared four-process 256-step screen then produced
two stable balanced replicates at `+4.995%` and `+4.688%`, motivating the
independent 18-process long-window matrix above. No failed record was removed.

Earlier device-only and work-item screens are also retained at the artifact
root. They establish the source of the gain but are not used for final
admission because their source hashes precede the production implementation.

## Decision

**Retain as a scenario-specific production optimization.** Active built-in
penalty decoding at 24,601 tokens clears every correctness, order, stability,
memory, and 3% performance gate. Short B2/B4 remain within the 1% no-regression
boundary. The result is not a claim about structured processors, custom
processors, prefill, TTFT, or default requests without active penalties.

Power is unavailable because `powermetrics` requires elevated privileges; no
energy claim is made.

Rollback removes `logits_processor_context_size` from decode state/work items,
restores full-history construction in `InferenceEngine._decode_work_item`, and
restores the previous unbounded processor call in `ModelRunner`.

## Next Priority

The remaining sampler graph still normalizes every row and materializes host
token IDs before stop and streaming logic. Iterations 052-054 show that isolated
greedy normalization and neutral-processor changes are below the global gate.
The next iteration should profile output materialization and homogeneous active
penalty rows under B2/B4/B8 without weakening arbitrary structured/custom row
ownership or RNG order.
