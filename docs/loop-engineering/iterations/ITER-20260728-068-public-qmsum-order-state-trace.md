# Iteration 068: Public QMSUM Order-State Trace

- **Date:** 2026-07-28
- **Phase:** completed
- **Scope:** diagnose I067's order-sensitive public `longbench-qmsum` result
  with source-bound repeat blocks and measurement-only state telemetry. No
  production inference source changes are in scope.

## Objective

Determine whether the I067 `qmsum` direction reversals are explained by
repeatable block order, observable host/process state, or residual unmeasured
variation before attributing Aster's public prefill gap to a runtime component.

## Hypothesis

If a block-level state interaction is present, four balanced fresh-process
public repeats will retain the order contrast while the outside-timing state
trace identifies which recorded state variables co-vary with it. If the effect
does not reproduce, I067 remains a measurement-boundary warning rather than a
component claim.

## Predeclared Method

1. Extend the public matrix tooling with an outside-timing process/host-state
   record. It must capture block/order identity, process launch/finish time,
   PID, load average when available, available memory, RSS/swap baseline and
   post-run values, and no prompt text. It must not change tokenization,
   generation, metric timing, or engine execution.
2. Reuse all 200 pinned public `longbench-qmsum` records and the existing
   model, source lock, 2,048-token prefill step, and greedy contract. Run four
   independent fresh-process blocks in the predeclared ABBA order:
   Aster-first, MLX-LM-first, MLX-LM-first, Aster-first.
3. Validate deterministic input/output token parity, metrics, model/tokenizer,
   execution contract, RSS, and swap for every block. Analyze paired effects
   by block and first-engine stratum; retain the host/process trace as context,
   never as a causal proof by itself.
4. Report whether the I067 reversal repeats. No production candidate is
   eligible in I068; its sole output is a bounded measurement-state diagnosis
   that can safely select a later component trace or another repeat design.

## Gates

1. All 1,600 engine-records (200 public records x 2 engines x 4 blocks) pass
   the public comparability gates and have zero output-token drift and zero
   swap growth.
2. Every block is fresh-process isolated; each engine is first in exactly two
   blocks. Telemetry is collected only outside the timed request interval.
3. A claimed reproduced order interaction requires the same signed effect in
   both blocks of an order stratum and 95% bootstrap intervals outside the 3%
   no-op band. Otherwise it remains inconclusive.
4. Any parity, metric, source, or execution-contract drift invalidates the
   diagnostic result. No production inference change is admitted from this
   iteration.

## Results

- All four fresh-process ABBA blocks completed: 1,600 engine records over the
  same 200 public QMSUM rows. Every cross-block source, effective-input,
  model/tokenizer, execution, deterministic-token, metric, state-trace, and
  zero-swap gate passed.
- The I067 QMSUM direction reversal did not recur. Decode throughput was
  consistently lower for Aster: `-7.775%` when Aster ran first (95% bootstrap
  `[-8.042%, -7.682%]`) and `-8.216%` when MLX-LM ran first
  (`[-8.408%, -8.008%]`). End-to-end time was consistently higher for Aster:
  `+5.452%` / `+5.352%` by the same two order strata.
- Prefill throughput (`-1.982%` / `-1.514%`) and TTFT (`-1.457%` /
  `-1.928%`) stayed inside the declared 3% no-op band. Peak RSS was not
  reproducible across the order strata.
- The outside-timing trace contains PID, launch/finish time, available memory,
  system swap, and process RSS. Load average and process CPU times were
  unavailable on this host. The trace is retained as context, not a causal
  explanation; zero per-record swap growth still passed.

## Decision

Keep production code unchanged. The only eligible next step is a source-bound
decode component trace: cache merge/rebuild, model forward, sampling/processor
work, materialization, and delivery boundaries must be measured before choosing
any scheduler, cache, SIMD, native-kernel, or tokenizer change. The compact
summary is `artifacts/ITER-20260728-068-public-qmsum-order-state-trace/state-trace-summary.json`;
raw results remain in the ignored run directory.

## Bounded Files

- `scripts/dev/public_engine_matrix.py`
- `tests/test_public_engine_matrix.py`
- `docs/loop-engineering/iterations/ITER-20260728-068-public-qmsum-order-state-trace.md`
- `docs/loop-engineering/CURRENT.json`, `STATUS.md`, `DECISIONS.md`, and
  `KNOWN_ISSUES.md`
- One compact tracked summary below 5 MiB; raw results remain ignored under
  `run/loop-engineering/`.
