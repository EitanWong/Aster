# Iteration 064: Short Decode Call Position Classification

- **Date:** 2026-07-28
- **Phase:** complete
- **Scope:** benchmark infrastructure only; no production runtime candidate is
  authorized until the first/second-call interaction is classified.

## Objective

Separate same-variant call-position effects from the serial/pipeline difference
in the frozen Qwen3.5-0.8B 128-word / 256-token greedy short-decode workload.

## Primary Metric

For fresh prompt caches in one process, measure the percentage change from the
first to second decode call for serial/serial and pipeline/pipeline pairs. The
mixed serial/pipeline order remains a counterfactual, not the primary effect.

## Hypothesis

If call position is a material contributor to I062/I063's sign reversal,
same-variant pairs will retain a consistent first/second-call timing difference
under the same frozen model, inputs, sampler, output cap, and prewarm protocol.

## Predeclared Gates

1. Preserve I061/I062 raw prompt IDs, model files, greedy sampling, 256-token
   cap, exact completion IDs/finish, and non-growing swap.
2. Use fresh prompt caches for both calls and record call position, allocator
   state, and available host state before and after each call.
3. Balance serial/serial and pipeline/pipeline independent-process records;
   retain the existing mixed-order result as a separately labeled counterfactual.
4. Report every order and call-position stratum. A production candidate remains
   blocked unless a later stabilized protocol clears the 3% effect floor and
   I061's long-context counterfactual.

## Results

Eight independent processes crossed serial/pipeline variant with two terminal
prewarm orders and two repeats. Every same-variant pair allocated a fresh prompt
cache for both calls, generated the same 256 IDs and `length` finish, and did
not grow swap.

| Variant | Second vs first elapsed gain p50 | Bootstrap median interval |
| --- | ---: | ---: |
| Serial | -25.955% | [-38.293%, -17.931%] |
| Pipeline | -28.110% | [-45.824%, +4.869%] |

Seven of eight second calls were slower. The one positive pipeline observation
does not reverse its p50, but it prevents a claim that the magnitude is fully
stable. The shared negative serial/pipeline medians show that call position is
a material measurement confound. This explains why I062/I063's pipeline-first
and serial-first strata reverse, without proving that a specific allocator,
stream, thermal, or hardware mechanism is responsible.

## Evidence And Verification

- `call-position-evidence.tar.gz`: 8 minimal raw records / 23,097 bytes /
  SHA-256 `efa181f3bf7c1f84b5a4d48f8e0ac460f44033e2f4abcce2a72f3ba4ba9e0738`.
- `screen-summary.json`: source-bound rejection and per-variant summaries.
- `test_i064_artifacts.py`: source/hash validation and archive-only aggregate
  recomputation passed (`2 passed`).
- All work was benchmark-only; Aster production and test sources were unchanged
  during I064.

## Conclusion And Next Priority

Decision: **reject** adjacent same-process serial/pipeline comparisons as
evidence for an asynchronous runtime candidate. I065 will audit I061's
isolated-process records and make its timed-call/prewarm boundary explicit
before selecting another measured production bottleneck.

## Retained Artifact Set

- `call_position_probe.py`: source-bound benchmark-only collector.
- `call_position_aggregate.py`: exactness, independent-PID, source/model, and
  balanced-strata verifier.
- One compressed raw-record archive, one summary, and archive-only tests.
