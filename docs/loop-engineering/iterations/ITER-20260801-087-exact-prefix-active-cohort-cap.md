# Iteration 087: Exact-Prefix Active Cohort Cap

- Date: 2026-08-01
- Phase: completed
- Baseline: `03c222ac687d8a5ffe9ef08eedbd943209285391`
- Scope: benchmark and harness control only; no production scheduler/default change

## Objective

Prove or reject a four-request active-lifecycle cap for the locked public
Qwen3.5-9B exact-prefix B8 arrival plan. The candidate must lower physical
memory pressure without hiding the queueing cost paid by the final requests.

## Corrected Boundary

I084's B8 records already ran with `max_decode_batch=4`; Aster's round-robin
decode loop was therefore already issuing 4/3 microbatches. The observed
10.588 GB peak came from seven exact replays being live together, not from one
seven-row decode call. I087 keeps the arrival plan and decode batch cap fixed
and changes only `max_active_requests` from the configured 16 to 4.

This distinction matters: an arrival delay would omit real queueing, while a
decode-only cap would repeat the existing baseline. The active cap leaves all
seven requests submitted at the same time and measures their complete wait,
TTFT, latency, completion spread, and cleanup.

## Reference Mapping

- Feather (`arXiv:2605.06046v1`) reports that smaller prefix-homogeneous
  batches can outperform larger heterogeneous batches by improving KV
  locality. This workload is already fully homogeneous and Aster's native MLX
  merge does not share prefix reads, so I087 tests only the active-width memory
  tradeoff and does not assume Feather's reported throughput gain.
- vllm-metal main is pinned at `b6e35b6c642162dbf6f31009b81635426a91b64a`.
  Its runner follows vLLM scheduler membership and materializes contiguous
  cache state for native MLX paths; it does not supply an Aster-equivalent
  active cohort policy to copy directly.
- SGLang main is pinned at `e1964da451ef9fbec04b326c729916281f90809b`.
  Its scheduler maintains a bounded running batch and prefix-aware waiting
  policy, but the CUDA-oriented runtime is not an execution-equivalent Apple
  Silicon comparator for this exact memory boundary.

## TDD Contract

The first focused run produced three expected failures:

1. the public arrival harness did not accept `max_active_requests`;
2. the cohort-cap matrix module did not exist for the passing-gate case; and
3. it did not exist for incomplete/mismatched-pair rejection.

The green contract adds an opt-in override whose `None` default preserves all
existing behavior. Pure matrix tests require five complete pairs, identical
plan/workload/model/decode width, both execution-order strata, every hard gate,
and explicit rejection below the 97% throughput floor.

## Locked Matrix

- Workload: public QMSUM record, 10,334 prompt tokens.
- Plan: one cold request followed by seven simultaneous exact replays.
- Generation: greedy, eight output tokens, thinking disabled.
- Both lanes: 8 GiB snapshot budget, 256 entries, 512-token decode-active
  prefill budget, reservation trace 64, `max_decode_batch=4`.
- Baseline: configured `max_active_requests=16`.
- Candidate: configured and observed `max_active_requests<=4`.
- Five fresh pairs: baseline-first for pairs 1/3/5 and candidate-first for
  pairs 2/4.

## Hard Gates

1. All 80 requests have one token hash, one text hash, `length` finish, no
   error, and eight decode steps. Every process has seven exact hits, one
   store/entry, zero eviction/preflight/drop, and clean terminal state.
2. Candidate engine-owned active-cache equivalents never exceed four while
   all seven replay requests remain submitted and queue-visible.
3. Paired median peak MLX reduction is at least 10% and 1.0 GB, with both
   thresholds met in at least four of five pairs.
4. Paired aggregate replay throughput ratio is at least 0.97 overall and in
   baseline-first/candidate-first strata.
5. Replay p95 TTFT, p95 latency, and maximum latency ratios are at most 1.03
   overall and in both order strata.

Any failed gate rejects the candidate. A live cancellation/fallback screen and
conditional scheduler implementation run only after every matrix gate passes.

## Production Invariants

The engine scheduler, default active/decode widths, native cache merge, prefix
cache policy, 8 GiB budget, reservation/eviction behavior, persistence, and
rollback controls remain unchanged during this screen.

## Results

All five pairs and both execution-order strata passed. The compact lane
medians are:

| Metric | Baseline active 16 | Candidate active 4 | Paired median ratio/change |
| --- | ---: | ---: | ---: |
| Peak MLX memory | 10.588 GB | 8.551 GB | -2.037 GB / -19.237% |
| Aggregate replay throughput | 12.804 tok/s | 20.353 tok/s | 1.551x |
| Replay p95 TTFT | 2.148s | 1.911s | 0.890x |
| Replay p95 latency | 4.373s | 2.751s | 0.645x |
| Replay maximum latency | 4.373s | 2.751s | 0.645x |
| Peak active-cache equivalents | 7 | 4 | bounded at cap |

The candidate still observed seven submitted replay requests, so its result
includes queueing rather than deleting the delayed cohort. Every pair reduced
memory by the same 2.037 GB. Baseline-first/candidate-first throughput ratios
were 1.864x/1.412x; p95 latency ratios were 0.536x/0.714x; p95 TTFT ratios were
0.890x/0.914x. All 80 matrix requests preserved token, text, finish, cache,
trace, and terminal-state contracts.

The post-pass candidate cancellation process accepted the long-prefill
cancellation, completed the eight-token follow-up, ended with one cancelled /
one completed / zero failed request, zero active/pending/pinned state, and zero
swap delta.

## Decision

Admit the active-cap result as a scheduling candidate and advance a conditional
policy experiment. Do not lower the global default or modify the production
scheduler in I087. The evidence covers one exact-prefix QMSUM B8 shape only;
it neither establishes cap 4 as the global optimum nor proves no regression
for short, distinct-prefix, or mixed-prefix traffic.

I088 must compare caps 2/3/4/5/6 on the exact-prefix workload and add short and
distinct/mixed-prefix guards before selecting a conditional classifier. The
I086 shared-pool kernel remains rejected and is not part of this result.

The retained, source-bound artifact is
`docs/loop-engineering/artifacts/ITER-20260801-087-exact-prefix-active-cohort-cap/active-cohort-cap-summary.json`.

## Verification

- Focused arrival/matrix files: `28 passed`.
- Full suite: `582 passed, 9 skipped, 1 warning`.
- Touched-file Ruff passed; new benchmark/test formatting passed.
- Retained summary recomputes all eight matrix gates, both order strata, the
  post-pass cancellation gate, and four source hashes; it records hashes for
  all eleven raw processes.
