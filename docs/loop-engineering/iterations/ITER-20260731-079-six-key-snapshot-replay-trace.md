# Iteration 079: Six-Key Snapshot Replay Trace

- Date: 2026-07-31
- Phase: admitted for balanced multi-window screening
- Baseline commit: `d69557e1b1801cf47619b2bbe2978d36e356e661` plus the
  SHA-bound I077 observer
- Scope: widen source-bound reuse distance before another budget candidate;
  no production cache default or policy change.

## Problem

I078 proves 3 GiB retains four distinct QMSUM snapshots plus first-record
replay, but bounded caches trade capacity against reuse distance. A four-key
order does not represent a longer agent session and cannot select a global
default. The next candidate must start from a wider traced control rather than
extrapolation.

## Hypothesis

The configured 8 GiB control will retain six distinct locked QMSUM snapshots
and replay the first exactly. Its per-decision trace will expose whether a
lower configured budget has enough target-store headroom for that wider reuse
distance.

## Predeclared Work

1. Add a deterministic `capacity-replay-six` arrival plan using six distinct
   resolver-owned QMSUM records followed by the first-record replay, with a
   completed-dependency chain and focused plan/execution tests.
2. Run one source-bound 8 GiB traced control with manual runtime, concurrency
   2, greedy 8-token output, 512-token decode-aware prefill, trace capacity 64,
   and disabled persistence.
3. Require all seven terminal identities, zero active state, zero preflight
   skips, and an exact zero-prefill replay before using the trace to select a
   lower-budget candidate.
4. Preserve every eviction as evidence. Do not change the production default
   or cache policy in this baseline iteration.

## TDD Contract

Three focused boundaries were added before the implementation. They failed
because `capacity-replay-six` was not registered, then passed after the minimum
harness-only change:

- six distinct source-owned QMSUM identities followed by replay of the first;
- a completed dependency chain across all seven requests;
- explicit failure when fewer than six distinct QMSUM records are available.

The existing dependency executor test now covers the seven-key submission
order. The full arrival-load file passes `14` tests and Ruff passes for the
harness and its test file. No engine or cache-policy source changed in I079.

## 8 GiB Control

The locked-source control completed all seven requests with matching first and
replay token/text hashes, six retained snapshots / `2,846,359,552` bytes, zero
store evictions, zero preflight skips, zero active requests, and an exact
zero-prefill replay at `0.172272s` TTFT. All seven prompt-free trace events
were accepted and none were dropped.

The observed per-decision lower bound is
`max(store_bytes_before + clone_reserve_bytes) = 3,626,565,632` bytes. A 3 GiB
budget would fall below that bound on the final replay reservation, while 4
GiB leaves `668,401,664` bytes of configured headroom. Therefore 4 GiB is the
only lower-budget candidate admitted to this iteration.

The control's host-global workload swap increased `479,264,768` bytes. The
meter is not process-owned, so it is retained as resource context and blocks
any memory-benefit claim from this single pair.

## Candidate Gates

Run one fresh 4 GiB process under the identical source, model, generation,
runtime, concurrency, output, prefill, persistence, and trace contract. It
must match all seven control terminal workload IDs, completion counts,
finishes, token hashes, and text hashes; retain six snapshots at or below 4
GiB; record zero evictions and preflight skips; end with zero active requests;
and preserve an exact zero-prefill first-record replay. Every trace event must
be accepted, bounded, prompt-free, and have store-before bytes at or below its
target. Timing and host-global swap are context only.

## Results

The candidate passed every predeclared gate. Source and plan are identical to
the control, and the execution contracts differ only in the configured
snapshot budget.

| Metric | 8 GiB control | 4 GiB candidate |
|---|---:|---:|
| terminal identities | 7/7 | 7/7 exact |
| snapshots / bytes | 6 / 2,846,359,552 | 6 / 2,846,359,552 |
| store/reservation evictions | 0 | 0 |
| preflight skips | 0 | 0 |
| replay prefill steps | 0 | 0 |
| replay TTFT, context only | 0.172272s | 0.167437s |
| elapsed, context only | 114.010998s | 114.019547s |
| workload host-global swap, context only | +479,264,768 | +284,557,312 |

All seven candidate events are accepted, none are dropped, every
store-before value is below its target, and every effective budget is at or
below 4 GiB. The engine ends with zero running, waiting, pending, or failed
requests. Candidate replay TTFT moves `-2.807%` and elapsed moves `+0.007%`;
neither single-pair value is a performance result. Candidate/control process
RSS baselines differ materially, so RSS also remains context only.

## Verification

- Public source verification passed with lock SHA-256
  `d6d0877b452ed5627bf0fd39ebc1e59ccad6284cdb4eace27a954603a5211c16`.
- The focused red run failed three new boundaries before implementation; the
  complete arrival-load file then passed 14 tests.
- The affected config/engine/arrival suite passed 80 tests and touched-file
  Ruff passed.
- The full suite passed `549 passed, 9 skipped, 1 warning`.
- The compact artifact and both ignored raw results are SHA-bound at
  `artifacts/ITER-20260731-079-six-key-snapshot-replay-trace/six-key-snapshot-replay-admission.json`.

The machine remained macOS 27.0 build 26A5388g, Apple M5, 10 logical CPUs,
25,769,803,776 bytes unified memory, Python 3.14.5, MLX 0.32.0, and Apple clang
21.0.0. Energy measurement was unavailable.

The structured eviction-observability design remains the already reviewed
I077 vLLM/SGLang boundary. I079 adds only a public-source harness scenario and
does not adopt or alter allocator, radix, or eviction code.

## Decision

Admit 4 GiB only to a fresh balanced multi-window screen. It passes this
six-key retention chain and lowers the experiment ceiling by 50%, but one
ordered source sequence cannot justify a production default. The production
8 GiB value, two-clone reserve, eviction policy, entry limit, and snapshot
representation remain unchanged. No cross-engine or global performance claim
is made.

## Rollback

No runtime rollback is required. The 4 GiB value was a process-local harness
override. Revert the `capacity-replay-six` scenario and its tests to remove the
I079 measurement surface.

## Next

I080 adds an opt-in source-window offset, then runs four new disjoint six-key
QMSUM windows in fresh processes with budget order `8/4, 4/8, 8/4, 4/8`.
Any candidate replay loss, eviction, preflight skip, output drift, cleanup
failure, or source/plan mismatch rejects 4 GiB. I079's favorable pair is not
reused as formal evidence.
