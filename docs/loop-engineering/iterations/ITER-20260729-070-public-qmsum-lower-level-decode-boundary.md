# Iteration 070: Public QMSUM Lower-Level Decode Boundary

- **Date:** 2026-07-29
- **Phase:** rejected
- **Scope:** align a lower-level lazy-MLX decode boundary between Aster and
  direct MLX-LM. No production inference behavior change is in scope.

## Objective

Resolve I069's remaining attribution boundary without turning a diagnostic
measurement into a new synchronization point. The public QMSUM decode driver
gap is stable, but Aster's dominant `sampling_completion_seconds` includes a
lazy MLX completion barrier that does not have a directly comparable private
MLX-LM substep.

## Predeclared Method

1. Instrument only existing source call sites in Aster and direct MLX-LM's
   current generation path. Preserve graph laziness, model calls, sampler
   order, cache mutations, and token materialization semantics.
2. Record aligned submit and mandatory-materialization boundaries only when
   both engines expose the same operation. Keep implementation-private fields
   descriptive and outside cross-engine effect claims.
3. First run a locked public MT-Bench traced/untraced no-op smoke. Require
   exact token/text/finish parity and no material movement in decode throughput,
   end-to-end time, RSS, or swap before running QMSUM.
4. If the smoke passes, repeat the locked QMSUM ABBA schedule with fresh
   process isolation, the same Qwen3.5 model/tokenizer, greedy generation,
   2,048-token prefill step, and bootstrap/order gates.

## Candidate Rules

- If a semantically common lower-level component repeats outside the 3% no-op
  band in both first-engine strata, profile that exact source path before a
  production change.
- If no lower-level common boundary exists without perturbing evaluation,
  retain the conclusion as an irreducible measurement limit and move to the
  public arrival/load matrix rather than optimizing a private timing label.
- Do not enable Gigatoken, Metal SIMD, paged KV, compressed KV, speculative
  decoding, prompt microbatching, or a native backend from this iteration.

## V1 Preflight Rejection

The first recovered MT-Bench smoke covers all 80 locked records and has exact
token, text, and finish parity with zero swap growth. It is rejected as a
no-op admission input: the recovered traced/Aster-untraced shards use adapter
source `e284d6a7...`, while the later recovered direct-MLX-LM untraced shard
uses `6cff1608...`. The latter also started roughly three hours later.

The descriptive deltas therefore do not identify observer cost: Aster peak RSS
median is `+4.896%`, while direct MLX-LM decode throughput, end-to-end time,
and TTFT move `+15.856%`, `-13.480%`, and `-18.103%`. No QMSUM trace run or
runtime candidate is permitted from this input. The ignored raw comparison is
`smoke/lower-level-trace-noop-comparison.json`.

## Frozen V2 Smoke

Freeze `scripts/dev/public_engine_matrix.py` at SHA-256
`dd3c1b4be6ba1c6a2019291391b46fde93d2bc8feb3fe5bd445747f544672730`.
Run four fresh isolated full MT-Bench shards under `smoke-v2/` without source
changes, in this order:

1. Aster untraced.
2. Aster lower-level traced.
3. Direct MLX-LM lower-level traced.
4. Direct MLX-LM untraced.

This puts the trace first for direct MLX-LM and untraced first for Aster. The
same shard no-op gate must pass source, output parity, zero swap, and each
engine-local 3% median metric bound before the QMSUM ABBA schedule starts.

## V2 Result And Conclusion

V2 completed all four fresh isolated MT-Bench shard runs, with 80 locked public
records per engine/condition. The frozen adapter source, public source lock,
model/tokenizer fingerprints, generation settings, execution settings outside
the observer, token IDs, text hashes, finish reasons, and zero-swap condition
all match. The compact recomputation artifact is
`artifacts/ITER-20260729-070-public-qmsum-lower-level-decode-boundary/lower-level-trace-noop-comparison-v2.json`
(7,519 bytes, SHA-256 `f0c49d7ee676fe3985dcbae7461c408734eb154181a3c1c74f4113647ebb3cf7`).

The original comparison incorrectly required source-fingerprint equality across
different engines. Direct MLX-LM legitimately fingerprints its installed
package in addition to the common harness sources. The comparison now requires
an exact traced/untraced fingerprint match for each engine separately; a
focused regression test covers both the engine-specific dependency and a
within-engine mismatch. This correction changes the V2 decision from a false
source rejection to the actual no-op result below.

| Engine | Decode tok/s | End-to-end | TTFT | Peak RSS |
| --- | ---: | ---: | ---: | ---: |
| Aster traced vs untraced median | -3.635% | +3.818% | +7.432% | -7.558% |
| Direct MLX-LM traced vs untraced median | -7.277% | +7.717% | +11.477% | -13.299% |

Every value exceeds the absolute 3% no-op band. The decision is therefore
`trace-no-op-rejected-metric-movement`, not an engine comparison or a runtime
candidate. The observer preserves deterministic output but changes the timed
path materially, principally through Python proxy/wrapper work around source
calls. Do not run the QMSUM ABBA trace, attribute the I069 driver gap to these
substeps, or change production inference from this experiment.

## Research And Next Boundary

Local reference reading confirms the next boundary should be load-shaped rather
than another private decode split. vLLM's scheduler accounts one token budget
across running and waiting requests, schedules active work first, and admits
waiting work only within request/KV limits
(`examples/vllm/vllm/v1/core/sched/scheduler.py:424`). SGLang similarly gates
prefill admission by waiting-queue and pool capacity
(`examples/sglang/python/sglang/srt/managers/scheduler.py:2847`). I071 will
build a public-source arrival/load baseline with controlled B1/B4/B8 arrivals,
staggered long-prefill traffic, shared-prefix traffic, and cancellation before
selecting any scheduler, prefill, or cache candidate. These designs are
referenced as measurement requirements only; no reference code is imported.

An attempted external refresh on 2026-07-29 produced no usable source: the
configured Web search endpoint returned HTTP 404 and read-only GitHub API
queries returned HTTP 403. No remote claim or version update is incorporated in
this iteration.

## Bounded Files

- `scripts/dev/public_engine_matrix.py`
- `tests/test_public_engine_matrix.py`
- `docs/loop-engineering/iterations/ITER-20260729-070-public-qmsum-lower-level-decode-boundary.md`
- `docs/loop-engineering/CURRENT.json`, `STATUS.md`, `DECISIONS.md`,
  `KNOWN_ISSUES.md`, `CORE_REFERENCE_MATRIX.md`, and `FRONTIER_RADAR.md`
- One compact tracked result summary below 5 MiB; raw results remain ignored
  under `run/loop-engineering/`.
