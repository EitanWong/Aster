# ITER-20260821-093: Low-Overhead Decode-Stage Attribution

## Baseline

I092 establishes a valid Aster-only B4 observer matrix but rejects the
implementation as production instrumentation. With the observer enabled,
decode changes `-1.063%` (B4-short) and `-4.804%` (B4-mixed); TTFT/e2e p95 and
mixed-load memory also regress. The default remains
`decode_stage_observer_max_events=0`.

## Objective

Measure the same decode-driver ownership boundaries without per-step Python
event dictionaries or multiple `perf_counter()` calls on the hot path. The
result must identify whether cache preparation, model enqueue, sampler graph
construction, lazy evaluation, or result delivery is worth a production
experiment while preserving MLX's existing synchronization behavior.

## Candidate Designs

- Aggregate-only counters with one start/end timestamp per decode call and a
  fixed numeric stage accumulator; no event list or copied context metadata.
- Deterministic periodic sampling (for example one call in 32) with a bounded
  sample count, leaving unsampled calls identical to the disabled path.
- A separate benchmark process trace that correlates existing engine timing,
  decode batch diagnostics, and allocator counters rather than adding hooks to
  ModelRunner.

The first implementation must compare these designs against the I092
observer-off baseline. It must not add an explicit `mx.eval`, change cache
ownership, alter sampling, or change request scheduling.

## Gates

1. Focused unit tests prove disabled mode has no timer/event calls and enabled
   mode remains bounded.
2. A fresh balanced B4-short/B4-mixed matrix has exact output/finish identity,
   clean terminal state, zero fallbacks, and comparable source/input hashes.
3. Observer overhead is below `1%` for decode TPS, TTFT p95, e2e p95, peak MLX,
   and peak RSS in both cells; any failed gate rejects the observer.
4. Only after the no-op screen passes may stage shares guide one minimal
   runtime candidate. That candidate needs a fresh A/B matrix, >=3% repeated
   end-to-end gain, rollback, and cancellation/structured-processor coverage.

## Research Inputs

LLMVisor (`arXiv:2608.08382`) supplies the piecewise attribution framing.
TileMix (`arXiv:2608.17336`) and CoRun (`arXiv:2608.14376`) are current watch
items for tile-level attention precision and fixed-shape batching. Their
reported gains are not transferred to Aster without an Apple/MLX reproduction.
MTP, DFlash, EAGLE-family, and tree speculation remain foundation-gated.

## Implementation

The benchmark-only observer now has a bounded
`engine.decode_stage_observer_sample_interval` setting (`1..1024`, default
`1`). Disabled mode still returns before any observer timer or event work.
Enabled mode samples the first decode call and then every Nth call until the
existing `decode_stage_observer_max_events` bound is reached. The runner keeps
only numeric stage totals and a sampled-step counter; it does not add an
`mx.eval`, change cache ownership, alter sampler inputs, or change scheduling.
`reset_decode_stage_observer_window()` clears diagnostic state after warmup and
before the timed window without touching model or KV state. The runtime-kernel
protocol delegates the reset for the manual path and is a no-op for the
unsupported BatchGenerator adapter.

The reusable harness
[`scripts/dev/benchmark_decode_observer.py`](../../../scripts/dev/benchmark_decode_observer.py)
runs adjacent observer-off/on pairs for Aster and direct MLX-LM, balances two
off-first and two on-first repetitions per cell, validates source/input/output/
terminal/fallback contracts, and reports both engine order strata. The focused
regression suite is
[`tests/test_decode_observer_benchmark.py`](../../../tests/test_decode_observer_benchmark.py).

## Formal Measurement

The locked public-data verification retained source-lock SHA-256
`d6d0877b452ed5627bf0fd39ebc1e59ccad6284cdb4eace27a954603a5211c16`, model
SHA-256 `d77667c10dd92f5f94e7a2b3d290e411dd9564d88940a31286648cfa8b138b2a`,
and tokenizer SHA-256
`94b66525e309d7ce24691be8194369f880e4f8a5ba82b726782e70fc97e1559e`.
The adjacent matrix contains 32 successful rows: B4-short and B4-mixed,
Aster and MLX-LM, observer off/on, four repetitions, with two state orders.
Source comparability, exact output/finish identity between observer states,
clean terminal state, zero decode fallbacks, zero swap, and bounded observer
events all pass.

| Cell | Aster observer-off -> on decode TPS | Paired Aster decode delta | Aster observer-on samples | Diagnostic stage shares (cache / model / sampling / eval / result) |
| --- | ---: | ---: | ---: | ---: |
| B4-short | `60.516900 -> 60.354442` | median `-0.275%` | `2` per repetition | `6.531% / 1.936% / 0.013% / 91.476% / 0.037%` |
| B4-mixed | `38.796075 -> 38.007012` | median `-2.032%` | `3` per repetition | `3.258% / 1.993% / 0.012% / 94.678% / 0.033%` |

The four B4-short paired decode deltas are `+0.599%, +1.530%, -1.150%,
-1.514%`. B4-mixed is `-32.011%, -2.375%, -1.689%, +8.740%`; its Aster
order strata are `-16.850%` (off-first) and `+3.183%` (on-first). The mixed
observer-on peak-MLX order strata are `-0.149%` and `+10.816%`, so the strict
`<1%` no-op gate is false even though all correctness and resource-safety
contracts pass. The MLX-LM control also has mixed decode order strata of
`+7.422%` and `+5.522%`, with corresponding tail/RSS variation. The harness
therefore marks the measurement valid but explicitly confounded by control
variance; it does not treat the favorable mixed aggregate row as a speedup.

The observer-off full cross-engine baseline has 32 rows and source-comparable
inputs. Relative to direct MLX-LM, Aster's decode-driver deficit is `+34.432%`
in B4-short and `+26.539%` in B4-mixed. This is a scoped B4 baseline, not a
global engine ranking; the public inventory still records unavailable engines
and their reasons.

## Decision

**Reject** periodic sampled observation as admitted production instrumentation.
Keep `decode_stage_observer_max_events=0` and the sampling interval
benchmark-only. The data is sufficient to locate the measured boundary but
not to select a runtime optimization: the mixed-load no-op gate fails in Aster
state strata, and the control engine moves materially under the same adjacent
state schedule. No cache, sampler, model, scheduler, or default configuration
change is admitted in I093. The sampled stage shares are diagnostic evidence
only, not a private-kernel attribution or an end-to-end gain claim.

## Reproducibility and Rollback

The compact evidence is archived at
[`docs/loop-engineering/artifacts/ITER-20260821-093-low-overhead-decode-stage-attribution`](../artifacts/ITER-20260821-093-low-overhead-decode-stage-attribution/).
The recomputable matrix is
`decode-stage-observer-sampled-matrix.json` (SHA-256
`f64adc494134c63b046a4ed4606bd7bc1fbe3efd0b43eb2ca0ca25d6620f31b5`). The
raw 32-row collection remains at
`/tmp/aster-i093-paired-final/matrix.json` (SHA-256
`9262ed8e5f48c356a62140f51a6f65aae28a71e405cf4a6d9a3a513d25628558`).
`BASELINE_FILE`, `MODIFIED_FILE`, `DIFF_FILE`, executable `ROLLBACK.sh`, and
`VERIFICATION.txt` record the exact field change, hashes, commands, focused
test result, and a successful rollback on `ROLLBACK_COPY`; the modified copy
is intentionally left changed.

The independent interval-16 matrix collected before the timed-window reset is
retained only as a failed measurement attempt. It is not combined with the
formal result. The final artifact uses `--reuse-existing` solely to recompute
the summary from the unchanged raw rows.

## Reference Refresh

The source review refreshed MLX `27fec909a3df9e572f5195607a453e273e7d80d0`,
MLX-LM `d06c5374a12e1f9384aad5fece583d7be9d2619d` (upstream observation:
MLX 0.32.1 update on 2026-08-19), SGLang
`0f744b684836edadb0b6ab18d6dd4beda457ccb2`, vLLM
`bfb6c134997aace3e801c9ae3251728bd5312003`, vLLM-MLX
`8c814e30f54ee2a8e06acf768713cf0f24e22850`, vllm-metal
`67100ba77780dec48adeb569724efaf8fe928b19`, and local llama.cpp
`0e1d9185c5fe82e905d1f5ae6b2e5dcd607a8dfd`. The fetched upstream llama.cpp
branch tip is `0e1d9185c5fe82e905d1f5ae6b2e5dcd607a8dfd`; its recent backend
change gates compiler-specific workarounds because measured throughput can
regress. The prior `6503355d` observation is retained as historical metadata.
Seven reachable submodules advanced to their fetched branch tips; unavailable
endpoints retain their pins in `REFERENCES.md`. LLMVisor (`2608.08382`) remains
the attribution method input;
TileMix (`2608.17336`), CoRun (`2608.14376`), QEvict (`2608.05326`), QUASAR
(`2608.13966`), LibraSpec (`2608.08721`), HYMELL (`2608.06723`), and DBLAST
(`2608.05448`) are watch-only research inputs with no Aster code transfer.

## Next Iteration

I094 is `ITER-20260822-094-mixed-load-attribution-stability`. Its single
objective is to separate host/control variance from decode-stage overhead by
using longer generation windows, repeated adjacent state pairs, and explicit
control-engine stability gates. It must retain exact B1/B4 cross-engine
coverage, require stable order strata and zero swap/fallbacks, and admit no
runtime change without a repeatable `>=3%` end-to-end gain. MTP and other
multi-token/speculative paths remain deferred until this foundation boundary,
rollback, sampler, cancellation, and mixed-load evidence is closed.
