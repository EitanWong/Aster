# Iteration 062: Short Decode Runtime Profile

- **Date:** 2026-07-28
- **Phase:** complete
- **Scope:** profile and benchmark-only investigation; no production candidate
  is authorized until a measured source-level bottleneck clears the 3% screen.

## Objective

Explain the I061 128-word prompt / 256-token completion result in which Aster
was 5.743% below direct MLX-LM at paired p50 despite exact token/text/finish
parity, while Aster was ahead after a 2,048-word prompt.

## Primary Metric

Attribute single-request short-context decode wall time to model forward,
sampling, logits processing, Python work-item/result construction, and stream
handling without changing output tokens or the I061 frozen workload.

## Hypothesis

One or more Aster manual-runtime per-token host costs are large enough at short
context to explain a material fraction of the I061 delta, but amortize after a
long prefill. A benchmark-only trace can identify whether a production screen
is justified.

## Predeclared Gates

1. Reuse the I061 model, raw prompt IDs, greedy sampler, 256-token cap, warmup,
   process isolation, and exact token/text/finish/swap checks.
2. Instrument only benchmark/profiling code; do not change Aster production
   sources in the profile phase.
3. Report median and p95 component shares across independent processes and
   preserve every observation.
4. Only a component with a credible >=3% end-to-end opportunity may authorize
   one bounded production candidate in a later phase.
5. Keep the long-context result as a counterfactual check; a short-context
   improvement must not regress its exact-output or resource gates.

## Profile Results

### Work-Item Token History

I061's hand-written decode harness copied the full prompt/completion history
into `DecodeWorkItem.logits_processor_tokens` even though the frozen request
has no logits processor and production `InferenceEngine` passes `[]` when its
context size is zero. Two independent opposite-order records matched all 256
completion IDs, text, finish reason, model hashes, and swap observations.

| Record order | Decode-loop change after matching engine semantics | Legacy list-build share |
| --- | ---: | ---: |
| full-history then runtime-context | -0.052% | 0.057% |
| runtime-context then full-history | -0.246% | 0.055% |

The allocation is real but two orders place it far below the 3% screen floor.
It does not explain the I061 short-context cross-engine result and does not
justify a production change.

### MLX-LM Lookahead Pipeline

Direct MLX-LM's `generate_step` schedules a next-token graph before forcing the
previous token to the host. A first independent-process screen showed large but
unstable gains, so it was replaced by a stronger six-process paired design:
each process warmed serial and pipeline graphs, then measured both with fresh
prompt caches; serial-first and pipeline-first occurred three times each.
Every pair retained exact token IDs, `length` finish, stable inputs/sources,
and non-growing swap.

| Metric | Result |
| --- | ---: |
| Pipeline elapsed gain p50 | +9.013% |
| 95% bootstrap median interval | [-27.195%, +38.942%] |
| Serial-first gain p50 | -27.123% |
| Pipeline-first gain p50 | +30.879% |
| Pair range | [-27.266%, +47.005%] |

The order strata reverse sign, the interval crosses zero, and individual
pairs regress. The lookahead design is therefore **rejected as a production
candidate in this iteration**, despite exact output parity. It remains a
reference design to revisit only after the measurement boundary is stabilized.

## Evidence And Verification

- `rejected-screen-evidence.tar.gz`: 13 minimal raw records / 5,996 bytes.
- `screen-summary.json`: source-bound rejection, archive SHA-256
  `056e3e05bac49ab5dfb482d271fd60f22afa178e5844ba1f6487a3611f58dd80`.
- `test_i062_artifacts.py`: source/hash validation and archive-only pipeline
  aggregation recomputation passed (`2 passed`).
- All work was benchmark-only; no Aster production or test source changed.

## Conclusion And Next Priority

Decision: **reject** both short-decode screen candidates. The next iteration
will stabilize the short decode measurement boundary itself by separating
first/second-call, allocator/cache, stream, and thermal effects before
reopening an asynchronous runtime proposal.
