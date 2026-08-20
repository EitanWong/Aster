# ITER-20260822-094: Mixed-Load Attribution Stability

## Entry From I093

I093 produced valid sampled stage evidence but rejected the observer as
production instrumentation. The B4-mixed Aster decode delta had a `-16.850%`
off-first versus `+3.183%` on-first order split, while the MLX-LM control also
varied by more than `5%` in both order strata. A longer, state-balanced window
is required before the remaining Aster/MLX-LM decode-driver gap can be
assigned to an implementation owner.

## Objective

Separate host/resource variance from decode-stage overhead using longer
generation windows and explicit control-engine stability. The primary metric is
`decode_driver_tps`; secondary metrics are aggregate generation TPS, TTFT p95,
end-to-end p95, peak MLX/RSS, and swap delta.

## Hypothesis

If the sampled observer is genuinely low overhead, its paired effect will stay
within `1%` across longer B4-short and B4-mixed windows, both state orders, and
the MLX-LM control. If the control remains unstable, the correct result is a
measurement boundary rather than a runtime change.

## Design

- Keep the locked public model, tokenizer, workload IDs, prompt reconstruction,
  greedy settings, prefill chunk, cache-off state, and source hashes from I093.
- Use fresh processes and adjacent off/on calls for Aster and direct MLX-LM.
- Increase timed output length enough to amortize fixed setup while retaining a
  short-window control; predeclare the exact token cap in the artifact.
- Use at least four repetitions per cell, two off-first and two on-first, and
  preserve every raw row including failures and resource excursions.
- Reset the observer window after warmup, verify no explicit `mx.eval`, and
  retain the zero-event disabled fast path.

## Gates

1. All source, input, execution, token/text, finish, terminal, fallback,
   cancellation, and swap gates pass.
2. Aster and MLX-LM control order strata remain within `1%` for the primary
   metric and within the declared resource ceilings for every cell.
3. Observer-on event counts remain bounded and observer-off remains empty.
4. No production candidate is admitted unless the stage attribution is valid
   and a fresh candidate A/B matrix shows a repeatable `>=3%` end-to-end gain
   with exact behavior and rollback.

## Reference Boundary

Use the refreshed MLX/MLX-LM lazy-evaluation and grouped-sampling sources as
the direct control contract. Use llama.cpp's current timing/KV graph comments,
LLMVisor's piecewise attribution framing, and the QEvict/TileMix/CoRun/QUASAR/
LibraSpec/HYMELL/DBLAST papers only to formulate measurable hypotheses. Do not
port CUDA, proprietary, training-only, or model-head-specific mechanisms.

## Deliverables

- One raw state-balanced matrix and a compact recomputable summary.
- A baseline/candidate delta ledger with order strata and control variance.
- Exact output/finish/terminal/resource/rollback verification.
- A single `admit`, `reject`, or `defer` decision and one bounded next target.

MTP, DFlash, EAGLE-family, tree speculation, and multi-token prediction remain
deferred until this stability boundary and the wider sampler/KV/rollback gates
are closed.
