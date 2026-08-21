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

## Implementation

The public foundation harness now accepts an explicit benchmark-only
`--max-output-tokens` value. The value propagates through the frozen public
cohort plan, both Aster and direct MLX-LM child processes, the execution
contract, and the completion-length gate. The default remains `8`, so existing
I090-I093 baselines are unchanged. The observer harness records the requested
cap and rejects a matrix whose rows use a different cap.

No production inference code, scheduler policy, cache ownership, sampler, or
MLX evaluation boundary changed in I094.

## Formal Measurement

The locked public source remained unchanged:

- workload SHA-256: `d6c7fa000ec3daca7a9756f906ab997624b678bfa9128949a7630fc4a9444e46`
- source-lock SHA-256: `d6d0877b452ed5627bf0fd39ebc1e59ccad6284cdb4eace27a954603a5211c16`
- model SHA-256: `d77667c10dd92f5f94e7a2b3d290e411dd9564d88940a31286648cfa8b138b2a`
- tokenizer SHA-256: `94b66525e309d7ce24691be8194369f880e4f8a5ba82b726782e70fc97e1559e`
- common benchmark source SHA-256: `967d0c1367b10352ba432568b7eda7ff4065d9ae70fbfab7bbfff131316dbebe`
- Aster engine source SHA-256: `7a3bda525247d2d04504a126713f301cee78062f8ea445a542d1d0b224216e0c`

The matrix used `max_output_tokens=32`, observer-off/on, sample interval `8`,
four repetitions, two off-first and two on-first state orders, fresh processes,
and one-second inter-process cooldown. All 32 rows completed exactly 32
tokens. Source comparability, exact observer-state output/finish identity,
clean terminal state, zero decode fallbacks, zero swap, zero dropped events,
and bounded event counts passed.

| Cell | Aster observer-off -> on decode TPS | Paired Aster decode delta | Aster order strata | Observer-on samples |
| --- | ---: | ---: | ---: | ---: |
| B4-short | `81.406115 -> 81.652300` | median `+0.613%` | `+1.594% / +0.160%` | `5` per row |
| B4-mixed | `46.954306 -> 49.853883` | median `+6.393%` | `-2.174% / +30.592%` | `7` per row |

The mixed result is not a speedup claim: the four paired deltas are
`-5.593%, +49.643%, +1.245%, +11.541%`, and the direct MLX-LM control has
decode strata `-3.114% / -3.438%` plus TTFT/e2e strata reaching roughly
`+7%`. The strict `1%` stability gate is false for both the Aster observer and
the control. Median sampled stage shares are evaluation `95.501%` (short) and
`96.565%` (mixed), model enqueue `1.776%/1.726%`, and cache preparation
`2.662%/1.657%`; these remain diagnostic boundary shares, not private-kernel
claims.

The same 32-token observer-off rows provide a length-sensitive cross-engine
baseline: Aster's decode-driver TPS is `12.693%` higher than MLX-LM in B4-short
and `36.373%` higher in B4-mixed. This reverses the short-window relationship
seen in I093 and is explicitly a workload/window-scoped result, not a global
engine ranking.

## Decision

**Reject** the sampled observer as production instrumentation and keep
`engine.decode_stage_observer_max_events=0` by default. The longer window
validates exact behavior and reduces fixed-window ambiguity in B4-short, but
mixed-load control variance remains too large to attribute the result to the
observer or to choose a runtime optimization. No production candidate is
admitted and no inference default changes.

## Evidence and Rollback

The compact evidence is archived at
[`docs/loop-engineering/artifacts/ITER-20260822-094-mixed-load-attribution-stability`](../artifacts/ITER-20260822-094-mixed-load-attribution-stability/).
The matrix is
`mixed-load-attribution-stability.json` with SHA-256
`2cddae9bf06f6fb129a2b86893c76417b76b50bd7c307594ba713b118b2f7fb4`.
The raw 32-row collection is `/tmp/aster-i094-long-final/matrix-i094.json`
with SHA-256
`4eaad8503963788e590a71c0c55cc91170ba7010b154a9ba55420d6aa8447334`.
`BASELINE_FILE`, `MODIFIED_FILE`, `DIFF_FILE`, executable `ROLLBACK.sh`,
`ROLLBACK_COPY`, and `VERIFICATION.txt` bind the `8 -> 32` benchmark cap,
record exact hashes and commands, and prove rollback while leaving the
modified fixture changed.

## Reference and Research Refresh

I094 uses the refreshed local heads recorded in
[`REFERENCES.md`](../REFERENCES.md), including llama.cpp `0e1d9185`, vLLM
`bfb6c134`, SGLang `0f744b684`, vLLM-Metal `67100ba7`, vLLM-MLX `8c814e30`,
OMLX `fa3e94b3`, MLX-LM `d06c5374`, and MLX `27fec909`. LLMVisor
(`2608.08382`) remains the attribution method input. TileMix (`2608.17336`),
CoRun (`2608.14376`), QEvict (`2608.05326`), QUASAR (`2608.13966`), LibraSpec
(`2608.08721`), HYMELL (`2608.06723`), and DBLAST (`2608.05448`) remain
watch-only; no code or unverified paper claim entered Aster.

## Next Iteration

I095 is `ITER-20260823-095-decode-boundary-control`. Its objective is to
classify the host/control state at the common decode boundary before any new
observer or runtime candidate. It must predeclare the state controls, retain
the public B1/B4 cross-engine baseline, require `<=1%` primary/order-strata
variance, and admit no production change without exact semantics, rollback,
resource gates, and a repeatable `>=3%` end-to-end gain.
