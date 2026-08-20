# Loop Engineering Iteration Protocol

This is the short operating contract for future Aster performance work. The
long-term objective remains in `docs/LOOP_ENGINEERING_GOAL.md`; the only current
state is `docs/loop-engineering/CURRENT.json`.

## Priority Order

1. Restore a bounded, attributable workspace and a green correctness baseline.
2. Finish or reject the single active performance hypothesis.
3. Address the next measured bottleneck.
4. Expand missing workload coverage before making broader performance claims.

Do not start a second production candidate while the active iteration is in
`tdd`, `formal`, or `consolidate` phase.

## Iteration Phases

`recover -> baseline -> profile -> screen -> tdd -> formal -> consolidate`

- `recover`: read `CURRENT.json`, inspect Git state, and run the workspace check.
- `baseline`: freeze model, inputs, configuration, source hashes, and metrics.
- `profile`: measure the current path and identify one dominant cost.
- `screen`: use a benchmark-only candidate to test whether the effect can clear 3%.
- `tdd`: predeclare correctness and invalidation behavior, then implement the minimum change.
- `formal`: run independent processes, balanced AB/BA assignment, exactness, memory, and stress gates.
- `consolidate`: admit or reject, update state/docs, remove scratch duplication, and prepare one reviewable change boundary.

Every phase change updates `CURRENT.json`. An interrupted agent resumes from its
recorded `phase` and `next_actions`; it does not repeat completed acquisition or
screening.

## Scientific Method

Before a candidate changes production code, record:

- one falsifiable hypothesis and one primary metric;
- the control and independent variable;
- fixed model, prompt, output length, seed, cache state, and environment;
- predeclared sample count, order balancing, success floor, memory ceiling, and rollback condition;
- known confounders such as warmup, compilation, desktop load, swap, thermal state, and second-call effects.

Use the 3% core gate as an effect-size floor, not as proof by itself. Formal
admission also requires independent process repeats, balanced AB/BA strata,
intervals, exact output, stable memory, and no discarded observations. Scenario
results must stay scenario-scoped; percentages from different iterations are
never added together.

## Performance Baseline Ledger

Every iteration must leave a recomputable performance baseline, even when the
iteration is diagnostic and makes no production change. The iteration record
must name:

- the baseline and candidate/reference boundary, primary metric, and unit;
- absolute baseline/candidate values, signed delta, and relative percentage;
- process/repetition count, order strata, workload scope, and aggregation rule;
- correctness, terminal-state, memory, swap, and measurement-validity gates;
- the decision (`admit`, `reject`, or `defer`) and the one measured next target.

For a diagnostic whose timing is invalidated (for example, forced logit
evaluation), record `measurement_status: invalidated`, a null timing delta, the
reason, and a linked valid baseline. Such a record must not claim a speedup or
slowdown; the next iteration must restore a valid timed boundary before a
production decision. A benchmark-only screen may select a follow-up profile,
but it cannot change production defaults without the full gates.

## Public Data and Cross-Engine Gate

- Cross-engine performance evidence begins with
  `docs/loop-engineering/benchmarks/public-dataset-lock.json`, not a locally
  invented prompt. Run `scripts/dev/public_benchmark.py sync` once to download
  the pinned data under ignored `run/loop-engineering/public-benchmarks/`, then
  run `verify` before each new matrix.
- The current lock contains FastChat MT-Bench and LongBench v1 plus LongBench's
  official prompt templates and output limits. The lock records immutable
  revision, source URL, SHA-256, size, and structural validation rules.
- Generate an engine inventory before a matrix. Every locally available,
  compatible engine belongs to the required-engine set. An unavailable engine
  needs a recorded reason such as absent runtime, absent model format, or failed
  same-model/tokenizer equivalence; it must not disappear from the comparison.
- Generate workload manifests from public source record IDs and prompt hashes.
  `cross-engine-core` is a scoped diagnostic profile; `full-public` is the only
  profile eligible for a complete cross-engine statement within its declared
  MT-Bench plus LongBench-primary scope. A limited profile is always screen-only,
  even when the source data is public.
- `validate-results` must pass before choosing a production candidate: every
  required engine covers every workload record exactly once, model/tokenizer
  fingerprints and generation settings match, public prompt hashes match,
  deterministic token hashes match, and TTFT/prefill/decode/end-to-end/RSS/swap
  metrics are present. Tooling fixtures may be synthetic only to test the gate;
  they are never benchmark evidence.
- Public result adapters must reconstruct each prompt from the locked source and
  record effective input-token and output-token hashes rather than copied prompt
  text. Pin prefill chunk size, truncation policy, warmup, cache-maintenance
  scope, process isolation, and memory-sampling cadence in a shared execution
  contract; a chunk-size difference can change greedy output tokens.
- A single complete `cross-engine-core` matrix is a scoped, order-alternated
  screen. Before selecting a production bottleneck, rerun the same public
  records with every shard's engine order reversed and report workload and
  input-length-bin results by order stratum with bootstrap intervals. A
  directional disagreement rejects the attribution.

## Workspace Contract

Run this at recovery and consolidation:

```bash
uv run python scripts/dev/check_loop_workspace.py --strict
```

- Keep one active iteration in `CURRENT.json`.
- Put exploratory and repeated raw output under the ignored
  `run/loop-engineering/<iteration>/` directory.
- Promote only benchmark source, manifest, aggregate/admission result, and the
  minimum raw records needed to recompute the formal claim.
- Do not stage duplicate screens, obsolete confirmations, Python caches, model
  data, or generated logs.
- A formal archive above 100 MiB or 150 files requires a written retention
  justification and a compact representation before consolidation.
- Existing user changes are inventory, not cleanup targets. Work around them
  and only prepare files owned by the active iteration.
- If inherited work already exceeds the absolute budgets, record its HEAD and
  counts once as an immutable debt baseline. Keep it as a warning, block growth
  beyond the explicit allowance, and enforce a separate active-iteration file
  and byte budget. Never move the baseline upward to absorb new work.
- Do not introduce a new mixed staged/unstaged path. Keep inherited mixed paths
  visible until their owner establishes a clean review boundary.
- Follow current repository instructions for commits. Without an explicit
  commit request, leave a verified, clearly reported change boundary.

## Completion Gate

An iteration closes only when all of the following are true:

1. The decision is `admit` or `reject`, not an ambiguous partial result.
2. Affected tests pass and the full-suite result is recorded; any inherited
   failure is resolved or reproduced on the frozen baseline.
3. Formal statistics, output parity, memory, swap, and workload scope are recorded.
4. Failed screens and contrary evidence remain named in the iteration record.
5. `STATUS.md`, `DECISIONS.md`, `KNOWN_ISSUES.md`, and `CURRENT.json` agree.
6. Scratch duplication and generated caches are removed after the retained
   evidence is verified.
7. The next iteration has one objective, one primary metric, and a bounded file set.
8. Any cross-engine conclusion names its public workload profile, source-lock
   hash, required-engine inventory, excluded-engine reasons, and completeness
   result. A scoped profile cannot be described as a global engine ranking.
9. The performance baseline ledger records a valid delta, or an explicit
   invalidation with a linked baseline and a bounded follow-up measurement.
10. Consolidation ends with one reviewable commit pushed to the configured
    remote. Record the pushed commit and remote result; a failed push leaves the
    iteration open rather than silently treating local state as delivered.

If a gate fails, keep the iteration active or reject the candidate. Do not hide
the failure by widening tolerances, changing inputs, or selecting favorable runs.
