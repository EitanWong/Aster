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

If a gate fails, keep the iteration active or reject the candidate. Do not hide
the failure by widening tolerances, changing inputs, or selecting favorable runs.
