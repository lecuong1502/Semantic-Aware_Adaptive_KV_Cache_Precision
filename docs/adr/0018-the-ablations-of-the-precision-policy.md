# The ablations of the precision policy

ADR-0016 fixed the policy, and ADR-0017 fixes how RQ3 is measured. This ADR
fixes which parts of the policy are removed or varied to show what each
contributes, and how each is measured. It is decided before any ablation is
run, so that the list is not chosen after the results.

## Decision

**Ten variants.** Each changes one element of the full policy and keeps the
rest at ADR-0016's values. Each is measured on ADR-0017's needle set, in the
question-blind experiment at 80%, against the full semantic policy on the
same samples. The exception is the monitor's K, below.

| Element | Variants | Default |
|---|---|---|
| Recency floor | W = 0; W = 512 | W = 128 |
| Scoring cadence | every decoding step | every 4th step |
| EWMA weight | α = 0.05; α = 0.5 | α = 0.2 |
| Marginal cost | importance alone, the tier's error left out | score × error ÷ bytes |
| The INT2 tier | lowest tier INT4 | INT2 allowed |
| Layer weights | scores weighted by measured layer sensitivity | all layers equal |
| Attention sinks | the first page never downgraded | no special case |
| Monitor hysteresis | K = 1; K = 5 | K = 3 |

**What each needs.**
- **Configuration only:** W, the scoring cadence and α. α is to be exposed
  on the engine, where today only the cache takes it.
- **Small planner options:** importance alone, a lowest tier, and a
  protected first page. Each is a parameter recorded in every plan.
- **Layer weights:** a sensitivity measurement first.
  - Each layer alone is taken to INT2 on WikiText-2, and the KL divergence
    from FP16 of the next-token distribution is measured.
  - The weights are those divergences, normalised.
  - A page's score is multiplied by its layer's weight before planning.
- **K:** measured on the monitor, not on quality. In the controlled
  experiments no monitor runs, so K cannot change the answers. K = 1 and
  K = 5 are scored on #64's synthetic grid, by detection latency, false
  negatives and false positives, as ADR-0014 scored K = 3. This makes it an
  ablation of RQ2.

**Reporting.** Each quality ablation gives:
- the change in needle accuracy from the full policy, with its bootstrap
  interval;
- McNemar's test against the full policy;
- the average effective bits it held, so that equal bytes can be checked
  rather than assumed.

## Considered options

- **Drop the layer weights, sink protection and K**, as a shorter list.
  Rejected by the owner. Each answers a question a reader will ask:
  - whether some layers matter more;
  - whether the scorer protects sinks well enough unaided;
  - whether K = 3 was the right hysteresis.
- **Measure K on quality, under simulated contention.** Rejected. K moves
  when a plan starts, by a poll or two, not which pages it degrades. Its
  effect on quality would be lost in the noise of contention runs.
- **Take layer weights from attention entropy or from the literature.**
  Rejected. A weight is only as good as the measurement behind it, and this
  engine can measure its own model's sensitivity directly.
- **Run every ablation at both levels and in both experiments.** Rejected
  for its cost, four times the runs. The closing level and the primary
  experiment are where an element's contribution matters.

## Consequences

- **Four small planner options** are added under ADR-0016's planner and
  recorded in every plan: a lowest tier, importance alone, a protected first
  page, and layer weights.
- **The ablations add about 10 GPU-hours** at 8K positions, run by the owner
  in batches.
- **A variant that beats the full policy is reported as such.** The policy
  is not changed after the results are read without a new ADR (ADR-0016's
  consequence).
