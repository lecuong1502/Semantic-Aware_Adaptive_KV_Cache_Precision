# The precision policy: a byte target met by marginal cost

ADR-0015 lets a page change tier at runtime. This ADR decides which pages
move, how far, and when. These are the decisions of #88, built in #98 to
#107.
- **The importance scorer** says which pages the model attends to.
- **The precision controller** turns pressure and scores into a plan.
- **The engine** applies the plan between decoding steps.

The policy is what RQ3 measures against a uniform allocation. It therefore
has to be stated before any result is read, together with the baselines it
is compared with.

## Decision

### The importance scorer

- **A page's attention mass comes from the attention kernel,** written
  during decode when asked (#98). Each tile is one page. It writes its
  partial sum of `exp(s − m)` with its `m`, and these are normalised into
  each page's share once the softmax's `m` and `l` are final. A layer's
  pages, for each query head, sum to 1. With the output off, the kernel's
  outputs are bit-identical to before.
- **A page's importance score is an EWMA of its layer's mass,** averaged
  over query heads, with α = 0.2. It is folded in on the device (#99). A
  page's first observation seeds its score, rather than 0. Scores stay on
  the device until a plan asks for them.
- **All layers are weighted equally.** Layer weights taken from measured
  per-layer sensitivity are left to RQ3's ablations.
- **The scorer folds in every 4th decoding step's mass,** not every step's
  (#100). This is the fallback #88 allowed if scoring every step cost more
  than noise.
- **Attention sinks get no hard protection.** On both Qwen2.5 models the
  scorer rates the first page highest when averaged over layers, and among
  the three highest in at least three layers in four (#100's test).

### The precision controller

The controller is pure logic with no device code (#101).

- **A byte target.** A plan reclaims what brings headroom back above
  YELLOW's threshold (ADR-0013) plus a 64 MiB margin: `T_high + margin −
  headroom`. At GREEN it reclaims nothing.
- **Greedy by marginal cost.** The controller repeatedly takes the cheapest
  next move, one page one tier down, until the target is met. A move's
  marginal cost is the page's score × the error the lower tier adds ÷ the
  bytes it saves. The plan overshoots the target by less than one move.
  - Each tier's error is its relative mean squared error from Milestone 0's
    logged roundtrips.
  - INT2 is allowed. Its error puts it last unless a page's score is low
    enough to outweigh it.
  - A page with no score yet is given the mean score of the scored pages.
    Before any page has a score, pages go in position order, oldest first.
- **The recency floor** (#102). The open pages and every page reaching into
  the last W = 128 positions are never downgraded, under every policy.
  Every plan records its W.
- **Upgrades only at GREEN** (#103, #106):
  - at least 5 s after the later of the last downgrade and the end of the
    last pressure;
  - only while headroom stays at or above `T_high` + 512 MiB, one P90
    spike (128 MiB since #237; see its amendment);
  - in the reverse of a downgrade plan's order, largest gain first;
  - in batches within the same per-step time budget as YELLOW's downgrades.
- **Three score sources plug into the same planner** (#104). All three meet
  the same byte target, to within one page's move. Only the scores differ,
  so a difference between them is a difference of allocation, not of
  compression:
  - **semantic:** the scorer's scores;
  - **uniform:** every page scored alike, RQ3's main baseline. A partial
    step is spread evenly across positions, in bit-reversed order.
  - **random:** seeded, a secondary control.

### The engine

- **A YELLOW or RED event triggers a plan,** made from the scores as they
  stand and applied synchronously between steps (#105). Each plan is
  recorded with the event that caused it.
  - **At YELLOW**, a step applies moves only while the step stays within a
    20 ms budget, planning included. The engine times every move, so the
    budget holds by measurement rather than by the plan's estimate.
  - **At RED**, the whole plan is applied at once.
  - **While the level stays YELLOW or RED**, the engine plans again each
    time the cache seals new pages: a *persisting* plan.
- **On an OutOfMemory, the engine makes an emergency plan** (#107). It is a
  RED plan for no headroom, applied at once, and the step is retried once.
  If the plan has nothing to downgrade, frees nothing, or the retry fails,
  the generation ends *exhausted*. It keeps its tokens, records the
  position reached, releases its cache, and raises nothing.

## Considered options

- **Fixed fractions per level.** The research notes' first draft:
  downgrade the bottom X% of pages one tier at YELLOW, and Y% two tiers at
  RED. Rejected: a fraction says nothing about how many bytes the machine
  needs back. The plan then either falls short of what the contention took
  or degrades more than it had to. The byte target ties the plan to
  headroom.
- **Breadth-first or depth-first by score alone.** Rejected as the policy,
  kept as ablations. Both are special cases of marginal cost:
  - equal scores give breadth-first, since each tier down costs more error
    per byte than the last;
  - a near-zero score goes deep first.
- **A separate scoring pass over the cache.** Rejected: it doubles the reads
  of the cache. The kernel already holds the softmax's partial sums.
- **Key-norm proxies for importance.** Rejected: they are not attention.
- **Scoring every step.** Measured and rejected, because it was outside
  noise on Qwen2.5-0.5B at 512 positions:
  - every step: −0.71% decode throughput against a noise of 0.68%
    (`cadce177…`, at fa73af3);
  - every 4th step: −0.65% against a noise of 2.5%, within noise
    (`f9166c24…`, at 6a09671);
  - on Qwen2.5-1.5B, every 4th step: −0.02% against a noise of 4.6%
    (`a2f8cc80…`, at 6a09671).
- **Hard protection of the first page.** Not needed. The scorer rates it
  highly on its own, and a hard rule would protect a sink page in a model
  that had none.
- **Hysteresis per policy, planning again at most every M steps.** Dropped.
  The monitor's K = 3, the budgeted batches and the upgrade cooldown already
  cover it.
- **An upgrade cooldown from the last downgrade only.** Amended in #106.
  Under a train of pulses whose later pulses find nothing left to
  downgrade, upgrades between pulses would be undone by the next one. The
  cooldown therefore also runs from the end of the last pressure.
- **A uniform partial step spread every k-th page.** #88's wording.
  Replaced by bit-reversed order over positions, which spreads every prefix
  of a partial step across the cache even where the page count is not a
  multiple of k. With powers of two it is exactly every k-th page, and
  otherwise no window of positions is off its even share by more than
  log2(positions) / 2 + 1 pages (#104).
- **Re-scoring on a topic shift.** Deferred to Milestone 3's multi-turn
  dialogue, if the EWMA proves too slow there.
- **On OutOfMemory, offload pages to the host or evict tokens.** Rejected:
  offloading is FlexGen's family, and eviction is H2O's, which is
  irreversible. **Raise and lose the session.** Rejected: this is the
  failure the project exists to prevent.

## Consequences

- **The survival experiment** (#108, `1fdbeebd…`). The adaptive engine
  survived a shortfall the static engine could not, with 33 plans:
  - 11 made on events that moved pages;
  - 21 persisting;
  - 1 with nothing to move.

  No emergency plan was needed. Planning took
  6 ms and applying a plan 62 ms at the median. The first RED plan applied
  44,016 downgrades in 1.7 s. No plan met its target, because after the
  first, every page outside the recency floor that could go lower had done
  so. The controller then reclaims what it can and carries on, as designed.
- **During prefill, the engine acts one chunk late.**
  - The monitor reported RED 165 ms after the contention began, but the
    engine drains events only between steps, and a 512-position prefill
    chunk at 16K to 24K positions takes 11 to 16 s.
  - Each event during prefill therefore waited 13.2 s at the median before
    its plan.
  - The run survived because the first RED left more headroom than one
    chunk's pages need.
  - #135 shortens steps under pressure; see the amendment below.
- **RQ3's comparison is fixed here:** semantic against uniform at the same
  byte target, with random as a control. A change to the planner after RQ3's
  results are read changes what they mean, and needs a new ADR.

## Amendment (#135): prefill chunks bounded by time under pressure

While an adaptive engine's monitor reads YELLOW or RED, each prefill chunk
is sized from what the steps before it cost (`StepCost`). A step's time is
taken as a fixed part and a part per position.
- The per-position part comes from comparing the step with the one just
  before it, if their sizes are at least a factor of two apart. A step from
  further back ran at a shorter context, where a position cost less, and
  would make the per-position part too small: a first version that compared
  with such steps sized its chunks up to 512 under RED.
- Until a split is made, a chunk is at most half the one before, so that two
  steps in a row are wide apart.
- A comparison that makes no sense, as noise can make a larger step the
  quicker, is ignored, and the estimate stands.
- The fixed part follows every step, since it grows with the context.

The chunk is then chosen by how the fixed part compares with
`pressure_chunk_seconds`.
- **If the fixed part is within it:** the chunk holds as many positions as
  are left after the fixed part. It is one tile at least, which may take a
  little longer.
- **If the fixed part alone overruns it,** as at long contexts it does, no
  chunk can meet the time. The chunk is then the positions worth the fixed
  part again, so that a step takes about twice its fixed part.
- **In every case:** whole query tiles, the attention kernel's (ADR-0011),
  at least one tile, and no more than `prefill_chunk`.

`pressure_chunk_seconds` is 0.8 s, below #135's target of 1 s from an event
to its plan by what a plan between steps and a step's growth can add. At
GREEN, and in an engine that does not adapt, chunks stay at
`prefill_chunk`.

**The first design, and what it measured.** The first design sized a chunk
from the time a position of the chunk before took, as though a step had no
fixed part, within 0.5 s. The survival experiment was run again with it
(`survival` entry `27567876…`, at 8e3ccf0), against the run without it
(`1fdbeebd…`):

| | Chunks of 512 | First design |
|---|---:|---:|
| Event to plan, after an episode's first event, median | 13.2 s | 1.5 s |
| Events within 1 s | 0 | 24 of 74 |
| From the take to the end | 668 s | 1617 s |

- **From 26K positions every step was one tile,** 16 positions, and still
  took about 1.7 s.
- **A step has a large fixed part**, which on this engine grows with the
  context, and a per-position estimate charged all of it to 16 positions.
  One tile was therefore always chosen, though a larger chunk would have
  cost a step little more.
- **Prefill under pressure ran at a third of its speed,** with no gain in
  latency.

The present design separates the two parts. It was probed, unlogged, on
Qwen2.5-1.5B with a prompt of 8192 positions under a simulated RED, which
downgrades the cache as it grows. A step there took 0.43 s before its first
position and 4.3 ms per position. These are the costs of a quantised cache,
which differ from those of an FP16 one. Prefill ran at 130 tokens per second
with 0.8 s, and at 106 with 0.5 s, against 190 with no pressure. With 0.8 s,
steps took 0.75 s at the median and 1.26 s at most.

**Measured at 32K** (`survival` entry `04b95e3f…`, at acd2217). The survival
experiment, run again with the present design:

| | Chunks of 512 | First design | Present design |
|---|---:|---:|---:|
| Episode's first event, to its plan | 11.1 s | 11.1 s | 11.1 s |
| Later events, to their plans, median | 13.2 s | 1.5 s | 0.86 s |
| Later events within 1 s | 0 | 24 of 74 | 3 of 5 |
| From the take to the end | 668 s | 1617 s | 1588 s |

- **The session survived** with 489 plans, every one seen by the driver to a
  granule.
- **The latency met the target at the median.** The sample is small: the
  level held more steadily this time, so most plans were persisting ones,
  made on no event. The slowest event, 1.87 s, came during decoding, where a
  step at 32K positions takes about 1.5 s.
- **The run took no less time than the first design's.** At these contexts a
  step's fixed part dominates. Short steps therefore cost about 2.4 times
  the prefill time of chunks of 512 under pressure, whatever the estimate.
  That is the price of acting within a second or two rather than within
  13 s.

**Considered options.**

- **Bound every chunk by time, at GREEN too.** This is the only option that
  also bounds the first event, GREEN to YELLOW or RED. Rejected for its cost.
  Measured on Qwen2.5-1.5B with a prompt of 8192 positions, at 6K to 8K
  positions, one unlogged run each:

  | Chunk | Prefill throughput | Per position |
  |---|---:|---:|
  | 512 | 190 tok/s | 9.1 ms |
  | 128 | 151 tok/s | 11.4 ms |
  | 64 | 129 tok/s | 13.3 ms |
  | 32 | 115 tok/s | 14.7 ms |

  A 1 s bound at GREEN would cost 30 to 60% of prefill throughput with no
  contention at all.
- **A fixed small chunk under pressure.** Rejected. The time a chunk takes
  grows with the context, so a fixed size bounds nothing at long contexts.
- **Size by time per position alone.** The first design, rejected on its
  measurement above.

**The accepted limits.**

- **The first event of every episode** of pressure, each transition from
  GREEN to YELLOW or RED, still waits for the chunk in progress, at
  `prefill_chunk`. In a train of pulses, that is every pulse. The engine is
  not at risk while it waits, since a step reserves all of its pages before
  it runs (`KVPages::reserve`). The process that took the memory is the one
  kept waiting.
- **Where a step's fixed part exceeds the target, the target cannot be
  met.** On Qwen2.5-1.5B this is the case at long contexts: in the first
  design's run, every step from 26K positions took over 1.5 s. Draining
  events only at step boundaries cannot do better than one step, and the
  chunk is sized to keep a step about twice its fixed part.
- **Prefill under pressure at long contexts takes about 2.4 times as long**
  as with chunks of 512, as measured above.
- **A step can overrun its time** where the cost changes from one step to
  the next, as it does when a plan has just moved pages: the longest step
  in the 8K probe took 1.26 s against 0.8 s.

Every pressure event's wait, and its time to its plan, is logged by the
survival experiment.

## Amendment (#210): ZipCache's normalised attention, a fourth score source

Semantic is compared with uniform and random, baselines of our own. ZipCache
(He et al., 2024, arXiv 2405.14256) is prior work whose rule ranks cached
tokens by attention, so it is added as a source. It plugs into the planner
above, at the same byte target: only the scores differ.

**Its rule.** A token's normalised attention is its equation 8,
`sum_k A[k, i] / nnz(A[:, i])`: the attention the token receives, divided by
the number of queries that could attend to it. Accumulated attention, H2O's
and MiKV's, is the sum undivided. ZipCache takes it from probe tokens, a
tenth of the positions: the most recent 5% and a random 5% (its section 4.3).

**Why not from the scorer's masses.** #210 asked for it from the same
per-page masses the scorer reads, which are a decode step's. Measured before
it was built, unlogged, on three needle samples of 8K positions on
Qwen2.5-1.5B, over the pages a plan may move: from the summary's decode steps
alone, every page has been attended by the same queries, so normalised and
accumulated rank them the same (rho = 1.000), and both almost as the scorer's
EWMA does (rho = 0.96 to 0.98). That source would have been the semantic one
without its decay, under ZipCache's name. From probes over the prompt, the
ranking is another (rho = -0.01 to 0.22 with the scorer's). The owner chose
the probes (2026-11-24).

**As built** (`microinfer.zipcache`, `Engine.probed_scores`).
- **A pass of its own.** When a controlled plan reads the source, the tokens
  its conversation holds are fed again into a second cache, at FP16. Each
  probe is a step of one position, for which the attention kernel already
  gives per-page masses; the positions between are prefilled in chunks. The
  cache the plan is made on is untouched, so every source's plan is made on
  the same cache, to the bit. In ZipCache the probes are in its one prefill:
  the pass's time, 43 s at 8K positions, is the experiment's, not the method's.
- **What ZipCache does not say, or does otherwise.** A layer's mass is its
  heads' mean, as the scorer's is, and each layer's pages are scored by its
  own. A page's score is its mass over the probes divided by the (query,
  token) pairs that could meet there: for a page every probe is past, the
  mean of its tokens' equation 8, exactly. ZipCache gives two bit-widths to
  tokens by a saliency ratio; here the score is the planner's.
- **The probes** are drawn by the seed the plan is given, with the sample's
  id (`rq3.sample_seed`). The last position is always one, so every page has
  a score.
- **Not under pressure.** An engine's own plans read scores as they stand
  (`RUNTIME_SOURCES`); a pass over the cache would cost what it is planning
  for. The survival matrix has no such condition, and the emergency plan of
  a run under this source reads the scorer's scores.

---

## Note from #214: on the present engine, chunks stay at 512

Since split-key decode (#204) and prefill on the tensor cores (#213), a chunk of
512 positions of Qwen2.5-1.5B takes at most 0.44 s at 32K, at any tier
(`prefill-chunk-cost`, #214). That is under `pressure_chunk_seconds`, 0.8 s,
so the step-cost model leaves chunks at 512 under pressure, and the first event
of an episode waits for one such chunk at most. Measured in the survival
experiment, S5 (`survival` `1e099d65…`): every YELLOW and RED event waited at
most 218 ms for its step, the episode's first 195 ms, and was planned 184 ms
after it at the median. With chunks kept at 512 under pressure
(`--pressure-chunk-seconds 1000`, `9544e952…`) the waits and the time from the
take to the end, 16.9 s, were the same. The 2.4 times that sizing cost on the
engine before is gone, and so is the first event's 11 s.

The design stands for a slower engine or a longer context. Two of its
constants were set for steps of a second or more, and now meet steps of tens of
milliseconds:

- **The plan budget, 20 ms a step,** is about half a decoding step at 32K
  (26 tokens a second), where it was about 1% of one. A paced plan nearly
  doubles a step while it lasts.
- **Upgrades keep 512 MiB above YELLOW.** Prefill now ends before a 30 s pulse is
  released, so the whole cache is held when the upgrades come, and the margin
  left no room to restore it: no S6 run of #214 restored its cache
  (`pulse-release` `a2f64a62…`, `b384d307…`, `054c06ed…`). #237 is to find out
  whether that, and the semantic source's slow pacing there, should change.

---

## Amendment (#237): upgrades keep 128 MiB above YELLOW, and are paced by large batches

#214's S6 runs restored no cache to FP16 (ADR-0016, note from #214). Two
causes, each measured on the engine, Qwen2.5-1.5B at 30,720 positions, a
controlled plan at 80% then a controlled restore, which keeps no margin:

**1. The pace.** The pace is twice the largest time a move of the last 64
batches, and a batch of fewer than 32 moves gave its whole time over its
moves as that time. A step's last batch, cut short by what is left of the
budget, is mostly its fixed part: a batch of one move takes 0.14 to 0.43 ms,
of 512 moves 2.6 to 4.3 ms (unlogged). Read as a move's time it set the pace
at 0.3 to 0.9 ms a move, against about 10 µs from large batches, for the next
64 batches, which were then as small and read the same way. The semantic and
random sources, whose orders leave such batches, fell to about 36 moves a step;
uniform's, which rarely did, did not. A batch too small to measure past its
fixed part now gives no time, and the pace stays the large batches'. The
controlled restore of every page (unlogged):

| Source | Before: steps, seconds | After: steps, seconds |
|---|---:|---:|
| Semantic | 1,816, 37.1 s | 231, 4.6 s |
| Uniform | 334, 6.8 s | 68, 1.5 s |
| Random | 1,778, 36.0 s | 200, 4.0 s |

Semantic's moves still cost the allocator about five times uniform's, 38
against 7.6 µs a move of `move_pages` (unlogged). Part of it is that its pages
are scattered through each tier, and a free fills each hole by moving a page
down from the tier's end: timed alone, a batch of 512 moves freeing from the
middle cost 1.6 times one freeing from the end. The rest is not accounted for.
Each batch also reads `page_tiers()` and `shadowed()` afresh, about 3.5 ms each
at 30K positions whatever the order: a third of the budget, left for later.

**2. The margin, the owner's decision of 2026-11-26.** Upgrades kept headroom
at or above `T_high` + 512 MiB, 1536 MiB, one P90 spike above YELLOW, which
itself keeps one P90 spike above RED (ADR-0013). On the RTX 4050, Qwen2.5-1.5B
with an FP16 cache of 32K positions leaves about 1310 MiB with nothing
contending. Once a cache at 32K was downgraded, it could not come back to FP16.
#155's runs came back only because their cache grew at FP16 after the release,
and growth keeps no margin: the semantic one's last upgrade left 1511 MiB
(`survival-timeline` `6ae3da2f…`, of `94948638…`).

The margin is now **128 MiB, two spike thresholds**: enough that the desktop's
own jitter does not swing the level between YELLOW and GREEN as upgrades bring
headroom down to YELLOW. One P90 spike's room above RED is kept by YELLOW
itself. `controller.DEFAULT_UPGRADE_MARGIN_BYTES`.

- **What it changes.** A cache can be restored to FP16 wherever its FP16 size
  leaves headroom at or above 1152 MiB. Restored that close to YELLOW, the
  next spike finds the engine at YELLOW, not at GREEN with a spike to spare,
  and downgrades again gradually rather than not at all.
- **What it does not.** RQ3's controlled restore kept no margin and keeps none.
  Results before #237 were measured with 512 MiB; #216 runs the survival matrix
  again with 128.

**As measured (#237).** S5 and S6 again at 1442f35, Qwen2.5-1.5B. S6, a 30 s
pulse during prefill, 30,720 positions and 2,048 decoded: every source restored
its whole cache to FP16 (`pulse-release` `f0294324…`, `76b5cf30…`,
`731c6406…`):

| Source | Moves up | Steps | Moves a step | Restored, after the release | Bits after |
|---|---:|---:|---:|---:|---:|
| Semantic | 81,144 | 506 | 160 | 51 s | 12.09 |
| Uniform | 81,060 | 133 | 609 | 17 s | 14.54 |
| Random | 81,060 | 557 | 146 | 56 s | 12.04 |

The first upgrade came 5 s after the release, the cooldown. Each step's batches
took about 22 ms. Once restored, 1281 to 1297 MiB stayed free at the lowest,
above the 1152 MiB kept, and the level stayed GREEN: no run downgraded again or
had an upgrade plan cancelled. S5 (`survival` `e5c675ad…`, `2588d7d5…`,
`15e965a7…`): the static engine ran out of memory at 21,504 positions, 2.7 s
after the take; the semantic engine completed, 33 plans, 748 MiB returned and
seen by the driver plan by plan; every event waited at most 192 ms for its step.
