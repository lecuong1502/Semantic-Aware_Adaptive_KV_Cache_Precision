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
    spike;
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
  - Shortening steps under pressure is #135. Until it is decided, a sharp
    contention mid-prefill is answered by the emergency plan.
- **RQ3's comparison is fixed here:** semantic against uniform at the same
  byte target, with random as a control. A change to the planner after RQ3's
  results are read changes what they mean, and needs a new ADR.
