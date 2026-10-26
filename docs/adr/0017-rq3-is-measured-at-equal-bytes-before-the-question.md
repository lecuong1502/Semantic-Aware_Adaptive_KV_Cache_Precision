# RQ3 is measured at equal bytes, before the question is asked

RQ3 asks whether allocating precision by attention-derived importance keeps
task quality better than allocating it uniformly, at equal average
compression. Milestone 2 (#88) built the mechanism and showed that it works.
The adaptive engine survives contention that ends a static one (ADR-0015,
ADR-0016), and three score sources plug into one planner and meet the same
byte target: semantic, uniform and random (#104).

This ADR fixes how RQ3 is measured, before any result is read. The protocol
is the part that is hard to reverse. Once results are measured against it,
changing it changes what they mean.

## Decision

**Milestone 3 is done when RQ3 is answered on needle retrieval.** The
answer compares semantic against uniform, and random, at equal reclaimed
bytes on Qwen2.5-1.5B, with confidence intervals and a paired test, and is
logged. The other tasks, the ablations (ADR-0018) and the confirmations
belong to the milestone but do not hold its closing.

**Compression is imposed by a controlled plan, not by contention.** At a
chosen point in a generation, the engine is given a byte target and a score
source, and the planner meets the target as it would under pressure. The
same plan logic runs (ADR-0016), but nothing waits on the monitor or the
simulator.
- Every source then reclaims the same bytes, to within one page's move.
- Runs repeat exactly.
- Each sample costs a few minutes rather than tens.

One end-to-end run under simulated contention confirms that the result
survives real pressure.

**The byte target is a share of the FP16 bytes outside the recency floor:
70% and 80%.**
- INT2 caps what a cache can give back at about 84%, at 2.63 of 16
  effective bits.
- Below about 50%, uniform only takes pages to INT8, which costs almost
  nothing (ADR-0011's perplexity: +0.007% on Qwen2.5-1.5B), and no
  difference can show.
- At 70% uniform takes nearly every page to INT4, and at 80% it must use
  INT2.
- 80% is the closing level.

**The primary experiment compresses before the question is asked.** It runs
as a conversation of turns:
1. **Turn 1:** the document and a general request, "Summarize the document
   above in two sentences", answered in 64 decoded tokens. The scorer folds
   in attention mass only while decoding, so these tokens are what give it
   scores.
2. **The controlled plan** is applied at the turn boundary.
3. **Turn 2:** the question, answered over the compressed cache.

This tests the thesis's premise directly: that the attention paid so far
predicts what will be consulted again, when contention arrives before the
question. A secondary experiment compresses after the first decoded tokens
of an answer to a question already asked. That gives an upper bound, for a
scorer that has seen the question.

**Configuration.**
- **Model and context:** Qwen2.5-1.5B at about 8K positions. RQ3's quality
  needs equal bytes and a known page holding the answer, not a 32K window,
  and at 8K the main set takes about 6 GPU-hours rather than 80.
- **32K subset:** ten needle samples, one per depth, run by the owner for
  FP16, semantic and uniform at 80%. They confirm the direction, with no
  test claimed for them.

**The needle set.**
- **Haystack:** articles of WikiText-2's test split, the text the
  perplexity runs already use, joined to about 8K tokens.
- **Needle:** a synthetic fact the model cannot guess, a named code drawn
  from a fixed seed, planted at ten depths, 0% to 90%. A needle's depth
  fixes the page that holds it.
- **Size:** five samples per depth, 50 in all.
- **Scoring:** an answer is correct if it contains the code.
- **Screening:** a sample that FP16 answers wrongly is replaced before the
  comparison, so that the model's failures are not counted as the cache's.

**Conditions.** Each sample is run once at FP16, then under each source at
each level. Static INT4 and INT2 for the whole session, near the two levels
in effective bits, stand for the uniform-quantisation family. Eviction at
the same bytes is a separate ticket that does not hold the closing.

**Secondary tasks.**
- **Continuation fidelity:** a WikiText passage is compressed part way
  through, and the rest is teacher-forced. The measures are top-1 agreement
  and KL divergence from FP16. No free generation is needed, so samples are
  cheap.
- **Anchor-fact dialogue:** a fact is stated in an early turn, other turns
  follow, and a later turn asks for the fact back. This is the case eviction
  fails and reallocation is built for.
- **Recovery:** after the compressed turn, pressure passes, upgrades restore
  the pages from their shadows, and a third turn asks again. A 32K pulse and
  release under the simulator measures the time to restore.

**Statistics.**
- **Needle:** McNemar's test on the paired outcomes of semantic and
  uniform, sample by sample, with bootstrap 95% confidence intervals on
  accuracy.
- **Continuation fidelity:** the paired difference in mean KL, with its
  bootstrap interval.
- **A null result is an answer.** If semantic does not beat uniform at 80%,
  RQ3's answer is no, and it is reported as such.

## Considered options

- **Impose compression by simulated contention.** Rejected for the
  comparison, kept as one confirmation.
  - The bytes a plan reclaims then depend on the headroom and on when an
    event lands, so the three sources do not reclaim the same bytes sample
    by sample.
  - Each run costs a survival experiment's time.
- **Compress after the question only.** Rejected as the primary experiment.
  - The scorer would already see the needle attended to, which tests a
    query-aware allocation, the one ZipCache and Cocktail make.
  - It does not test the premise that importance can be predicted before
    the question.
  - It is kept as the secondary experiment.
- **Run the main set at 32K.** Rejected: about 80 GPU-hours for 50 samples.
  The confirming subset covers 32K.
- **Use Qwen2.5-0.5B.** Rejected. Its own errors would blur those of the
  cache.
- **A third, lighter level, such as 50%.** Rejected: about 40% more runs, at
  a level where uniform costs nothing measurable.
- **Long-document QA, summarisation and model-as-judge dialogue.** Out of
  scope, as future work.
  - They need external datasets, and ROUGE or BERTScore, or a second model
    as judge, which a 6 GiB card cannot hold beside the engine.
  - The needle and anchor-fact tasks place the answer at a known page, which
    is what RQ3 needs.

## Consequences

- **The engine must continue a generation over its cache across turns.**
  Today each call starts a new cache.
- **The engine must apply a plan for a given byte target and source between
  turns.** Today a plan follows from the headroom a pressure event carries.
- **Results are reported at 8K positions,** with the 32K subset as a check,
  and the thesis states why.
- **Every run is a benchmark-log entry.** A summary entry pools them, as
  `monitor-summary` does.
- **The long runs are the owner's.** Tickets prepare the commands and hand
  them over.

## Amendment (#160): eviction at the same bytes

Eviction is run as a condition that does not hold the closing question, at
both levels, against the semantic source there.

- **Which pages.** The pages the semantic source's controlled plan for the
  level would downgrade most: those it leaves at the lowest tier first,
  and among those at one tier, the first it took there first. They are
  evicted instead of downgraded until the bytes they held meet the plan's
  target. An evicted page gives back all its bytes, so fewer pages go than
  the plan moves.
- **A page, not a token.** H2O drops tokens; the cache's unit is a page of
  32 positions, and a page is what is freed and returned to the driver.
  The pages chosen are the least attended by the scorer's EWMA, as H2O's
  heavy hitters keep the most attended.
- **How it is checked.** Attention over a cache with pages evicted is, to
  the bit, attention over the same cache with those pages masked, and its
  memory returns to the driver to a granule (`tests/test_eviction.py`).
  Masked is defined by the attention kernel: a masked page's scores are
  -inf by the same select a later position's are. Its output is checked
  against attention over the keys without those positions, near, not to
  the bit, since the tiles fall otherwise.
- **The shadow.** A page downgraded before it was evicted keeps its FP16
  shadow in pinned host memory, never read, until the cache goes: the
  shadow store only ever adds (#94). The device memory is what returns.
- **Where it is run.** The screened needle set and the anchor-fact
  dialogue, question-blind, with the semantic source at both levels run
  again beside it at the same code (`tools/rq3.py --evictions`,
  `tools/anchors.py --evictions`).

## As measured (#161): the closing answer is no

Every figure is from a summary entry in the benchmark log, named by its
hash. Qwen2.5-1.5B, the screened set of 50, question-blind unless said.

**The closing answer is no.** At 80%, the semantic source answered 48 of
50 needles and uniform 49 (`rq3-summary` `291c718f…`, on the screening
`7dafc4ea…`). One sample was answered by uniform alone and
none by semantic alone: McNemar's exact p = 1, where rejecting at 0.05
needs six discordant samples or more, all one way.

| Condition | Bits, 70% / 80% | Needles, 70% / 80% |
|---|---|---|
| FP16 | 16.00 | 50 / 50 |
| Uniform | 4.98 / 3.40 | 50 / 49 |
| Random | 4.98 / 3.40 | 50 / 50 |
| Semantic | 4.98 / 3.40 | 50 / 48 |
| Eviction (`71cbc603…`) | 4.98 / 3.40 | 22 / 16 |
| Static INT4 / INT2, whole session | 4.63 / 2.63 | 50 (INT4) / 45 (INT2) |

- **The needle set sits at its ceiling.** Static INT2 for the whole cache
  answered 45 of 50. The semantic source left 43% of the needles' (layer,
  page) pairs at INT2 at 80%, uniform 68% and random 73%: the order worked,
  but the code survived INT2 in most samples whichever source sent it
  there. No depth shows a gap.
- **Question-aware:** semantic 49 of 50 at 80%, uniform and random 50; p = 1.
  The upper bound shows no gap for the blind experiment to keep.
- **The 32K subset** (`rq3-long-summary` `c75bdd05…`): 10 of 10 under FP16,
  semantic and uniform, at 3.25 bits. No direction.
- **End to end** (`end-to-end-summary` `b856c0ab…`): under the simulator's
  contention, the same in all 50 samples, semantic 48 and uniform 49,
  p = 1. The 20 ms a step plans give had not finished the reclamation by
  the question: 157 and 161 MiB of a 172 MiB target, met in 0 and 19 runs.
- **Continuation fidelity** (`continuation-summary` `19ad5127…`, 50
  passages, 2048-token prefix, 128 tokens teacher-forced): at 80% mean KL
  0.028 semantic, 0.101 uniform, 0.082 random. Semantic's paired
  difference from uniform is −0.073 [−0.082, −0.065], lower on all 50
  passages, and its top-1 agreement 93.6% against 86.5%. At 70%, KL 0.0023
  against 0.0031, the difference within its interval, but top-1 still
  1.3 points higher [0.8, 1.8].
- **Anchor-fact dialogue** (`anchor-summary` `794253c8…`): at 80% semantic
  recalled 48 of 50, uniform 45 and random 48. Semantic against uniform
  is three to none, p = 0.25. Uniform held every fact's page at INT2,
  semantic 38% of them. Every source recalled all 50 at 70%.
- **Eviction** (`rq3-eviction-summary` `71cbc603…` and `e01276d7…`): 16 of
  50 needles and 16 facts at 80%, 22 and 21 at 70%. Every sample eviction
  answered, semantic answered too, so p < 1e-8 at each. At 80% eviction kept
  the facts of the last two turns, and few from three or more turns back.
- **Recovery** (`recovery-summary` `a206569e…`): after every page is
  restored, in under 0.7 s at 8K, a third turn answered 50 of 50 under
  semantic, both its turn-2 misses recovered; uniform's one miss stayed. At
  32K, a 30 s pulse during prefill was survived under both sources, every
  page restored 532 s (semantic) and 541 s (uniform) after the pressure
  ended, with prefill still under way (`pulse-release`, #155).

**What it means.** At equal bytes, allocation by attention did not keep
needle accuracy better than uniform allocation, on a task too easy at this
length to separate them. It kept the output distribution closer to FP16,
where the measure is graded. The thesis's premise was tested by these
outcomes only. The correlation between a page's score and the damage its
downgrade causes, by single-page downgrades, was not measured.

## Amendment (#188): a page's score against its own downgrade's damage

The premise behind RQ3 is tested page by page, on all three tasks
(`microinfer.page_damage`, `tools/page_damage.py`).

- **Where.** Each task's plan point, where its controlled plan would be
  applied: the needle at the end of turn 1, question-blind; continuation at
  the end of the warm tokens; the anchor-fact dialogue at the end of its
  last turn. The cache is held there as every source condition holds it,
  at FP16, adaptive and scoring.
- **Which pages.** For each sample, 64 (layer, page) pairs drawn without
  replacement, from a seed and the sample's id, among the pages a plan may
  move, outside the recency floor, that the scorer has observed. A page's
  score is the scorer's EWMA there. The samples are RQ3's: the screened
  needle set, the screened anchor-fact set and the first 50 passages.
- **The damage.** The page alone is moved to INT4, and the measured tokens
  are teacher-forced: for the needle and the dialogue, the question and the
  answer FP16 gives to it from the plan point, the KL over the positions
  that predict the answer's tokens; for continuation, its 128 measured
  tokens. The damage is the mean over those positions of KL(FP16 || INT4),
  in float64, against the same tokens forced with no page moved.
- **Each page from the plan point.** Every pair starts a conversation from
  the plan point's snapshot, so that no pair's move, nor the tokens forced
  after it, reaches another's. The forced tokens are what would have to be
  undone: the cache cannot be cut back, so the page is not moved back and
  the conversation is dropped instead. A page moved down and back from its
  shadow gives the next forward pass's logits of a cache never moved, to
  the bit (`tests/test_page_damage.py`).
- **Forced in prefill chunks**, not a position a step as continuation
  fidelity forces its measured tokens. The reference is forced the same
  way, so the damage is the move's alone.
- **The statistic.** Spearman's rho between score and damage, per task,
  pooled over every sample's pairs and per sample, their mean, each with a
  95% bootstrap interval over samples, since a sample's pages share its
  text and its answer.
