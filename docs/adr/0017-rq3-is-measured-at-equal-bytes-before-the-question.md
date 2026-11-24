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
downgrade causes, by single-page downgrades, was not measured then; the
amendment below measures it.

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

### As measured (#189)

On Qwen2.5-1.5B at commit `bc429ad`, 64 pages in each of 50 samples a task,
3,200 pages a task; intervals are 95% bootstrap over samples.

| Task | Pooled rho | Mean rho per sample | Samples rho > 0 | Median KL | Summary |
|---|---|---|---|---|---|
| Continuation | +0.350 [0.303, 0.394] | +0.355 [0.306, 0.400] | 49/50 | 1.1e-6 | `8fd16470…` |
| Needle, question-blind | +0.006 [-0.039, 0.051] | +0.006 [-0.036, 0.049] | 26/50 | 6.0e-8 | `6a417b1a…` |
| Anchor-fact dialogue | +0.082 [0.047, 0.116] | +0.075 [0.044, 0.107] | 37/50 | 5.0e-8 | `03dcd478…` |

**What it means.** A page's score ranks pages by the damage their downgrade
does to the continuation of the text the scorer watched, the task on which
the semantic order kept the distribution 3.6 times closer to FP16. It does
not rank them, or barely, by the damage to the answer of a question asked
after the plan point, which agrees with the null closing comparison: the
premise holds for what follows the text, not for what a later question
will need. One page alone does little harm, the largest damage
7.7e-4; the test shows the ranking a plan relies on, not how many pages'
damages combine.

---

## Amendment (#206): static INT8 joins the static tiers

Static INT4 answered all 50 needles, more than the semantic source's 48. A
reviewer's first question is then why not a static cache at all, and INT8
is the one to ask it of: near lossless, at 8.63 effective bits, about half
an FP16 cache's bytes. RQ3's static tiers are therefore three:
`static-int8`, `static-int4` and `static-int2`.

- **The needle set:** `static-int8` is a condition of every question-blind
  run, as the other static tiers are. A summary of runs logged before
  #206, which have none, pools the conditions they have
  (`rq3.LATER_STATIC_CONDITIONS`). Once static INT8 has a run, every sample
  needs one.
- **Continuation fidelity:** a static tier holds the passage at its tier
  from its first position, with no plan. It runs alone and is measured
  against the same FP16 reference as a plan.
- **The anchor-fact dialogue:** a static tier runs its dialogue alone, as
  a condition that holds the turns otherwise does.

The comparison the closing question makes is unchanged. The static tiers
answer a different question: what a session pays when its cache is held
low throughout, where the adaptive engine pays only under pressure.

---

## Amendment (#208): RULER's tasks, a quality task below its ceiling

The needle set sits at its ceiling: static INT2 answered 45 of 50 at 8K,
so at most five samples could be discordant at 80%, fewer than McNemar's
test needs (Chapter 4). The owner chose RULER's subsets (2026-11-14), and
RULER's own essay haystack for the tasks that have one (2026-11-15). This
amendment fixes how they are generated, adapted to RQ3's turns, screened
and tested, before any of them is run.

**The tasks.** Five of RULER's (Hsieh et al., 2024), each by its template,
answer prefix, generation length and arguments in RULER's
`scripts/synthetic.yaml` and `scripts/data/synthetic/constants.py`:

| Task | Haystack | Keys | Values | Needles | Answer |
|---|---|---|---|---|---|
| `niah_multikey_1` | essays | words | 7-digit numbers | 4 keys, one asked | 1 value |
| `niah_multikey_2` | needles | words | 7-digit numbers | 1, among needle-shaped fillers | 1 value |
| `niah_multikey_3` | needles | UUIDs | UUIDs | 1, among needle-shaped fillers | 1 value |
| `niah_multivalue` | essays | words | 7-digit numbers | 1 key, 4 values | 4 values |
| `vt` | noise | 5 capitals | 5-digit number | 1 chain of 4 hops | 5 names |

A needle reads "One of the special magic numbers for {key} is: {value}."
Words are wonderwords' adjective-noun pairs. An essay needle goes in at a
depth drawn without replacement from RULER's 40 evenly spaced depths, at
the sentence boundary there. A filler or noise needle goes at a random
sentence index. VT's statements read "VAR {name} = {value}" and
"VAR {name} = VAR {name} ", one a line, in order, at random positions among
the noise sentences. As in RULER, VT's turn opens with a worked example: a
chain of three-letter names among noise, fitted to 500 tokens, with its
question, answer prefix and answer. Its layout is drawn once for the task,
and its names and value again for each sample, as RULER's `randomize_icl`
does. Its tokens count against the haystack's.

**Where it departs from RULER**, each recorded with the data's hashes, and
each for a reason:
- **The essays** are fetched as RULER's script fetches them, from the URL
  list at a pinned RULER commit, the GitHub copies first, then
  paulgraham.com's. Their text is extracted with Python's standard library,
  not html2text: the engine's environment is pinned and minimal
  (requirements.txt), and a data script should not widen it. A page's
  scripts and styles are dropped, as html2text drops them. The haystack is
  RULER's essays, but not RULER's bytes. For the same reason, sentences
  are split by a fixed rule, a stop before a capital, a digit, a quote or
  a bracket, not by nltk's punkt, which needs a model of its own.
- **Words** are wonderwords' lists at its last release, v2.2.0. RULER does
  not pin a version.
- **Keys** that a sample already holds are drawn again, so that no filler
  shares the asked key. RULER does not check. A filler with the asked key
  would make a sample with two answers.
- **Seeds.** Each task and length draws from a numpy generator of its own,
  seeded by the set's seed, the task and the length. Drawing more samples
  leaves the first ones as they are, as the needle set's do (#146), which
  screening needs. RULER's Python `random` stream does not.
- **Length.** A sample's haystack is cut to a number of tokens, 7936 at 8K
  and 32256 at 32K, the needle set's, rather than RULER's binary search to
  a sequence length. This keeps the two sets alike, and every turn of a 32K
  run within Qwen2.5's 32768 positions.
- **The answer prefixes** lose the spaces at their ends: they open the
  assistant's reply in the chat template, where a leading space is not
  how a reply begins, and the reply goes on after the prefix's last word.

**RQ3's two turns.** RULER asks its question in the same message as its
context. Here, as for the needle set:
1. **Turn 1** is the template's opening, the context, and the general
   request, "Summarize the document above in two sentences", answered in
   up to 64 tokens.
2. **The controlled plan** is applied at its end, question-blind.
3. **Turn 2** is the template's question. The answer prefix opens the
   assistant's reply, fed rather than decoded. The reply is decoded for
   RULER's generation length: 128 tokens for the needle tasks, 30 for VT.

Only the question-blind experiment is run.

**Scoring.** RULER's `string_match_all`: a sample's score is the share of
its references found in the answer, case-insensitive. A sample is
**answered** if its score is 1, every reference found. McNemar's test pairs
answered samples; the score's mean is reported beside it.

**Screening.** For each task and length, samples are drawn in order and run
at FP16. The first 50 answered are kept, and those before them that FP16
did not answer are replaced, as ADR-0017's screening does.

**The ceiling.** At each length, static INT2 is run on each task's
screened set. A task is **below its ceiling** if static INT2 answers at
most 40 of its 50, 80%. Only tasks below their ceiling enter the closing
test. A task above it is reported and not tested.

**Conditions**, all question-blind on the screened sets:
- B1, FP16 (its screening runs);
- B8, B2 and B3: static INT8, INT4 and INT2;
- B4, B5 and B6: the uniform, random and semantic sources at 70% and 80%;
- B7: eviction at 70% and 80%.

**The tests, fixed now.**
- **At 8K, the closing test:** semantic against uniform at 80%, pooled over
  the tasks below their ceiling at 8K. McNemar's exact test, two-sided at
  0.05, on the samples only one of the two answered. It is a yes only if
  semantic also answered more.
- **At 32K:** the same test over the tasks below their ceiling at 32K,
  reported as a second test at 0.05, not as a correction of the first.
- **Descriptive:** each task, each level, random, eviction and the static
  tiers, with Wilson 95% intervals on accuracy and the paired mean score
  difference's bootstrap 95% interval.

Both answers are reported, yes or no.

## As measured (#208): RULER's closing answer is yes at both lengths

Qwen2.5-1.5B, question-blind, 50 screened samples a task. Every figure is
from the summaries `ruler-summary` `2b7db8a7…` (8K) and `7c6ac229…` (32K),
which also give where each plan left the needle (#208's last code). FP16
answered 50 of the first 50, 53, 63, 96 and 76 drawn at 8K (MK-1, MK-2,
MK-3, MV, VT) and of 54, 57, 135, 209 and 68 at 32K.

**Every task is below its ceiling** at both lengths. Static INT2 answered
21, 5, 0, 2 and 1 of 50 at 8K, and 10, 0, 0, 0 and 0 at 32K.

| Task | INT2 | INT4 | Sem. 70 / 80 | Uni. 70 / 80 | Rand. 70 / 80 |
|---|---|---|---|---|---|
| 8K MK-1 | 21 | 50 | 49 / 38 | 50 / 37 | 50 / 38 |
| 8K MK-2 | 5 | 50 | 50 / 15 | 50 / 19 | 49 / 20 |
| 8K MK-3 | 0 | 43 | 17 / 1 | 46 / 1 | 44 / 4 |
| 8K MV | 2 | 29 | 41 / 20 | 31 / 13 | 34 / 9 |
| 8K VT | 1 | 35 | 49 / 49 | 50 / 30 | 45 / 29 |
| 32K MK-1 | 10 | 50 | 49 / 35 | 50 / 29 | 50 / 30 |
| 32K MK-2 | 0 | 41 | 35 / 6 | 46 / 3 | 43 / 4 |
| 32K MK-3 | 0 | 30 | 0 / 0 | 38 / 0 | 21 / 0 |
| 32K MV | 0 | 42 | 30 / 13 | 35 / 7 | 33 / 9 |
| 32K VT | 0 | 43 | 48 / 46 | 48 / 13 | 47 / 15 |

Bits at 70% / 80%: 4.98 / 3.40 at 8K, 4.84 / 3.25 at 32K, the same for
every source.

- **At 8K the closing answer is yes:** semantic 123 of 250, uniform 100;
  40 samples answered by semantic alone and 17 by uniform alone, McNemar's
  exact p = 0.0032, answered share +0.092 [0.032, 0.148]. VT holds the
  difference: 19 to 0 there, 21 to 17 over the other four tasks.
- **At 32K the closing answer is yes:** semantic 100 of 250, uniform 52;
  58 to 10, p = 2.4e-9, +0.192 [0.132, 0.252]. VT gives 33 to 0; the
  other four give 25 to 10, MK-1, MK-2 and MV leaning to semantic, none
  alone significant (p ≥ 0.15), and MK-3 answered by neither (corrected
  by #209; this said each task leaned to semantic).
- **Where the plans left the needle**, the share of the needle's (layer,
  page) cells at INT2 at 80%, 8K / 32K, semantic against uniform: VT 3% /
  3% against 70% / 70%; MK-1 63% / 52% against 74% / 68%; MV 60% / 49%
  against 73% / 66%; MK-3 63% / 68% against 74% / 71%; MK-2 75% / 70%
  against 69% / 72%. MK-2's and MK-3's haystack is needle-shaped fillers.
- **At 70% semantic loses MK-3**, 17 against uniform's 46 at 8K
  (p = 3e-8) and 0 against 38 at 32K (p = 7e-12), and MK-2 at 32K, 35
  against 46 (p = 0.0034). Uniform at 70% held no cell at INT2. Semantic
  held 6% and 9% of MK-3's cache there, 4% and 8% of the needle's cells, and an
  INT2 cell on the needle in 44 and 47 of 50 runs. At 70% semantic won
  MV at 8K, 41 against 31 (p = 0.0063).
- **Eviction** answered at most 11 of 50 on the needle tasks, and 48 and
  43 of VT at 8K, 43 and 39 at 32K.

## As measured (#209): on Llama-3.2-1B, at 8K

Llama-3.2-1B-Instruct (ADR-0003's amendment), the same tasks, seeds, conditions
and tests, each set screened for this model. Every figure is from the summaries
`continuation-summary` `aae583e9…`, `rq3-summary` `dd008159…` and `ruler-summary`
`740cdeb2…`, or, where pooled over tasks or against eviction, from the runs
they name.

**RULER.** FP16 answered 50 of the first 52, 56, 120, 95 and 177 drawn (MK-1,
MK-2, MK-3, MV, VT). Static INT2 answered 20, 6, 0, 1 and 0, so every task is
below its ceiling. Bits at 70% / 80% are 4.98 / 3.40 for every source.

| Task | INT2 | INT4 | INT8 | Sem. 70 / 80 | Uni. 70 / 80 | Rand. 70 / 80 | Evict 70 / 80 |
|---|---|---|---|---|---|---|---|
| MK-1 | 20 | 50 | 50 | 50 / 43 | 50 / 29 | 49 / 34 | 24 / 22 |
| MK-2 | 6 | 46 | 50 | 47 / 25 | 48 / 15 | 48 / 22 | 6 / 4 |
| MK-3 | 0 | 31 | 49 | 32 / 0 | 32 / 0 | 27 / 0 | 1 / 1 |
| MV | 1 | 41 | 49 | 45 / 36 | 45 / 11 | 41 / 8 | 11 / 4 |
| VT | 0 | 24 | 41 | 42 / 30 | 31 / 3 | 33 / 3 | 39 / 37 |

- **The closing answer is yes:** semantic 134 of 250, uniform 58; 80 to 4,
  McNemar's exact p = 2.1e-19, +0.304 [0.244, 0.364]. Without VT 52 to 3,
  p = 1.5e-12: unlike Qwen2.5's at 8K, it does not rest on VT. Per task 15 to
  1 (MK-1), 11 to 1 (MK-2), 0 to 0 (MK-3), 26 to 1 (MV), 28 to 1 (VT). Against
  random 73 to 6.
- **At 70%** 216 against 206, 20 to 10, p = 0.099. Semantic did not lose MK-3
  as on Qwen2.5 (32 against 32); VT 14 to 3, p = 0.013.
- **The needle's cells at INT2 at 80%**, semantic against uniform: MK-1 50%
  against 84%, MV 41% against 71%, VT 12% against 76%, MK-3 74% against
  80%, MK-2 76% against 76%. MK-2's 25 against 15 is not explained by them.
- **Eviction** 68 at 80%, 81 at 70%, against 134 and 216. On VT it was not told
  apart from semantic: 37 against 30 at 80% (13 to 6, p = 0.17), 39 against
  42 at 70%.
- **VT's 30 tokens.** 57 of the 127 VT samples FP16 did not answer ran to
  RULER's 30-token answer limit, as did 18 of the 50 it did. The limit is kept,
  and binds every condition alike.

**Continuation, 50 passages.** At 80%, mean KL 0.050 semantic, 0.194 uniform,
0.158 random, lower than both on all 50 passages, difference from uniform
-0.144 [-0.159, -0.130], top-1 0.897 / 0.801 / 0.815. At 70%, 0.0017 / 0.0049 /
0.0089, lower than uniform on 49 passages and than random on 50.

**Needles at 8K.** FP16 answered the first 50. At 80% semantic 50, uniform 41,
random 46: 9 to 0 against uniform, p = 0.0039, and 4 to 0 against random,
p = 0.125. Static INT2 26, so the set is not at its ceiling as it is on
Qwen2.5-1.5B (45). At 70% every source answered 50. Eviction was not run.

**S5, one run each** (`survival` `71a80153…`, `121e14dd…`). Static ran out of
memory at 23,040 positions, 1990 MiB taken. Adaptive (semantic) completed 32,767 in 154
plans, 1954 MiB taken, 910 MiB returned and seen, every plan to a granule, its
chunks sized by time under pressure (0.8 s), unlike #108's run on Qwen2.5-1.5B.
