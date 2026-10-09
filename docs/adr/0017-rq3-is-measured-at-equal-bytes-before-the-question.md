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
