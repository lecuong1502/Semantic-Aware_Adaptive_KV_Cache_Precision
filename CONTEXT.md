# MicroInfer

A single-user LLM inference engine that reallocates KV cache precision at
runtime in response to VRAM contention it does not control. The language below
exists because the surrounding literature uses several of these words to mean
different things, and because the project's own research notes were not yet
consistent.

## Language

### The cache

**Page**:
The fixed-size unit of KV cache memory: the keys and values for a span of `P`
consecutive token positions, in one layer. It is the unit that carries a
precision tier, receives an importance score, and appears in a requantisation
plan.
_Avoid_: block, chunk, slab

**Page table**:
The mapping from a logical page — identified by `(layer, page_index)` — to its
physical location, precision tier, and quantisation scales. vLLM calls this a
*block table*; the paper should note the correspondence once, and the codebase
should not use that name.
_Avoid_: block table

**Slot**:
A page's position within its precision tier's address range, counted from the
low end. A tier's live pages always occupy slots `0..n-1` with no holes;
freeing a page moves the tier's last page into the vacated slot. A slot is
where a page is *now*, never an identity: a page's identity is
`(layer, page_index)`.
_Avoid_: offset, index (that is `page_index`), position (that is a token's)

**Granule**:
The unit in which the driver backs a tier's address range with physical
memory, as `cuMemGetAllocationGranularity` reports it. Memory returns to the
driver a whole granule at a time, and only once no live page reaches into it.
_Avoid_: chunk, block, slab

**Spare granule**:
An empty granule a tier keeps mapped beyond its tail, so that the next
pages at the tier need nothing from the driver. An adaptive cache keeps one
at each quantised tier, for an emergency plan's first downgrades: a
downgrade maps its new page before it frees the old (#107). Kept as far as
the driver allows; a spare it refuses is not an error.
_Avoid_: reserve (reserving is address space, ADR-0007), headroom

**Precision tier**:
The numeric format a page's keys and values are stored in: `FP16`, `INT8`,
`INT4`, or `INT2`. A tier belongs to a `(layer, page)` pair, not to a token
position, so one span of text may be held at different tiers in different
layers.
_Avoid_: precision level, quantisation level, bit-width

**Open page**:
Where a layer's positions wait while their page's `P` positions are still
being written. A key channel's scale spans the whole page, so they cannot be
quantised yet and stay at `FP16`: a mechanical requirement, independent of the
recency floor, though the two agree (ADR-0005). At `FP16` it is simply the
layer's last page; at a quantised tier it is one of two FP16 pages of the
layer's own, `(layer, -1)` and `(layer, -2)`, reused span after span, so that
every other page is written once, at its tier (ADR-0011).

**Tier map**:
The tier each page of positions of a cache is born at, by (layer, page):
row l for layer l, page i at [l][i], and the cache's own tier beyond a row.
_Avoid_: precision map, quantisation plan (a requantisation plan moves pages
that exist)

**Mixed-tier cache**:
A paged cache whose pages are not all at one tier: born at a tier map's
tiers, or moved since by requantisation plans. One attention kernel reads
each page at its own tier, from the page table (ADR-0015). The paper and
the thesis call its manager the *paged mixed-precision cache manager*; the
two name the same thing.
_Avoid_: heterogeneous cache

**Birth tier**:
The tier a page of positions is allocated and sealed at: its tier map's, or
its cache's. It is where a page starts; a requantisation plan may move it
later (#88).

**Shadow**:
A downgraded page's FP16 bytes, copied to pinned host memory before its
first downgrade and kept until the cache is released, so that a later
downgrade can quantise from FP16 and an upgrade can restore the page bit
for bit (#88, #94). A page has one. A move from it to a quantised tier, down or up,
uploads it to an FP16 page of the allocator's, `(layer, -4)`, for as long as
the move takes (#95, #96); an upgrade to FP16 copies it into the new page.
_Avoid_: backup, host copy, offload (offloading moves a page off the device;
a shadow sits beside it)

**Staging page**:
Where a downgrade or an upgrade writes a page's new bytes, at the target tier, while the
page they replace is still read: the page table entry `(layer, -3)`, which
takes the page's own entry when the old page is freed (#93).
_Avoid_: temporary page, scratch page

**Seal**:
To write a page of positions at its birth tier, once and for good, when its
last position arrives: quantised from the rows that brought it, or from the
open page it filled, or, for a page born at FP16 in a cache that seals,
copied as it came. A cache seals when any of its pages is born at a quantised
tier, or when asked to, so that it can downgrade pages born at FP16
(ADR-0011 and its amendments). A sealed page is never written again: a
downgrade puts a new page in its place.
_Avoid_: flush, commit, finalise, close
_Avoid_: partial page, current block, tail page (the tail is the allocator's
last slot, ADR-0007)

**Scale metadata**:
A quantised page's scales and zero-points: one of each per `(head, channel)`
for its keys and per `(head, token)` for its values, in fp16. The same size at
every quantised tier, 1280 bytes on Qwen2.5-1.5B.
_Avoid_: quantisation parameters, header

**Effective bits**:
The bits a quantised tier spends per cached element once its scale metadata
is counted: 8.63 for INT8 on Qwen2.5-1.5B, not 8. The only figure a
compression ratio may be quoted from.
_Avoid_: bit-width (that is the nominal code width)

### Inference

**Prefill chunk**:
A span of consecutive token positions of a prompt that one step of prefill
runs through the model together, `prefill_chunk` of them at a time. It is a
unit of work, not of memory: it sizes the workspace, never the cache, and its
boundaries bear no relation to pages or granules. While an adaptive engine's
monitor reads YELLOW or RED, a chunk is fewer positions, sized from what the
steps before it cost, a fixed part and a part per position, so that a step
takes about `pressure_chunk_seconds`, or about twice its fixed part where
the fixed part alone overruns that (#135). The bare word "chunk" means this and
nothing else.
_Avoid_: batch (there is one sequence), block, window (that is the model's
context window)

### The control loop

**Pressure level**:
The VRAM Monitor's classification of currently available device memory, as
`GREEN`, `YELLOW`, or `RED`. It describes the *machine's* state, never a
page's. It is set from headroom by thresholds in MiB, and a new level is
reported only once K polls in a row have read that side of the current one.
The thresholds and K are ADR-0013's (`monitor.DEFAULT`), set from RQ1's data
by a rule.
_Avoid_: pressure state, memory level

**Pressure event**:
A transition between pressure levels, as the monitor reports it: when, the
levels before and after, the headroom that settled it, and its **own/others
split**, the device memory the engine's process holds and every other
process holds, with how much each changed since the transition before. The
engine drains pressure events between steps and records them; an adaptive
engine answers the latest YELLOW or RED with a requantisation plan (#105).
_Avoid_: attribution (that names the process a spike is laid to), alert

**Headroom**:
The device's free memory as the driver reports it (NVML): what any process,
the engine or another, could still allocate. Pressure levels classify it.
_Avoid_: free VRAM percentage (thresholds are absolute), available memory

**Poll**:
One reading of headroom by the pressure monitor, on its fixed schedule, every
50 ms. Hysteresis and detection latency are counted in polls.
_Avoid_: sample (the recorder's word), tick

**Attention scorer**:
What keeps every page's importance score: each decode step's attention mass,
from the attention kernel, folded on the device into an EWMA, every 4th
step by default (ADR-0016). It scores; it decides nothing. The paper and
the thesis call it the *attention importance scorer*, and ADR-0016 the
importance scorer; the three name the same thing.
_Avoid_: importance model, scorer policy

**Importance score**:
A page's accumulated share of attention mass, as maintained by the Attention
Scorer. It is an input to a requantisation plan, not a tier.
_Avoid_: attention score, relevance, weight, saliency

**Attention mass** (of a page):
A page's share of one query head's softmax in one decode step: the sum of
the attention weights over the page's positions, so that a layer's pages,
for each head, sum to 1. The attention kernel writes it when asked (#98);
the scorer turns it into an importance score.
_Avoid_: attention score (a score is a pre-softmax logit), page weight

**Score source**:
Where a requantisation plan's scores come from: the scorer's importance
scores (semantic), or one of two baselines that stand in for them, uniform
(RQ3's main baseline, every page alike, a partial step spread across
positions) and random (a seeded control). Every source plans to the same
byte target (#104). Only semantic scores are importance scores.
_Avoid_: policy (a source changes the scores, not the planner), scorer

**Requantisation plan**:
The `(layer, page, current_tier, target_tier)` entries the Precision
Controller emits in response to a pressure level, in the order they are to
be applied (held as parallel arrays, one entry each). Producing a plan is a
pure decision; nothing has moved until the KV Cache Manager applies it.
_Avoid_: re-quant plan, migration, schedule

**Precision controller**:
The pure logic, with no device code, that turns a pressure level, the
headroom, the pages' scores and their tiers into a requantisation plan, or
at GREEN into upgrades (ADR-0016). Every score source plans through it.
_Avoid_: policy engine, scheduler (there is none to consult)

**Byte target**:
What a requantisation plan reclaims: the bytes that bring headroom back to
YELLOW's threshold plus a margin (ADR-0013, #101). A plan meets it with
downgrades, and overshoots by less than its last one.
_Avoid_: quota, budget (the budget is a plan batch's time)

**Marginal cost** (of a downgrade):
A page's importance score times the error one tier down adds, per byte the
downgrade saves. The controller takes the cheapest downgrade, again and
again, until the byte target is met (#101).
_Avoid_: priority, penalty

**Marginal gain** (of an upgrade):
A page's importance score times the error one tier up removes, per byte
the upgrade takes: a downgrade's marginal cost, the other way. Upgrades
come largest gain first, the reverse of downgrades (#103).
_Avoid_: benefit, priority

**Planner options**:
The parts of ADR-0016's policy an ablation may vary, recorded in every
plan (ADR-0018, #144, #145): the lowest tier a plan may use; a marginal
cost of the score alone; a first page never downgraded; and layer weights.
Their defaults are the policy itself.
_Avoid_: flags, modes

**Layer weight**:
A layer's sensitivity, the KL divergence from FP16 of the next-token
distribution with that layer alone at INT2, divided by the layers' mean.
With weights, the planner multiplies each page's score by its layer's
weight before planning, so marginal costs and gains are of weighted scores
(#145). Without them, every layer counts alike.
_Avoid_: layer importance (importance is a page's score)

**Upgrade cooldown**:
The time an upgrade waits after the last downgrade, 5 s by default, so
that pages are not moved back as soon as they were moved down (#103). In
the engine it runs from the last downgrade or the end of the last
pressure, the later, so that a pulse train whose later pulses find nothing
to downgrade holds upgrades back until it stops (#106).
_Avoid_: hysteresis (#88 dropped a hysteresis of its own), backoff

**Plan record**:
A requantisation plan the engine made, with the pressure event it answered,
the plan batches it applied and how it ended: applied, cancelled at GREEN,
replaced by a later plan, or stopped (#105). Upgrades the engine makes at
GREEN are recorded the same way, with the event that brought GREEN, and a
YELLOW or RED cancels those not yet made (#106).
_Avoid_: plan log, history

**Persisting plan**:
A requantisation plan made not on a pressure event but because the level
stayed YELLOW or RED while the cache sealed new pages: it moves those pages,
for the headroom the monitor reads now. Its plan record says so.
_Avoid_: follow-up plan, re-plan

**Emergency plan**:
A RED plan for no headroom, made when allocating a step's pages raised
OutOfMemory, applied at once; the step is then run once more (#107). Its
plan record carries the error rather than a pressure event.
_Avoid_: panic plan, OOM handler

**Ending**:
How a generation ended, `Engine.ending`: its state, complete or exhausted,
the positions its cache held, the tokens it kept, and, if exhausted, why
(#107). Complete is every token asked for, or up to an end of sequence.
_Avoid_: session (a passive session is a recording), result, status

**Exhausted**:
The ending of a generation in an adaptive engine that ran out of memory
and could not go on: an allocation failed and an emergency plan did not
answer it, having nothing to downgrade, freeing nothing, or the step
failing again; or memory was refused outside a step's pages. The engine
returns the tokens kept, records the positions reached, and releases the
cache; nothing is raised (#107). An engine that does not adapt raises
OutOfMemory, as it did.
_Avoid_: crashed, aborted, OOM-killed

**Plan batch**:
The downgrades of a requantisation plan, or the upgrades of an upgrade
plan, applied between two decode steps: at YELLOW and at GREEN, as many as
fit a per-step time budget, by the time the engine measures, the plan
having estimated them; at RED, the whole plan.
Not a batch of sequences: there is one sequence.
_Avoid_: chunk (a prefill chunk is positions)

**Downgrade / Upgrade**:
Moving a page to a lower-precision tier to reclaim bytes, or back toward FP16
to recover quality once pressure passes. Always name the direction; "requantise"
alone is ambiguous.

**Recency floor**:
The most recent `W` token positions, `W = 128` by default and a parameter
every plan records, excluded from downgrade regardless of importance score
and policy, on the assumption that recent context is almost always relevant
(#102). A page any of whose positions is among them is never downgraded,
the open page with them. The engine's `kv_recency_floor` sets it for the
ablations of W (#158); a controlled plan's share is still of the bytes
outside the default floor.
_Avoid_: recency window, protected window, sliding window

### The measurements

**Contention**:
VRAM consumed by processes other than MicroInfer. The project's premise is that
this is unpredictable and invisible to any scheduler MicroInfer could consult.
_Avoid_: pressure (that is the *classification* of contention), load

**Reclaimable bytes**:
The device memory that downgrading a given set of pages would return to the
allocator. The project's central quantity: the mechanism is only useful when
reclaimable bytes exceed the size of a realistic contention spike.

**KV footprint fraction**:
KV cache size as a proportion of total device memory, for a given model and
context length. It varies by an order of magnitude across models with different
GQA ratios, and every reported result is qualified by it.

**Simulated contention**:
Device memory taken on a schedule of (time, bytes taken) by a process of its
own, the contention simulator: memory only, a granule at a time through the
engine's allocator, so that it is another process to NVML and to the monitor,
as real contention is. Its schedules come from synthetic patterns or from a
contention trace. It *takes* memory; *holding* is the engine's (Hold).
_Avoid_: load generator, stress test (it runs no kernels)

**Pulse**:
A trapezoid of memory the contention simulator takes and gives back on a
schedule: an amplitude, a linear ramp up and down, and a plateau between. It
is what the simulator plays; a spike is what is found in a trace.
_Avoid_: spike (for what the simulator plays), burst

**True RED episode**:
A run of the recorder's samples whose headroom is below the monitor's RED
threshold, from the first below to the first back at or above: the ground
truth the monitor is evaluated against, never its own events.
_Avoid_: RED event (that is the monitor's report of one)

**Detection latency**:
From a true RED episode's start to the first RED event that detects it, in
ms and in polls. Only episodes of 100 ms or more are counted.

**False negative / false positive**:
A true RED episode of 100 ms or more with no RED event; a RED event with no
true RED over the polls that settled it. False negatives shorter than K + 1
polls and a recorder sample are counted apart, as ones the monitor cannot be
sure to catch by design.
_Avoid_: miss, false alarm

**Replay**:
Simulated contention whose schedule comes from a contention trace: the device
stream's used memory, less the engine's where the trace ran beside it, above
its least over the window replayed, sample by sample at 50 Hz. It reproduces
every other process, desktop noise included, not only those a spike names. A
replay is recorded as the trace was, and compared with it sample by sample and
spike by spike.
_Avoid_: playback, re-run (a scenario run again is a new trace, not a replay)

**Survival experiment**:
Milestone 2's closing measurement (#108): a generation fills a long context
at FP16 while simulated contention, taken when the cache reaches a chosen
position and kept to the end, leaves the engine its **shortfall** short of
its full cache. A static engine runs out; an adaptive one is to downgrade
and complete. The generation waits at that position until the simulator has
taken its memory, so contention arrives at the same place in every run.
_Avoid_: stress test, survival test (it is logged, not asserted)

**Contention scenario**:
One of the thesis's six kinds of simulated contention a generation faces,
S1 to S6: the pulse grid, a pulse train, Poisson arrivals of RQ1's spikes,
RQ1's recordings replayed, a lasting drop (the survival experiment's
step), and a pulse released (ADR-0014, amendment #190). Its schedule is a
pure function of a run's seed and RQ1's measures. In the survival matrix's
code, `scenario` is this; RQ1's scripted desktop actions are named by their
recordings there.
_Avoid_: calling RQ1's scripted desktop actions a contention scenario

**Base**:
What the contention simulator takes first, and keeps to the end of the
run, for a scenario's schedule to play on: set from the headroom when
contention begins, to leave the scenario's start, so that headroom begins
in GREEN whatever else the desktop holds that day.
_Avoid_: floor, offset

**Survival matrix**:
The survival experiment run under each condition B1 to B6 against each
contention scenario, three runs a cell (#190, #191): whether a 32K
generation completes, and how its engine reacts. A cell is a condition
and a scenario; a matrix run, one of its runs, by its seed.

**Turn**:
One exchange of a conversation the engine continues over a single cache: a
prompt appended to what the cache holds, and the tokens decoded after it.
RQ3's experiments compress the cache between two turns (ADR-0017).
_Avoid_: round, message (a turn holds both sides of the exchange)

**Snapshot**:
What a conversation's cache held after its turns, copied to the host: every
layer's keys and values as they were stored, at FP16, the importance
scores, and the token not yet fed (#172). A conversation started from it
goes on as the one it was taken from would have, to the bit, so the
conditions of a needle sample that hold turn 1 alike run it once. Only a
conversation made for one, which keeps on the host what its cache stores,
can take one, once, and only while every page is at FP16.
_Avoid_: fork, clone (the two conversations share nothing), checkpoint

**Plan point**:
Where a task's controlled plan is applied (ADR-0017): a needle sample's end
of turn 1, a passage's end of its warm tokens, an anchor-fact dialogue's end
of its last turn. A page's damage is measured there (#188).

**Page damage**:
What downgrading one (layer, page) alone to INT4 at the plan point costs:
the mean KL from FP16 over the measured positions, the answer FP16 gives or
a passage's measured tokens (#188). Set against the page's importance score
by Spearman's rho, it tests the policy's premise page by page.

**Condition**:
How an RQ3 run holds its cache (ADR-0017, #147): FP16; a score source at a
level, a controlled plan from semantic, uniform or random scores for 70% or
80% of the FP16 bytes outside the recency floor; or a static tier, every
page born at INT4 or INT2; an ablation; or eviction at a level. RQ3's two experiments,
question-blind and question-aware, are when a condition's plan is made. The
survival matrix's conditions are B1 to B6: static engines at FP16, INT4 and
INT2, and the adaptive engine from the uniform, random and semantic sources.
_Avoid_: arm, treatment, setting

**Full policy**:
ADR-0016's policy as RQ3 runs it, the semantic source at 80%,
question-blind: the condition each ablation is measured against, on the
same samples (ADR-0018).
_Avoid_: baseline (uniform is the baseline), default policy

**Ablation**:
A condition that is the full policy with one element changed (ADR-0018,
#158): the recency floor at 0 or 512, scoring every step, the EWMA weight
at 0.05 or 0.5, importance alone, no INT2, layer weights, or a protected
first page; and no INT2 at 70% too, against the semantic source there,
since INT4 cannot give back 80% of the bytes. Each is reported by its
change in needle accuracy from the condition it changes, McNemar's test
against it and the effective bits it held. Those
that change only the plan share turn 1 with the full policy; those of the
scorer run it alone. The monitor's K is an ablation of RQ2, not a condition.
_Avoid_: arm, treatment

**Eviction**:
A condition of H2O's family (#160): at a level, the pages the semantic
source's controlled plan would downgrade most, those it leaves lowest
first, are evicted instead until they give back the same bytes. An evicted
page is freed, its memory back to the driver, and no query attends to its
positions again; none comes back. A shadow it had stays in host memory,
never read, until the cache goes: the shadow store only ever adds. It is
measured against the same pages masked, kept and read by no query, to the
bit; masked as a later position is, by the attention kernel's -inf. A
page, not a token, is what is evicted: the cache's unit.
_Avoid_: dropping, pruning, offloading (that keeps the bytes elsewhere)

**Controlled plan**:
A requantisation plan the engine is asked for between two turns, for a byte
target and a score source given by the experiment rather than by a
pressure event: the same planner, with no monitor or simulator involved,
so that every source reclaims the same bytes (ADR-0017).
_Avoid_: forced plan, synthetic pressure

**Needle** / **Haystack**:
A synthetic fact the model cannot guess, a named code, planted at a known
depth in a haystack, articles of WikiText-2's test split joined to the
context's length. The needle's depth fixes the page that holds it
(ADR-0017).
_Avoid_: probe, distractor (the haystack is not adversarial)

**Screening** / **Screened set**:
Running every needle sample at FP16 before the comparison, and replacing
one answered wrongly by the next of its depth from the seed, a
**replacement**, until each depth has five answered correctly: the
screened set, so that the model's failures are not counted as the cache's
(ADR-0017, #148).
_Avoid_: filtering, cleaning (nothing is dropped unrecorded)

**Question-blind** / **Question-aware**:
Whether a controlled plan is made before the question is asked, from the
attention of an earlier turn (question-blind, RQ3's primary experiment), or
after the first tokens of an answer to it (question-aware, an upper bound).
_Avoid_: offline, online (both are mid-session)

**Continuation fidelity** / **Passage**:
RQ3's secondary task (ADR-0017, #153): a passage of WikiText-2's test
split, its prefix prefilled and its **warm** tokens forced a position a step
so the scorer has scores, is compressed by a controlled plan, and its
**measured** tokens are **teacher-forced**, fed the passage's own tokens
rather than the model's choices. Its measures are top-1 agreement with the
**reference**, the same passage with no plan applied, which is FP16's, and
the KL divergence from the reference's logits, position by position.
_Avoid_: perplexity (that is ADR-0011's measure of a whole tier), free
generation (nothing is generated)

**Anchor-fact dialogue** / **Gap**:
RQ3's third task (ADR-0017, #154): a conversation of ten turns, each a
WikiText-2 passage and a request to summarise it, its reply decoded. One
turn opens with a **fact**, a needle stated as "Note for later: ...". After
the last turn a controlled plan is applied, and the question asks for the
fact back, scored as a needle answer is. Its **gap** is the turns between
the fact's turn and the question, 0 to 9. The fact's positions depend on
the replies before it, so a run records them. Its **recall** is the share
of dialogues whose answer holds the fact's code, what a needle's accuracy
is; #154 names it so.
_Avoid_: depth (that is a needle's place in one document), distance (it
reads as positions, not turns)

**Controlled restore** / **Turn 3**:
RQ3's recovery task on the needle set (ADR-0017, #155). After turn 2, the
compressed turn, a controlled restore stands for the GREEN after pressure,
as a controlled plan stands for the pressure: the upgrade policy's plan for
headroom enough for every upgrade, with no monitor and no cooldown, so that
every page holding a shadow is back at FP16. Turn 3 then asks the same
question again, over a cache that holds FP16's bytes but for turn 2's
answer, decoded while compressed. A turn-3 answer that is turn 2's word for
word **repeats** it.
_Avoid_: recompute (the shadow restores the page; nothing is recomputed),
re-ask (turn 3 is the term)

**Pulse and release** / **Time to restore** / **Thrash**:
The recovery task under the simulator (#155): it takes memory part way
through a 32K generation and gives it back after the pulse. The time to
restore runs from the end of pressure, the GREEN after the last YELLOW or
RED, to the first step whose cache holds no page below FP16. Thrash is
downgrades and upgrades alternating: the applied plans turning from one to
the other more than once, a downgrade after the first upgrade, or upgrades
that pressure cancelled.
_Avoid_: recovery time (recovery is the task), oscillation

**End-to-end run**:
The question-blind needle task with the plan made by the monitor and the
controller, not controlled (#157): part way through turn 1 the simulator
takes what leaves the engine's headroom below YELLOW by the bytes a
controlled plan at 80% reclaims, the same rule for semantic and uniform.
A run logs the bytes its pages gave back beside its answer.
_Avoid_: live run, real-pressure run (the pressure is simulated)

**Closing question**:
The one test Milestone 3 is closed on: whether the semantic source answers
more needle samples than uniform at 80% of the FP16 bytes outside the
recency floor reclaimed, question-blind, by McNemar's exact test on the
samples only one of them answered (ADR-0017). Its significance level, 0.05
two-sided, and the rule that semantic must also answer more, were set in
#152, after the runs were logged; ADR-0017 fixed the comparison and the
test. Its answer, yes or no, is reported whatever it is.
_Avoid_: hypothesis, headline result (the comparison was fixed before the results)

**RQ3 summary**:
One benchmark-log entry, `rq3-summary`, pooling the screened set's needle
runs, the latest of each sample in each experiment under each condition,
by their hashes (#152): accuracy by condition, level and depth with
intervals, the semantic source paired against uniform and random, and the
closing question's answer. Every other comparison in it is descriptive.
_Avoid_: report, results table (it is a log entry the tables are printed from)

**32K subset**:
Ten needle samples, one a depth, of haystacks of about 32K tokens, run
question-blind under FP16 and both sources at 80%, unscreened (#156). Its
summary, `rq3-long-summary`, gives the direction semantic and uniform point,
with no test claimed: ten samples are too few for McNemar's test to reject
(ADR-0017).
_Avoid_: 32K set, long set (it confirms the 8K set; it is not one of its own).
In code, "long" names it, as in `--long` and `summarise_long`.

**Shortfall**:
The device memory a survival experiment's contention leaves an engine short
of its full FP16 cache: the simulator takes the headroom, less its own
context and what the cache has still to grow, plus the shortfall. A rule,
not a number of bytes, so that a static and an adaptive run face the same
contention whatever else the desktop holds.
_Avoid_: deficit (a spike's), overshoot (a plan's, of its byte target)

**Contention trace**:
A recording of contention as the recorder takes it, in two streams on one
monotonic clock: the **device stream**, the driver's free and used memory at
50 Hz, and the **processes stream**, beside it at 5 Hz, the memory each GPU
process holds with the GPU's P-state and clocks. The processes stream is what
attributes a spike to the process that caused it. A scenario adds action
labels: the start and end of each action, stamped on the recorder's clock as
they arrive, which tie every sample to the action in progress. A trace closed
cleanly says it is complete; one cut short reads back to within a second.
_Avoid_: log (the benchmark log is a different, hash-chained record), profile

**Spike**:
Free memory at least 64 MiB below the baseline, the rolling median of the
preceding 5 s, for at least 100 ms. The baseline holds from a spike's start for
at most one window; a spike still below it then is a **lasting drop**, and does
not recover. A spike ends when its deficit falls below half the threshold, and
one cut by a gap in the trace or by its end is **censored**. Each spike has an
amplitude, a rise time (10% of the amplitude up to its 90% peak), a duration
(at or above the threshold) and a recovery (the peak's 90% back to 10%).
_Avoid_: burst, dip (a dip too shallow or too brief is not a spike), step
(an action's)

**Attribution**:
The process a spike is laid to: the one whose memory gained most between the
last 5 Hz process sample before the spike's rise and the samples during its
peak, if it gained at least a quarter of the amplitude; it is reported with its
share. A process new since that sample, or a pid now under another name, gained
all it holds. Otherwise the spike names no process, and says why.
_Avoid_: blame, cause (a process can gain with a spike it did not start)

**Hold**:
A generation the engine keeps in progress for as long as a scenario runs, so
that contention is recorded against a working engine: the prompt fills the
context less a span, and the span is decoded in **passes**, each going back to
the end of the prompt over the pages already held, so memory stays at the full
context. FP16 only, since a pass rewrites positions in place. A hold that runs
out of memory records it, with the allocation that failed and the headroom
then, and ends: running out is the outcome RQ1's with-engine runs observe. It
can only happen while the cache still grows: once the first pass is done a
hold allocates next to nothing, and contention meets the other processes.
_Avoid_: session (a passive session is a recording), loop

**Scenario**:
A scripted run of desktop actions, repeated, during which a contention trace is
recorded; RQ1's traces are scenarios and passive sessions, the latter with no
script at all.
_Avoid_: benchmark (that measures the engine), workload

**Span**:
One labelled stretch of a scenario, between a start and an end label, at a
fixed place in its schedule: an action, or one of the stages an action is
recorded in, as tabs are at 1, 5 and 10. A **wait span** is the exception: it
lasts until a person does what an action needs, such as starting the video
call, and the schedule after it is counted from its end. A span a wait span
prepares, if no one comes, is **skipped**: logged, not labelled.
_Avoid_: step, segment

**Action**:
One thing a user does on the desktop during a scenario, such as opening a
browser or starting a game, marked in the trace by an **action label** at its
start and another at its end. Nothing else in the project is an action: the
entries of a requantisation plan are not.
_Avoid_: event (the label is the event; the action is what it marks), step
