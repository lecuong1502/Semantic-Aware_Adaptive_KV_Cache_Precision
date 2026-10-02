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

**Birth tier**:
The tier a page of positions is allocated and sealed at: its tier map's, or
its cache's. It is where a page starts; a requantisation plan may move it
later (#88).

**Shadow**:
A downgraded page's FP16 bytes, copied to pinned host memory before its
first downgrade and kept for the session, so that a later downgrade can
quantise from FP16 and an upgrade can restore the page bit for bit (#88,
#94). A page has one. A move from it to a quantised tier, down or up,
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
boundaries bear no relation to pages or granules. The bare word "chunk" means
this and nothing else.
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
engine drains pressure events between steps and records them.
_Avoid_: attribution (that names the process a spike is laid to), alert

**Headroom**:
The device's free memory as the driver reports it (NVML): what any process,
the engine or another, could still allocate. Pressure levels classify it.
_Avoid_: free VRAM percentage (thresholds are absolute), available memory

**Poll**:
One reading of headroom by the pressure monitor, on its fixed schedule, every
50 ms. Hysteresis and detection latency are counted in polls.
_Avoid_: sample (the recorder's word), tick

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

**Requantisation plan**:
The list of `(layer, page, current_tier, target_tier)` entries the Precision
Controller emits in response to a pressure level. Producing a plan is a pure
decision; nothing has moved until the KV Cache Manager applies it.
_Avoid_: re-quant plan, migration, schedule

**Downgrade / Upgrade**:
Moving a page to a lower-precision tier to reclaim bytes, or back toward FP16
to recover quality once pressure passes. Always name the direction; "requantise"
alone is ambiguous.

**Recency floor**:
The most recent `N` token positions, excluded from downgrade regardless of
importance score, on the assumption that recent context is almost always
relevant.
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
