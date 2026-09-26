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

**Seal**:
To write a page of positions at a quantised tier, once and for good, when its
last position arrives: quantised from the rows that brought it, or from the
open page it filled. A sealed page is never written again, and never changes
tier (ADR-0011).
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
page's.
_Avoid_: pressure state, memory level

**Importance score**:
A page's accumulated share of attention mass, as maintained by the Attention
Scorer. It is an input to a requantisation plan, not a tier.
_Avoid_: attention score, relevance, weight, saliency

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

**Scenario**:
A scripted run of desktop actions, repeated, during which a contention trace is
recorded; RQ1's traces are scenarios and passive sessions, the latter with no
script at all.
_Avoid_: benchmark (that measures the engine), workload

**Action**:
One thing a user does on the desktop during a scenario, such as opening a
browser or starting a game, marked in the trace by an **action label** at its
start and another at its end. Nothing else in the project is an action: the
entries of a requantisation plan are not.
_Avoid_: event (the label is the event; the action is what it marks), step
