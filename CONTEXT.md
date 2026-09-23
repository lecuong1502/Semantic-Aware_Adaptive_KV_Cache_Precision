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
The list of `(layer, page, current_tier, target_tier)` actions the Precision
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
