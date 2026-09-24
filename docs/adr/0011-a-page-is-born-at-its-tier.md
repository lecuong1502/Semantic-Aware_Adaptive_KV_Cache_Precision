# A page is born at its tier, and the open page is kept apart

#18 runs the engine with its cache held entirely at FP16, INT8, INT4 or INT2,
chosen by configuration: Milestone 0's static operation. Two requirements
meet here. ADR-0005 says the page being filled cannot be quantised, because a
key channel's scale spans the whole page, so its positions must stay FP16
until the page is full. And #18 says nothing may change a page's tier after
allocation: moving pages between tiers is Milestone 2's, and must not arrive
early by the back door.

## Decision

At a quantised tier:

- **A page of positions is allocated at the cache's tier, once all its
  positions are reserved, and written once, when its last one is stored.**
  If the rows being stored cover the page whole, they are quantised straight
  into it; if they finish a page the open page began, the open page is. The
  page's bytes are then exactly quantise_page's for its positions, however
  they arrived.
- **The open page is a page of its own**: one FP16 page per layer, page table
  entry `(layer, kOpenPage)`, with `kOpenPage = -1` so it can never be a
  position's page. It is allocated from the same VMM allocator on the first
  reserve, and reused for every span in turn.
- **Attention dequantises as it loads.** The paged attention kernel has a
  third layout: the first pages in the table are at the tier, and the entry
  after them is the open page. Each code becomes `fp16(fma(q, scale, zero))`,
  exactly as dequantise_page computes it, so attention over the cache is, to
  the bit, attention over the FP16 pages dequantise_page would give.

At FP16 nothing changes: the open page is simply a layer's last page, and the
paged path is bit-identical to what it was.

## Considered options

- **Allocate a page at FP16 and move it to the tier when it fills.** Simpler
  to write, and exactly the tier change #18 forbids: the page `(l, i)` would
  be at two tiers in its life, and the move would be Milestone 2's
  requantisation arriving unannounced.
- **Keep the open page in a plain device buffer.** It is cache memory, and
  ADR-0007 requires cache memory to come from the VMM allocator so that what
  is freed reaches the driver. A cudaMalloc'd buffer would be the one part of
  the cache the driver never sees returned.
- **Dequantise every page into an FP16 scratch buffer before attention.** No
  new kernel layout, but a workspace that grows with the context, which #15
  removed, and three passes over the cache's bytes where one will do.

## Consequences

- At a quantised tier the last `length % P` positions of each layer are FP16:
  a residual of at most one page, where KIVI keeps 128 positions. Every
  result at a quantised tier is qualified by it. In a 2048-token perplexity
  window it is at most 31 positions, 1.5%.
- A quantised cache holds two address ranges, its tier's and the open pages'
  at FP16. The open pages are 28 FP16 pages on Qwen2.5-1.5B, 896 KiB, which
  the driver backs with one 2 MiB granule. `paged_cache_bytes` computes the
  whole footprint from the layout, and a test measures it against what the
  driver reports for this process alone: at Qwen2.5-1.5B's 32K window it is
  896, 486, 262 and 150 MiB at FP16, INT8, INT4 and INT2, to the byte.
- INT2 reclaims 112 MiB beyond INT4 there, the figure ADR-0008 argued from.
- The first code that changes a page's tier will be Milestone 2's, and this
  layout leaves it room: a page could be requantised into a new allocation
  at another tier and the old one freed, with the open page untouched.
