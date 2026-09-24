# A page is born at its tier, and attention reads it as decode would

#18 runs the engine with its cache held entirely at FP16, INT8, INT4 or INT2,
chosen by configuration: Milestone 0's static operation. Three requirements
meet here.

- ADR-0005: the page being filled cannot be quantised, because a key
  channel's scale spans the whole page. Its positions stay FP16 until it is
  full.
- #18: nothing may change a page's tier after allocation. Moving pages
  between tiers is Milestone 2's, and must not arrive early by the back door.
- Measurement: perplexity is scored by prefilling a window, and a window's
  numbers must say what generating it would. The review of the first design
  found they did not. There, every full page was quantised before attention,
  including the page a query is on. So a query read its own page through a
  scale taken partly from positions after it, which is information from the
  future. And no query ever read the FP16 page a decoding query reads.

## Decision

At a quantised tier:

- **A page of positions is allocated at the cache's tier, once all its
  positions are reserved, and sealed once, when its last one is stored.** If
  the rows being stored cover the page whole, they are quantised straight
  into it. If they finish a page an open page began, the open page is. A
  page's bytes are exactly quantise_page's for its positions, however they
  arrived.
- **Attention is causal as decode is.** A query reads every page before its
  own as sealed, and its own page at FP16: the positions its step brought
  from the step's own rows, and the earlier ones from the open page they
  were written to. A prompt prefilled at once and the same tokens decoded one
  at a time therefore read the same cache. They choose the same tokens, up to
  where cuBLAS rounds a key differently in a one-row GEMM than in a long one
  (ADR-0006, note from #15).
- **The attention kernel's query tiles are aligned to absolute positions.**
  No tile then spans two pages, and each tile knows which page its queries
  are on. P must be a multiple of the tile, 16, which the ablation's 16, 32
  and 64 all are. Each query's output depends on no other row, so where the
  tiles fall changes no bit of any output.
- **Each layer has two open pages**, `(layer, -1)` and `(layer, -2)`, FP16,
  from the same VMM allocator, allocated once. A step that begins mid-page and
  ends on a later page needs the first page's earlier positions to stay
  readable until attention has run, while its last, partial page needs
  somewhere to go. The second open page is where that goes.
- **A sealed page is dequantised as it is loaded**, exactly as
  dequantise_page does it, one fp32 fused multiply-add rounded once to fp16.

At FP16 nothing changes: the open page is simply a layer's last page, and the
paged path is bit-identical to what it was.

A diagnostic rides on the same machinery. With `Halves::Keys` or
`Halves::Values`, pages are stored at FP16, and when a page is sealed only
the named half is replaced by the tier's round trip. A tier's cost can then
be split between keys and values without a second layout. The engine never
uses this unless asked (`Engine.kv_halves`).

## Considered options

- **Allocate a page at FP16 and move it to the tier when it fills.** Simpler
  to write, and exactly the tier change #18 forbids. The page `(l, i)` would
  be at two tiers in its life, and the move would be Milestone 2's
  requantisation arriving unannounced.
- **Keep the open pages in plain device buffers.** They are cache memory, and
  ADR-0007 requires cache memory to come from the VMM allocator, so that what
  is freed reaches the driver.
- **Quantise every full page before attention, as the first design did.** No
  second open page and no tile alignment. But prefill then reads the cache
  differently from decode, and its scales look ahead. Perplexity would
  measure neither the model generating nor anything else.
- **Dequantise every page into an FP16 scratch buffer before attention.** No
  new kernel layout, but a workspace that grows with the context, which #15
  removed, and three passes over the cache's bytes where one will do.

## Consequences

- At a quantised tier each query reads at most P - 1 positions at FP16, those
  before it on its own page. KIVI keeps 128 positions at full precision, and
  KVQuant quantises every position attended to. Every result at a quantised
  tier is qualified by this page, and comparisons with either are not like
  for like.
- A quantised cache holds two address ranges: its tier's, and the open pages'
  at FP16. On Qwen2.5-1.5B the open pages are 56 FP16 pages, 1.75 MiB, which
  the driver backs with one 2 MiB granule. `paged_cache_bytes` computes the
  whole footprint from the layout, and tools/tier_footprint.py logs it
  against what the driver reports for this process alone.
- The first code that changes a page's tier will be Milestone 2's. This
  layout leaves it room: a page could be requantised into a new allocation at
  another tier and the old one freed, with the open pages untouched.
