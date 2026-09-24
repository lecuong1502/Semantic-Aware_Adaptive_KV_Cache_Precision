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

---

## What the static tiers cost, measured

Measured at 1c175f5, with the tree clean, except FP16's 32K prefill, which
is #15's (entry `56b00ad3`, at 71392f7). Every figure below is in the
benchmark log except one, the unlogged trial named as such.

**Footprint** (`kv-footprint`, Qwen2.5-1.5B, whole 32K window). This is what
the driver reports for this process alone, and it equals `paged_cache_bytes`
at every tier:

| tier | measured | effective bits, open pages counted | vs FP16 |
|---|---:|---:|---:|
| FP16 | 896 MiB | 16.000 | 1.00x |
| INT8 | 486 MiB | 8.656 | 1.85x |
| INT4 | 262 MiB | 4.656 | 3.44x |
| INT2 | 150 MiB | 2.656 | 6.02x |

**Perplexity** (`perplexity`, WikiText-2 test split, all 146 windows of 2048
tokens, causal as above):

| tier | Qwen2.5-0.5B | | Qwen2.5-1.5B | |
|---|---:|---:|---:|---:|
| FP16 | 14.319 | | 9.640 | |
| INT8 | 14.319 | +0.00% | 9.641 | +0.007% |
| INT4 | 14.429 | +0.77% | 9.698 | +0.59% |
| INT2 | 18.148 | +26.7% | 11.541 | +19.7% |

**Against published expectations.** KVQuant (arXiv:2401.18079, Tables 1 and 9)
reports relative increases on LLaMA-7B on WikiText-2 at 2K:

- 4-bit: +5.3% for uniform per-token int4, +0.9% for FlexGen's grouped 4-bit.
- 3-bit, keys per-channel and values per-token: +24.1%.
- 2-bit: +27.3% for KVQuant, +95.2% for FlexGen.

INT4 here sits with the grouped methods, and INT2 with KVQuant-2bit. Neither
is the large divergence #18 asks to investigate. Three differences keep this
from being a like-for-like comparison:

- The models differ, and a 7B model tolerates quantisation better than a
  0.5B or 1.5B one; KVQuant's own table shows the cost falling as the models
  grow.
- Every query here reads up to P - 1 positions of its own page at FP16, where
  KVQuant reads none.
- KVQuant's best 2-bit variant keeps 1% of values as sparse FP16 outliers
  (+5.8%). That is a different scheme, with its own storage, and is left out
  of the comparison for that reason. The plain KVQuant-2bit is the uniform
  quantiser this project's is closest to.

The investigation was needed all the same. On an unlogged trial of four
windows of the 0.5B model, the first design's prefill put INT2 far past
every published 2-bit figure, which *was* a large divergence. It came from the measurement, not the quantiser: every
query read its own page quantised, through scales that looked ahead. The
causal design above removed it.

**Where INT2 loses it** (the Halves diagnostic, 0.5B). Quantising the keys
alone costs +0.32% at INT4 and **+15.9% at INT2**. Quantising the values
alone costs +0.42% and +5.5%. At INT2 the keys cost three times what the
values do. That reverses what the round trip's signal-to-noise ratios
suggested (ADR-0005, note from #17). The whole-head value groups, kept over
KIVI's 32 channels (ADR-0005, note before #18), are therefore not the main
cause; +5.5% bounds what smaller value groups could recover. The two errors
compound: +15.9% and +5.5% together would be +22.3%, where both halves at
once cost +26.7%.

**Generation, read** (`generation-sample`, entries `eb0bac0e` to `1d6bb55e`
for 1.5B and `d93d4c23` to `f8d2fba9` for 0.5B). The prompts are four
retrieval prompts and three WikiText articles to summarise.

- FP16 and INT8 give the same text on 1.5B. On 0.5B they give the same text
  on six prompts; on the seventh INT8's summary differs, and is as faithful.
- INT4 on 1.5B is coherent: every retrieval right, every summary faithful,
  with one name repeated. INT4 on 0.5B is coherent in most places but falls
  into one repetition loop and inverts one answer.
- **INT2 is not coherent.** On 1.5B it still answers all four retrieval
  prompts, and its sentences stay grammatical. But it invents facts
  ("married a woman named Li Bai"), adds a sentence of nonsense to one answer,
  answers once in the user's voice, and falls into a loop, repeating one
  sentence eight times. On 0.5B it degenerates: two of the three summaries
  are "= = = =" to the end.
- INT2, as the tier of a whole cache, fails #18's coherence criterion. Its
  case in ADR-0008 is as a tier for the least important pages, which static
  operation cannot test.

**32K on Qwen2.5-1.5B** (`prefill-throughput`, with the opt-in test passing
at every tier). The whole window prefills at every tier: 45.8, 45.4 and 44.0
tokens per second at INT8, INT4 and INT2, against 45.8 at FP16. The cache at
the peak is 502, 278 and 166 MiB, that is, the footprint above plus the
16 MiB RoPE table. INT2's prefill took 4% longer than FP16's. Each figure is
a single run, and FP16's was made at another commit, so that is an upper
bound on what dequantising inside attention costs, not a measurement of it.
