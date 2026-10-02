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

## Amendment (#90): a cache built from a tier map

Milestone 2 (#88) builds a cache from a *tier map*, which names the tier
each page of positions is born at; a page beyond its layer's row is born at
the cache's tier. Two things change for such a cache.

- **A cache seals whenever any page is born at a quantised tier,** and then
  every page is sealed, including one born at FP16: it is allocated once all
  its positions are reserved and filled in an open page until then, like
  the others. Sealing an FP16 page copies its rows as they came, keys then
  values, the FP16 layout. One rule then holds for every page of the cache,
  where two (FP16's "the open page is the last page" and the quantised
  tiers' sealing) would have to meet page by page.
- **A page is allocated at its birth tier,** the map's, not the cache's.

A cache whose map names no tier but its own builds, byte for byte, what
this ADR's static operation builds. A sealed page still never changes tier
here; that is #93's, and its own ADR (#109).

## Amendment (#91): attention reads each page at its own tier

Attention reads each sealed page at the tier the page table records for it,
not at the cache's. Beside each layer's table of page addresses, the cache
uploads each page's tier, and the kernel picks the page's format as it
loads it: quant.h's layout at a quantised tier, dequantised as above, or
the FP16 layout. A query's own page is still read at FP16, from the open
page and the step's rows, so causality is unchanged, page by page.

There is still one kernel and one pass, with no workspace. The branch on a
page's tier does not diverge within a warp: a key tile is loaded one key
row at a time, and head_dim, 64 or 128 here, is a multiple of the warp's 32
lanes, so all the lanes of a warp load one page. A cache born at one tier
reads, bit for bit, as the static path did, since each element is decoded
by the same arithmetic.

## Amendment (#93): a sealed FP16 page may be downgraded

The rule that no page changes tier after allocation was Milestone 0's, and
the layout above left it room to go. #93 lets a page of positions sealed at
FP16 move to a quantised tier mid-session, between steps:

1. the page is allocated at the target tier under the cache's *staging
   page*, `(layer, -3)`, while the FP16 page is still read;
2. it is quantised from the FP16 page, which holds its rows as they came,
   so its bytes are exactly those of a page born at the target;
3. the page table names the new page (`PagedKVCache::replace`);
4. the FP16 page is freed, and its tier's tail moves into the slot
   (ADR-0007).

If step 1 cannot allocate, the cache is as it was. A sealed page is still
never written again: a downgrade puts a new page in its place. Attention
reads the page at its new tier from the next launch on, since the free moves
the allocator's generation and the page table is resolved again.

Only a cache that seals can downgrade, since in one that does not every page
is FP16 and its last is still being written. A cache whose pages are all
born at FP16 is asked to seal with `always_seal`, and then seals as one
born partly at a quantised tier does. A diagnostic `Halves` never
downgrades: its pages are FP16 by its definition.

Downgrading from a page already quantised, upgrades, and the FP16 shadow
they need are later tickets of #88. The decision as a whole, with the
options it rejected, is #109's ADR.

## Amendment (#94, #95): every downgrade quantises from FP16

Before a page's first downgrade its FP16 bytes are copied to its *shadow*
in pinned host memory, and the page is freed only once the copy is
complete (#94). A page already below FP16 moves lower by quantising its
shadow, uploaded to an FP16 page of the allocator's, `(layer, -4)`, held
only for the downgrade, never its current codes
(#95); this replaces the #93 amendment's note that it was later work. A
page at tier T is therefore always exactly `quantise_page(FP16,
T)`, however many steps it took to get there, and each tier has one error,
the one Milestone 0 measured. A page born at a quantised tier has no shadow,
and never moves. Upgrades were still to come; #96's amendment below adds
them.

## Amendment (#96): an upgrade restores a page from its shadow

A page moves back up by its shadow too: to FP16 the shadow is copied into
a new FP16 page, exactly the page's bytes before its first downgrade; to an
intermediate tier the shadow is uploaded and quantised there, as a
downgrade from it would be. The steps are a downgrade's (a page at the
target under the staging page, then the page table, then the old page
freed), and the shadow is kept. Dequantising the codes up is rejected, as
#88 decided: it recovers nothing.
