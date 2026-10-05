# A page changes tier at runtime, from its FP16 shadow

Amends ADR-0011, superseding its rule that a page keeps its tier. That ADR held each page at the tier it was born at for the
life of the cache: static operation, Milestone 0's. It left room for the
next step: "a page could be requantised into a new allocation at another
tier and the old one freed". Milestone 2 (#88) takes that step. Under
pressure the engine downgrades pages to give memory back to the driver, and
when the pressure passes it restores them. #90 to #96 built this one ticket
at a time, each recorded as an amendment to ADR-0011. This ADR states the
decision as a whole, with the options it rejected.

Three requirements meet here.

- **ADR-0007:** memory a downgrade frees must reach the driver, or the
  monitor cannot see it and the other process cannot take it.
- **Milestone 0's error figures** (`kv-quantisation-roundtrip`,
  `perplexity`) are only worth anything if a page at a tier has exactly that
  tier's error, however it got there.
- **An upgrade has to restore quality,** not just bytes.

## Decision

**The mixed-tier cache.**
- **Each page of positions is born at the tier a tier map names** (#90). A
  page beyond its layer's row is born at the cache's tier.
- **A cache that can downgrade seals every page,** including those born at
  FP16 (`always_seal`, #93). One rule then holds for every page.
- **One attention kernel reads each page at its own tier,** from the page
  table, in one pass with no workspace (#91). Query tiles are aligned to
  pages (ADR-0011), so a tile reads one tier and the branch does not diverge
  within a warp.

**A downgrade** of a sealed page, between decoding steps (#93):
1. A page is allocated at the target tier under the cache's *staging
   page*, `(layer, -3)`, while the old page is still read.
2. It is quantised from FP16:
   - while the page is FP16, from the page itself;
   - after that, from its *shadow*, uploaded to an FP16 page of the
     allocator's, `(layer, -4)`, held only for the move (#95).
3. The page table names the new page.
4. The old page is freed, and its tier's tail page moves into the slot
   (ADR-0007).

If step 1 cannot allocate, the cache is as it was.

**The shadow** (#94). Before a page's first downgrade, its FP16 bytes are
copied to pinned host memory, and they are kept until the cache is released. The old
page is freed only once the copy is complete. A page born at a quantised
tier has no shadow and never moves.

**Every move quantises from FP16.** A page at tier T is therefore always
exactly `quantise_page(FP16, T)`, however many moves it took to get there.
Each tier has one error, the one Milestone 0 measured, and the controller
(ADR-0016) plans with those errors.

**An upgrade restores from the shadow** (#96). To FP16, the shadow is copied
into a new FP16 page, exactly the page's bytes before its first downgrade.
To an intermediate tier, the shadow is uploaded and quantised there. The
steps are a downgrade's, and the shadow is kept.

**Moves are synchronous, on the decoding thread, between steps.** A move
therefore cannot race a kernel, and no kernel sees the cache half converted.

**An adaptive cache keeps one spare granule mapped at each quantised tier**
(ADR-0007, amendment of #107). A downgrade maps its new page before it frees
the old, so on a device with nothing left to give, an emergency plan could
not otherwise begin.

## Considered options

- **Dequantise up.** Upgrade a page by decoding its codes back to FP16.
  Rejected: it recovers nothing. The page would have FP16's size and the
  lower tier's error.
- **Quantise from the current codes.** Downgrade INT8 to INT4 from the INT8
  codes. Rejected: errors compound along the path a page took. A page at
  INT4 would no longer have INT4's measured error, and the controller's
  marginal costs would be wrong.
- **Downgrade in place.** Rejected: pages at different tiers have
  different sizes, and ADR-0007 packs each tier's pages in its own address
  range. A smaller page written into a larger slot frees nothing the driver
  sees. The research notes record this as the one approach that cannot
  work.
- **Keep the FP16 copy on the device.** Rejected: it would cost exactly the
  memory a downgrade is meant to free.
- **Recompute on upgrade.** Re-run the prefix's forward pass for the page's
  positions. Deferred as an ablation or later work. It needs no host
  memory, but it costs a forward pass over the prefix, and a layer's page
  depends on every layer below it.
- **Evict shadows when host memory is short.** Not needed. The shadows of
  every page of Qwen2.5-1.5B at 32K positions take about 0.9 GiB, against
  23 GiB of host RAM.
- **Copy to the host asynchronously.** Deferred, to be added only if
  measurement shows the copy dominating a move. It does not: see below.
- **Offload pages to the host instead of downgrading** (FlexGen's family).
  Rejected for this milestone. A page that is not on the device cannot be
  read by the attention kernel, so offloading changes what is computed, not
  only how precisely.

## Consequences

**The correctness gate holds.**
- A cache that reached its tiers through runtime moves reads, bit for bit,
  as one built at those tiers.
- An upgrade to FP16 restores the page's bytes bit for bit.
- After every plan, the driver sees the memory returned that
  `tier_page_bytes` predicts and the allocator reports, to a granule.

**What a move costs** (`requantisation-latency`, Qwen2.5-1.5B, one page of
one layer, median of 256, at 51cb21f):

| From → to | Median | Entry |
|---|---:|---|
| FP16 → INT8 / INT4 / INT2 | 13 / 19 / 30 µs | `1c4483a4…`, `9bc1a746…`, `99bb009d…` |
| The same, a first downgrade, pinning the shadow | 23 / 29 / 39 µs | `a1b2c50a…`, `140f162f…`, `d0e2af6b…` |
| INT8 → INT4 / INT2, from the shadow | 32 / 37 µs | `276f446e…`, `4e92c564…` |
| Any tier → FP16, the shadow copied back | 8 µs | `4f4ca93e…`, `a5b8bf8a…`, `25d4ae48…` |

- The shadow copy is about 7 µs of a first downgrade, 19% to 33% of it. That
  is not enough to justify an asynchronous copy.
- At these latencies a 20 ms budget holds about 500 moves (ADR-0016).

**The survival experiment** (#108, `survival` entries `e2f55721…` and
`1fdbeebd…`, at 6440b38).
- Qwen2.5-1.5B prefilled to 32K positions while the simulator left it
  256 MiB short of its full FP16 cache. The static engine ran out of memory
  at 20992 positions.
- The adaptive engine applied 85,596 downgrades and returned 746 MiB, which
  the driver saw as 746 MiB, every plan within a granule. It completed every
  token.

**Host memory now holds part of the cache's state.** A shadow lives as long
as its cache, so the host memory an adaptive engine uses grows with the
pages it has downgraded.
