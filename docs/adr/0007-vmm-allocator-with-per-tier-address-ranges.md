# The KV cache allocator uses the CUDA VMM API, one virtual address range per tier

Reclaimed bytes must return to the *driver*, because `nvmlDeviceGetMemoryInfo`
measures what the driver has handed out. A page "freed" inside a pool the
engine still holds is invisible to NVML, so the response to RED pressure would
be a no-op that cannot be measured. This is the same trap ADR-0002 rejected in
PyTorch's caching allocator; here it would be self-inflicted.

The cache reserves one virtual address range per precision tier with
`cuMemAddressReserve`, and backs each range on demand with physical granules
from `cuMemCreate` mapped by `cuMemMap`. Granule size is typically 2 MiB and
must be queried with `cuMemGetAllocationGranularity` rather than assumed. Pages
are packed from the low end of their tier's range. Downgrading a page
quantises it into the tail of the target tier's range, copies the source
range's last page over the vacated slot, updates two page table entries, and
retracts the source tail; granules that fall empty are released with
`cuMemUnmap` and `cuMemRelease`, and the memory genuinely returns to the
driver.

## Considered Options

- **Slabs from `cudaMalloc` with slab-level compaction.** Uses only the
  familiar runtime API, but reclamation becomes unpredictable: downgrading a
  hundred pages frees nothing if they are spread across slabs, and compacting
  costs time and copies at exactly the moment — RED pressure — when bytes are
  needed immediately. The paper would have to report a reclamation latency that
  depends on fragmentation.
- **A single pool of fixed 32 KiB slots.** §3.4 of the research notes suggests
  over-provisioning slots per tier as "simpler for a first implementation".
  **This is the one option that cannot work.** An INT4 page written into a
  32 KiB slot wastes 24 KiB; nothing returns to the driver; NVML never moves.
  Adopting it would require withdrawing RQ2 from the thesis and replacing it
  with a different question. That is a change of research scope, not an
  implementation shortcut, and the research notes should be corrected.

## Consequences

- A page's physical location moves when its tier's tail is retracted. Nothing
  may cache a page's device address across a requantisation; the attention
  kernel resolves addresses through the page table on every launch.
- Swap-with-tail costs one 32 KiB device-to-device copy per downgraded page,
  regardless of how many pages are downgraded. This is bounded and belongs in
  the overhead measurement §7 asks for.
- The engine links against the CUDA driver API, not only the runtime API.

---

## Note from #5: allocation count is itself a cost

Loading Qwen2.5-0.5B as one `cudaMalloc` per parameter group — 290 of them —
took 1080 MiB from the driver for 942 MiB of weights. Measured on this machine,
the overhead tracks the *number* of allocations rather than their size:

| allocations | claimed | taken | overhead |
|---:|---:|---:|---:|
| 10 | 915.5 MiB | 919.7 MiB | 0.5% |
| 100 | 915.5 MiB | 999.8 MiB | 9.2% |
| 290 | 915.5 MiB | 1158.8 MiB | 26.6% |

Two consequences for this decision.

**It supports the choice.** Packing pages from the low end of one reserved range
per tier means the cache is a handful of granule mappings, not one allocation
per page. At 28,672 pages on the 1.5B at 32K, an allocation-per-page design
would lose more to granularity than the mechanism reclaims.

**It is a live cost elsewhere.** 240 MiB of overhead on a 5762 MiB budget is
roughly 40% of what requantising the whole cache to INT4 would reclaim. Weight
loading should use one arena with offsets rather than an allocation per tensor.
That is not this ticket's work and is filed separately.
