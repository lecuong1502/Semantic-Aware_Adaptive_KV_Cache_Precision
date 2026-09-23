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

---

## Note from #6: how to measure the contract, and how not to

This decision's contract — that reclaimed pages genuinely return to the driver —
is verified by reading `cudaMemGetInfo` around an allocator operation. #6 hit
that reading's failure modes on a much smaller test, and #10 should not have to
rediscover them.

**Settle this process's own frees first.** A device buffer dropped by an earlier
test may be collected while a later one is measuring, freeing memory between the
two readings and shrinking the delta. Measured here: an exact 32 MiB allocation
read as 27.1 MiB. `tests/conftest.py::stable_free_bytes` runs `gc.collect()`
before reading, and the allocator tests should use it rather than calling
`device_memory_info` directly.

**Do not assume other processes are the problem.** Sampling free memory sixty
times across three idle seconds moved it by 0.19 MiB in total. External
contention is real on this machine, but it is not what makes a suite flaky —
this project's own deferred frees are.

**Size the allocation so the residual noise is negligible, and measure the
tolerance rather than guessing it.** The noise observed is absolute, a few MiB,
so a small allocation cannot be asserted exactly while a large one can be
asserted within a few percent.

The last point cuts both ways here, and the wrong direction is dangerous. A
tolerance loose enough to stop a test flaking is also loose enough to hide a
leak of the same size — and a leak is exactly what this ADR exists to prevent.
When an allocator assertion goes flaky, the shortfall should be measured before
the bound is widened.

---

## Note from #10: what was observed when it was built

**The granule is 2 MiB** on the RTX 4050 Laptop with driver 580.178.04, as
`cuMemGetAllocationGranularity` reports at `CU_MEM_ALLOC_GRANULARITY_MINIMUM`.
The allocator reads it at construction and nothing assumes it.

**The contract holds exactly, and NVML is read directly.** Reserving 7.5 GiB of
address space, well beyond the 6 GiB card, moves NVML-reported free memory by
nothing. Mapping and releasing 32 granules moves it by 32 granules each way, to
within 0.16 of a granule over thirty cycles. The allocator tests read
`nvmlDeviceGetMemoryInfo` through ctypes (`microinfer/nvml.py`, moved there from
`tests/` by #13) rather than `cudaMemGetInfo`. The two differ by about 90 MiB on
this machine, so they are not interchangeable as absolute readings, even though
their deltas agree.

**Contention on an idle desktop is ±6 MiB.** The display server, a browser and
an editor share the GPU, and they move NVML's free reading by up to three
granules over short intervals while this process does nothing. The allocator
tests take the median of seven repetitions instead of widening a tolerance. A
wider tolerance would also admit a leak of the same size. With granule release
disabled, the central test and six others fail.

**The CUDA context is about 87 MiB, and a cache must not be the last holder of
it.** The cache retains the primary context and releases it on destruction. If
nothing else holds the context, that release destroys it, and the next cache
rebuilds it. A test that builds and drops caches then reads 87 MiB that no
reservation took. The engine holds the context through the runtime API for its
whole life, so this does not affect it, but any measurement of the cache must
be taken with the context already alive.
