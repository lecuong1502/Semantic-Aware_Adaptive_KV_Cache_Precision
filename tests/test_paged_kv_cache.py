"""The KV cache allocator, against NVML (ADR-0007).

The contract is not "a page was freed". It is "NVML-reported free memory went
up". A page freed inside a pool the engine still holds is invisible to
`nvmlDeviceGetMemoryInfo`, so every test of memory here reads NVML itself
(`nvml.py`) and none asserts which granule was unmapped (CONTRIBUTING).

**Measuring under contention.** NVML is device-wide, and on this machine other
processes — the display server, a browser, an editor — move its free reading by
up to 6 MiB in either direction while this process does nothing, measured over
thirty idle intervals. That is three 2 MiB granules. Widening a tolerance to
cover it would also cover a leak of the same size, which is the failure this
file exists to catch (ADR-0007, note from #6). So each memory assertion repeats
its operation and takes the median: an external swing lands on one repetition,
not on most of them. Measured over thirty cycles of 32 granules, every
allocation and every release moved NVML by 32 granules to within 0.16 of one.

Page sizes are fractions of the granule the driver reports, never byte
literals: the allocator is handed page sizes and never interprets them, and a
test must not assume a 2 MiB granule any more than the allocator may.
"""

import random
import statistics

import numpy as np
import pytest

import nvml
from conftest import each
from microinfer import _microinfer
from microinfer._microinfer import PagedKVCache, PageNotFound, Tier

TIERS = [Tier.FP16, Tier.INT8, Tier.INT4, Tier.INT2]
REPEATS = 7


@pytest.fixture(scope="module", autouse=True)
def context_outlives_every_cache():
    """The CUDA context costs about 87 MiB of device memory on this machine,
    and it is not the cache's to measure.

    A cache retains the device's primary context and releases it when it dies.
    If nothing else holds the context, the release destroys it and the next
    cache creates it again, and a test that builds and drops caches reads 87 MiB
    that no reservation took. That happened here first. The engine always holds
    the context through the runtime API, so the tests do the same: any runtime
    call initialises it for the life of the process.
    """
    _microinfer.device_memory_info()


def granule() -> int:
    return PagedKVCache([1] * 4, [0] * 4).granule_bytes


def make_cache(page_bytes, capacity_pages=1024) -> PagedKVCache:
    if isinstance(capacity_pages, int):
        capacity_pages = [capacity_pages] * 4
    return PagedKVCache(page_bytes, capacity_pages)


def halving_pages(divisor: int = 64) -> list[int]:
    """One page size per tier, each half the one above, as the tier ladder's
    are. At divisor 64 and a 2 MiB granule the FP16 page is 32 KiB."""
    top = granule() // divisor
    return [top, top // 2, top // 4, top // 8]


def median_delta(operate, undo) -> float:
    """The median change in NVML free bytes across REPEATS runs of `operate`,
    with `undo` restoring the state between runs."""
    deltas = []
    for _ in range(REPEATS):
        before = nvml.free_bytes()
        operate()
        deltas.append(nvml.free_bytes() - before)
        undo()
    return statistics.median(deltas)


def pattern(layer: int, page_index: int, nbytes: int) -> np.ndarray:
    """Bytes that could belong to no other page."""
    return np.random.default_rng([layer, page_index]).integers(0, 256, nbytes, dtype=np.uint8)


# -- granules ---------------------------------------------------------------


def test_the_granule_is_the_drivers_and_reserving_takes_no_device_memory():
    """The granule is queried with cuMemGetAllocationGranularity, never
    hardcoded (2 MiB on the RTX 4050 Laptop, driver 580.178.04: ADR-0007),
    and each tier's range is whole granules. Reserving 7.5 GiB across the
    four tiers, more than the whole card, leaves NVML still: had it
    allocated anything it could not have succeeded."""
    g = granule()
    assert g > 0 and g & (g - 1) == 0, f"granule {g} is not a power of two"
    cache = make_cache([g // 3, g // 5, g // 7, g // 11], capacity_pages=[10, 20, 30, 0])
    for tier in TIERS:
        assert cache.reserved_bytes(tier) % g == 0
        assert cache.reserved_bytes(tier) >= cache.page_bytes(tier) * [10, 20, 30, 0][tier.value]

    held = []
    capacity = (4 << 30) // (g // 64)
    delta = median_delta(lambda: held.append(make_cache(halving_pages(), capacity)),
                         lambda: held.clear())
    assert abs(delta) < g / 2, f"reserving moved NVML free memory by {delta / 2**20:.2f} MiB"
    cache = make_cache(halving_pages(), capacity)
    assert sum(cache.reserved_bytes(t) for t in TIERS) > nvml.memory().total
    assert all(cache.mapped_bytes(t) == 0 for t in TIERS)


# -- allocation takes whole granules ----------------------------------------


def test_allocating_pages_takes_whole_granules():
    """A partly filled granule is still a whole granule to the driver, and one
    more page crossing into the next granule costs a whole one more."""
    def takes(granules_worth):
        g = granule()
        page = g // 64
        count = int(granules_worth * 64)
        expected = -(-count * page // g) * g  # ceil to whole granules
        cache = make_cache([page] * 4, capacity_pages=count)

        def fill():
            for i in range(count):
                cache.allocate(0, i, Tier.FP16)

        def empty():
            for i in range(count):
                cache.free(0, i)

        taken = -median_delta(fill, empty)
        fill()
        assert cache.mapped_bytes(Tier.FP16) == expected
        assert abs(taken - expected) < g / 2, (
            f"NVML used rose by {taken / g:.2f} granules for {expected // g} mapped")

    each([1, 2.5, 40.25], takes)


# -- the central test -------------------------------------------------------


def test_emptying_a_granule_returns_it_to_nvml_with_its_last_page_and_not_before():
    """Releasing enough pages to empty a granule raises NVML-reported free
    memory by at least the granule size.

    The pages freed are the *lowest* in the range, not the tail, so every one
    of them triggers a tail swap: memory comes back only because the tail
    retracts. Freeing the lowest pages and seeing nothing return would be the
    private-pool failure ADR-0007 was written against. And the counterpart:
    65 pages reach one page into a second granule; freeing any one of them
    empties it, and freeing another empties nothing."""
    def returns(granules_emptied):
        g = granule()
        page = g // 64
        per_granule = g // page
        held = (granules_emptied + 3) * per_granule
        freed = granules_emptied * per_granule
        cache = make_cache([page] * 4, capacity_pages=held)
        for i in range(held):
            cache.allocate(0, i, Tier.FP16)

        def release():
            for i in range(freed):
                cache.free(0, i)

        def restore():
            for i in range(freed):
                cache.allocate(0, i, Tier.FP16)

        gained = median_delta(release, restore)
        assert gained >= granules_emptied * g, (
            f"freeing {granules_emptied} granule(s) of pages returned "
            f"{gained / 2**20:.2f} MiB to the driver")

    each([1, 64], returns)

    g = granule()
    cache = make_cache([g // 64] * 4, capacity_pages=128)
    for i in range(65):  # one page into the second granule
        cache.allocate(0, i, Tier.FP16)
    assert cache.mapped_bytes(Tier.FP16) == 2 * g
    cache.free(0, 0)
    assert cache.mapped_bytes(Tier.FP16) == g
    cache.free(0, 1)
    assert cache.mapped_bytes(Tier.FP16) == g


# -- the tail swap ----------------------------------------------------------


def test_freeing_a_page_moves_the_tail_into_its_slot_with_its_contents():
    """Freeing a page other than the tail moves the tail into its slot, the
    tail's entry with it; freeing the tail moves nothing. Contents survive the
    swap, also where a page is three eighths of a granule, so pages straddle
    granule boundaries and the copy reads across two physical allocations.
    One tier's swap leaves the other tiers alone."""
    cache = make_cache(halving_pages())
    for i in range(4):
        cache.allocate(0, i, Tier.INT8)
    cache.free(0, 1)
    assert cache.pages(Tier.INT8) == [(0, 0), (0, 3), (0, 2)]
    assert cache.locate(0, 3) == (Tier.INT8, 1), "the tail's entry was not moved"
    assert (0, 1) not in cache, "the freed page's entry was not removed"
    assert cache.locate(0, 0) == (Tier.INT8, 0) and cache.locate(0, 2) == (Tier.INT8, 2)

    cache = make_cache(halving_pages())
    for i in range(3):
        cache.allocate(0, i, Tier.FP16)
    cache.free(0, 2)
    assert cache.pages(Tier.FP16) == [(0, 0), (0, 1)]

    def contents_survive(divisor):
        page = int(granule() / divisor)
        cache = make_cache([page] * 4, capacity_pages=64)
        for i in range(20):
            cache.allocate(0, i, Tier.FP16)
            cache.write(0, i, pattern(0, i, page))
        for i in [0, 3, 7, 8, 1]:
            cache.free(0, i)
        for layer, page_index in cache.pages(Tier.FP16):
            np.testing.assert_array_equal(cache.read(layer, page_index),
                                          pattern(layer, page_index, page))

    each([64, 8 / 3], contents_survive)

    pages = halving_pages()
    cache = make_cache(pages)
    for tier in TIERS:
        for i in range(3):
            cache.allocate(tier.value, i, tier)
            cache.write(tier.value, i, pattern(tier.value, i, pages[tier.value]))
    cache.free(Tier.INT4.value, 0)
    for tier in TIERS:
        for layer, page_index in cache.pages(tier):
            np.testing.assert_array_equal(cache.read(layer, page_index),
                                          pattern(layer, page_index, pages[tier.value]))
    assert cache.pages(Tier.FP16) == [(0, 0), (0, 1), (0, 2)]


# -- the page table ---------------------------------------------------------


def test_the_page_table_is_keyed_by_layer_and_page_and_says_what_it_refuses():
    """The same page_index in two layers is two pages, and may sit in two
    tiers: a tier belongs to a (layer, page) pair (CONTEXT.md). A page cannot
    be allocated twice; an unknown page is PageNotFound, a KeyError; a full
    tier says so and keeps what it held; a page is written and read whole,
    in any dtype of the right size."""
    cache = make_cache(halving_pages())
    cache.allocate(0, 5, Tier.FP16)
    cache.allocate(1, 5, Tier.INT2)
    assert cache.locate(0, 5) == (Tier.FP16, 0) and cache.locate(1, 5) == (Tier.INT2, 0)
    cache.free(0, 5)
    assert (0, 5) not in cache and (1, 5) in cache

    cache.allocate(0, 0, Tier.FP16)
    with pytest.raises(ValueError, match="already allocated"):
        cache.allocate(0, 0, Tier.INT8)
    assert issubclass(PageNotFound, KeyError)
    for call in (lambda: cache.free(9, 9), lambda: cache.read(9, 9), lambda: cache.locate(9, 9)):
        with pytest.raises(PageNotFound):
            call()

    cache = make_cache(halving_pages(), capacity_pages=[2, 1, 1, 1])
    cache.allocate(0, 0, Tier.FP16)
    cache.allocate(0, 1, Tier.FP16)
    with pytest.raises(ValueError, match="full"):
        cache.allocate(0, 2, Tier.FP16)
    assert cache.pages(Tier.FP16) == [(0, 0), (0, 1)]

    pages = halving_pages()
    cache = make_cache(pages)
    cache.allocate(0, 0, Tier.INT4)
    with pytest.raises(ValueError, match="bytes"):
        cache.write(0, 0, np.zeros(pages[Tier.INT4.value] + 1, np.uint8))
    data = np.arange(pages[Tier.INT4.value] // 2, dtype=np.float16)  # any dtype, right size
    cache.write(0, 0, data)
    np.testing.assert_array_equal(cache.read(0, 0).view(np.float16), data)


# -- packing ----------------------------------------------------------------


def test_pages_stay_packed_through_a_random_sequence():
    """Fragmentation is impossible by construction; this asserts it after every
    operation of a long random sequence, together with the two things packing
    is for: memory held is exactly the granules the live pages reach, and every
    live page still holds what was written to it."""
    g = granule()
    pages = [g // 16, g // 32 * 3, g // 64, g // 128]  # one size straddles granules
    cache = make_cache(pages, capacity_pages=256)
    rng = random.Random(10)
    live: dict[tuple[int, int], Tier] = {}

    for step in range(1500):
        if live and (len(live) > 300 or rng.random() < 0.45):
            key = rng.choice(sorted(live))
            cache.free(*key)
            del live[key]
        else:
            key = (rng.randrange(24), rng.randrange(64))
            if key in live:
                continue
            tier = rng.choice(TIERS)
            if len(cache.pages(tier)) == 256:
                continue
            cache.allocate(*key, tier)
            cache.write(*key, pattern(*key, pages[tier.value]))
            live[key] = tier

        for tier in TIERS:
            slots = cache.pages(tier)
            assert sorted(slots) == sorted(k for k, t in live.items() if t == tier), step
            for slot, key in enumerate(slots):
                assert cache.locate(*key) == (tier, slot), step
            reach = len(slots) * pages[tier.value]
            assert cache.mapped_bytes(tier) == -(-reach // g) * g, step

    for key, tier in live.items():
        np.testing.assert_array_equal(cache.read(*key), pattern(*key, pages[tier.value]))


# -- addresses ---------------------------------------------------------------


def test_no_device_address_crosses_the_boundary():
    """A page moves whenever its tier's tail is retracted, so an address handed
    out would go stale silently. Pages are reached only by key, resolved at the
    time of the call."""
    exposed = [name for name in dir(PagedKVCache)
               if any(word in name.lower() for word in ("addr", "ptr", "pointer", "handle", "base"))]
    assert exposed == []
    cache = make_cache(halving_pages())
    cache.allocate(0, 0, Tier.FP16)
    for value in (cache.locate(0, 0), cache.pages(Tier.FP16)):
        assert all(isinstance(v, (int, Tier)) for v in np.ravel(np.array(value, dtype=object)))
    assert _microinfer.PagedKVCache is PagedKVCache


def test_running_out_is_an_error_of_its_own_that_names_the_page_and_takes_nothing():
    """Filled until the device has no granule left, the cache raises
    OutOfMemory, a MemoryError, naming the page and the granule it needed,
    and holds exactly what it held before the call: running out is what the
    project studies, so a caller records it and goes on (#52)."""
    size = granule()
    # Room in the address range for more than the device has free, whatever
    # the device.
    capacity = _microinfer.device_memory_info()["free"] // size + 64
    cache = make_cache([size] * 4, capacity_pages=capacity)
    held = 0
    # Dropped however the test ends, pytest.raises holding its traceback or
    # not: the device is full until it is.
    try:
        with pytest.raises(_microinfer.OutOfMemory) as info:
            while True:
                cache.allocate(0, held, Tier.FP16)
                held += 1
        assert isinstance(info.value, MemoryError)
        assert f"page (layer 0, page_index {held})" in str(info.value)
        assert f"{size}-byte granule" in str(info.value)
        assert "bytes were free" in str(info.value)  # read before giving back
        assert cache.mapped_bytes(Tier.FP16) == held * size
        assert (0, held) not in cache
    finally:
        del cache
