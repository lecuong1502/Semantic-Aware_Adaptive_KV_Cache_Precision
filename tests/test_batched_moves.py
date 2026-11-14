"""A plan's page moves applied in batches on the device (#205, Seam B).

move_pages takes a list of (layer, page, tier) and moves every page to its
tier at once: downgrades and upgrades alike, each from the page's FP16
bytes, its own while it is FP16 and its shadow once it is not (#95, #96).
The target pages of the whole batch are allocated first, so a failed
allocation leaves nothing changed (ADR-0015); first downgrades copy their
FP16 bytes to their shadows in one copy; each target tier is quantised in
one launch; and the old pages are freed together, the tails moving into
their slots (ADR-0007).

The proof is equality with the per-page path, which the tests of
test_quantised_pages.py hold to quantise_page: after the same moves, every
page holds the same bytes at the same tier, every shadow the same bytes,
and attention reads the same bits.
"""

import numpy as np
import pytest
from conftest import each
from test_kv_pages import CFG, LAYERS, PAGE_TOKENS, normal
from test_quantised_pages import attend, pages_for, snapshot, store_in_runs

from microinfer import _microinfer
from microinfer._microinfer import PagedKVCache, Tier
from microinfer.model import tier_page_bytes

device = _microinfer.device
Halves = device.Halves
#: What the per-page path does, by direction.
STAGING = (device.staging_page, device.shadow_upload_page)


def twins(page_tokens, pages, seed, born=None):
    """Two caches that seal, holding the same keys and values in every
    layer, born at the same tiers; their allocators, and the rows."""
    rng = np.random.default_rng(seed)
    kv_heads, hd = CFG.num_key_value_heads, CFG.head_dim
    seq = pages * page_tokens + 5
    k = [normal(rng, seq, kv_heads, hd) for _ in range(LAYERS)]
    v = [normal(rng, seq, kv_heads, hd) for _ in range(LAYERS)]
    born = born or [[Tier.FP16] * pages for _ in range(LAYERS)]
    made = []
    for _ in range(2):
        allocator, _ = pages_for(CFG, page_tokens, Tier.FP16, tier_map=born)
        cache = device.KVPages(allocator, LAYERS, page_tokens, CFG.num_key_value_heads,
                               CFG.head_dim, Tier.FP16, Halves.Both, born, True)
        for layer in range(LAYERS):
            start = store_in_runs(cache, layer, k[layer], v[layer],
                                  [page_tokens + 3, seq - page_tokens - 3])
        made.append((allocator, cache))
    return made, k, v, start


def per_page(cache, moves):
    """The moves, one page at a time, as Engine._apply made them before #205."""
    for layer, page, target in moves:
        current = cache.page_tier(layer, page)
        (cache.downgrade if int(target) > int(current) else cache.upgrade)(layer, page, target)


def assert_same(a, b, pages):
    """Two caches hold the same pages, bytes, tiers and shadows."""
    (alloc_a, cache_a), (alloc_b, cache_b) = a, b
    held_a, held_b = snapshot(alloc_a), snapshot(alloc_b)
    assert held_a.keys() == held_b.keys()
    for key in held_a:
        np.testing.assert_array_equal(held_a[key], held_b[key], err_msg=str(key))
    assert cache_a.shadow_count == cache_b.shadow_count
    for layer in range(LAYERS):
        for page in range(pages):
            assert cache_a.page_tier(layer, page) == cache_b.page_tier(layer, page), (layer, page)
            if cache_a.has_shadow(layer, page):
                np.testing.assert_array_equal(cache_a.shadow(layer, page),
                                              cache_b.shadow(layer, page))
    for tier in Tier.__members__.values():
        assert alloc_a.mapped_bytes(tier) == alloc_b.mapped_bytes(tier), tier
    assert not [key for key in held_b if key[1] < 0 and key[1] not in device.open_pages], (
        "a batch left a page of its own behind")


#: Every tier pair, down and up, in two batches: the first from FP16, the
#: second from the tiers the first left, pages of both layers and of every
#: slot, the tail among them.
FIRST = [(0, 0, Tier.INT8), (0, 1, Tier.INT4), (0, 2, Tier.INT2), (0, 3, Tier.INT8),
         (0, 4, Tier.INT4), (0, 5, Tier.INT2), (1, 1, Tier.INT8), (1, 2, Tier.INT4),
         (1, 3, Tier.INT2), (1, 5, Tier.INT2), (1, 6, Tier.INT4), (1, 7, Tier.INT8),
         (0, 7, Tier.INT4)]
SECOND = [(0, 0, Tier.INT4), (0, 3, Tier.INT2), (0, 1, Tier.INT2),   # down from a shadow
          (0, 2, Tier.INT4), (0, 5, Tier.INT8), (1, 5, Tier.FP16),   # up from INT2
          (1, 2, Tier.INT8), (1, 6, Tier.FP16),                      # up from INT4
          (1, 1, Tier.FP16),                                         # up from INT8
          (0, 6, Tier.INT2), (1, 0, Tier.INT8),                      # first downgrades
          (0, 7, Tier.INT8)]


def test_a_batch_leaves_every_page_as_the_per_page_path_does():
    """At both page sizes, over a cache born at FP16 and one born with a
    page at INT8: the same moves, page by page in one cache and in two
    batches in its twin, leave every page with the same bytes at the same
    tier, every shadow the same, the same memory mapped at every tier, and
    attention reading the same bits, with and without the key bias."""
    heads, kv_heads, hd = CFG.num_attention_heads, CFG.num_key_value_heads, CFG.head_dim

    def same(page_tokens, mapped):
        pages = 8
        born = None
        if mapped:
            born = [[Tier.FP16] * pages for _ in range(LAYERS)]
            born[1][4] = Tier.INT8  # born there, so never moved
        (a, b), k, v, start = twins(page_tokens, pages, page_tokens, born)
        for batch in (FIRST, SECOND):
            per_page(a[1], batch)
            b[1].move_pages(batch)
            assert_same(a, b, pages)
        rng = np.random.default_rng(1)
        for layer in range(LAYERS):
            for bias in (None, normal(rng, kv_heads, hd) * 30):
                q = normal(rng, len(k[layer]) - start, heads, hd)
                np.testing.assert_array_equal(
                    attend(CFG, a[1], layer, q, k[layer], v[layer], bias, start),
                    attend(CFG, b[1], layer, q, k[layer], v[layer], bias, start))

    each([(p, m) for p in PAGE_TOKENS for m in (False, True)], same,
         name=lambda c: f"P={c[0]}{' mapped' if c[1] else ''}")


def test_a_batch_records_what_it_took():
    """The last batch's moves, its shadows taken, and its wall time."""
    (_, (_, cache)), _, _, _ = twins(PAGE_TOKENS[0], 8, 3)
    cache.move_pages(FIRST)
    record = cache.last_batch
    assert record["moves"] == len(FIRST)
    assert record["shadows_taken"] == len(FIRST)
    assert record["seconds"] > 0
    cache.move_pages(SECOND)
    assert cache.last_batch["shadows_taken"] == 2  # (0, 6) and (1, 0) only


def test_a_failed_allocation_mid_batch_leaves_the_cache_unchanged():
    """A batch whose target tier fills partway: the allocation fails, and
    every page keeps its bytes and its tier, no shadow is taken, no page of
    the batch's own is left, and the memory mapped at every tier is what
    it was. A batch that fits then goes through."""
    page_tokens = PAGE_TOKENS[0]
    kv_heads, hd = CFG.num_key_value_heads, CFG.head_dim
    rng = np.random.default_rng(205)
    pages = 6
    capacity = [0] * 4
    capacity[int(Tier.FP16)] = LAYERS * (pages + len(device.open_pages))
    capacity[int(Tier.INT8)] = capacity[int(Tier.INT2)] = 4096
    capacity[int(Tier.INT4)] = 3
    allocator = PagedKVCache(tier_page_bytes(CFG, page_tokens), capacity)
    cache = device.KVPages(allocator, LAYERS, page_tokens, kv_heads, hd, Tier.FP16,
                           Halves.Both, [], True)
    seq = pages * page_tokens
    for layer in range(LAYERS):
        store_in_runs(cache, layer, normal(rng, seq, kv_heads, hd),
                      normal(rng, seq, kv_heads, hd), [seq])
    cache.move_pages([(0, 0, Tier.INT2)])
    before = snapshot(allocator)
    mapped = [allocator.mapped_bytes(t) for t in Tier.__members__.values()]
    batch = [(0, 1, Tier.INT8), (0, 2, Tier.INT4), (1, 0, Tier.INT4), (1, 1, Tier.INT4),
             (0, 0, Tier.INT4), (1, 2, Tier.INT8)]  # four to INT4, which holds three
    with pytest.raises(ValueError, match="full"):
        cache.move_pages(batch)
    after = snapshot(allocator)
    assert after.keys() == before.keys()
    for key in before:
        np.testing.assert_array_equal(after[key], before[key], err_msg=str(key))
    assert [allocator.mapped_bytes(t) for t in Tier.__members__.values()] == mapped
    assert cache.shadow_count == 1
    assert cache.page_tier(0, 0) == Tier.INT2
    assert all(cache.page_tier(layer, p) == Tier.FP16 for layer in range(LAYERS)
               for p in range(pages) if (layer, p) != (0, 0))
    cache.move_pages(batch[2:])  # three to INT4
    assert cache.page_tier(0, 0) == Tier.INT4 and cache.page_tier(1, 2) == Tier.INT8
    assert cache.page_tier(0, 1) == Tier.FP16


def test_what_a_batch_refuses():
    """Before anything changes: a page twice in one batch, a page not held,
    a move to the page's own tier, and a page born at a quantised tier,
    which has no FP16 bytes to move from. An empty batch moves nothing."""
    page_tokens = PAGE_TOKENS[0]
    pages = 4
    born = [[Tier.FP16] * pages for _ in range(LAYERS)]
    born[0][3] = Tier.INT4
    ((allocator, cache), _), _, _, _ = twins(page_tokens, pages, 9, born)
    before = snapshot(allocator)
    for batch, match in (([(0, 1, Tier.INT4), (0, 1, Tier.INT2)], "twice"),
                         ([(0, 1, Tier.INT4), (0, 40, Tier.INT2)], "page 40"),
                         ([(0, 1, Tier.INT4), (0, 2, Tier.FP16)], "already"),
                         ([(0, 1, Tier.INT4), (0, 3, Tier.INT2)], "shadow")):
        with pytest.raises(ValueError, match=match):
            cache.move_pages(batch)
        after = snapshot(allocator)
        assert after.keys() == before.keys()
        for key in before:
            np.testing.assert_array_equal(after[key], before[key], err_msg=str(key))
        assert cache.shadow_count == 0
    cache.move_pages([])
    assert cache.last_batch["moves"] == 0
