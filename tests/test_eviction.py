"""Eviction at equal bytes, the H2O-style baseline (#160).

An evicted page is freed, its memory back to the driver, and no query
attends to its positions again. The proof is equality, not tolerance:
attention over a cache with pages evicted is, to the bit, attention over
the same cache with those pages masked, kept and read by no query; and a
masked page is masked, attention near what it is over the keys without
its positions.
"""

import numpy as np
import pytest
from test_kv_pages import CFG, LAYERS, PAGE_TOKENS, normal, put
from test_quantised_pages import attend, sealed, sealing_at_fp16, store_in_runs

from microinfer import _microinfer
from microinfer._microinfer import PagedKVCache, Tier
from microinfer.model import tier_page_bytes

device = _microinfer.device


def stored(page_tokens, seq, seed=160):
    """Two caches at FP16 that seal, holding the same keys and values in
    every layer, and those keys and values."""
    rng = np.random.default_rng(seed)
    kv_heads, hd = CFG.num_key_value_heads, CFG.head_dim
    k = [normal(rng, seq, kv_heads, hd) for _ in range(LAYERS)]
    v = [normal(rng, seq, kv_heads, hd) for _ in range(LAYERS)]
    caches = []
    for _ in range(2):
        allocator, cache = sealing_at_fp16(page_tokens)
        runs = [page_tokens + 5, seq - page_tokens - 5]
        for layer in range(LAYERS):
            start = store_in_runs(cache, layer, k[layer], v[layer], runs)
        caches.append((allocator, cache))
    return caches, k, v, start


def test_attention_over_evicted_pages_is_attention_with_them_masked_to_the_bit():
    """Pages evicted from one cache and masked in its twin, page 0 among
    them, at a quantised tier and at FP16: every query's output is the
    same to the bit, with and without the key bias, and for a decode
    step's per-page mass too, which gives the pages none. The masked
    cache's output is near attention over the keys without the pages'
    positions, and far from the cache's with nothing masked."""
    heads, kv_heads, hd = CFG.num_attention_heads, CFG.num_key_value_heads, CFG.head_dim

    def evicted(page_tokens):
        seq = 5 * page_tokens + 3
        (held, evicting), k, v, start = stored(page_tokens, seq)
        whole = attend(CFG, held[1], 0, normal(np.random.default_rng(1), seq - start, heads,
                                                 hd), k[0], v[0], None, start)
        evicting[1].downgrade(0, 2, Tier.INT4)
        held[1].downgrade(0, 2, Tier.INT4)
        for page in (0, 2, 3):
            evicting[1].evict(0, page)
            held[1].mask(0, page)
        evicting[1].evict(1, 1)
        held[1].mask(1, 1)
        rng = np.random.default_rng(2)
        for layer in range(LAYERS):
            for bias in (None, normal(rng, kv_heads, hd) * 30):
                q = normal(rng, seq - start, heads, hd)
                got = attend(CFG, evicting[1], layer, q, k[layer], v[layer], bias, start)
                masked = attend(CFG, held[1], layer, q, k[layer], v[layer], bias, start)
                np.testing.assert_array_equal(got, masked, err_msg=f"layer {layer}")

        # Masked is masked: the queries read the keys whose pages are read.
        q = normal(np.random.default_rng(1), seq - start, heads, hd)
        masked = attend(CFG, held[1], 0, q, k[0], v[0], None, start)
        assert not np.allclose(masked, whole, atol=1e-2)
        kept = np.ones(seq, bool)
        for page in (0, 2, 3):
            kept[page * page_tokens:(page + 1) * page_tokens] = False
        reference = without(held, k[0], v[0], q, kept, start, page_tokens)
        np.testing.assert_allclose(masked, reference, atol=2e-3)

        # A decode step's per-page mass gives an unread page none.
        one = normal(rng, 1, heads, hd)
        q_row, k_row, v_row = one, normal(rng, 1, kv_heads, hd), normal(rng, 1, kv_heads, hd)
        masses = []
        for _, cache in (evicting, held):
            cache.reserve(seq + 1)
            cache.store(0, put(k_row), put(v_row), seq, 1)
            mass = device.empty_f32(heads * (seq // page_tokens + 1))
            out = device.empty(heads * hd)
            rope = device.RopeTable(hd, CFG.rope_theta)
            rope.cover(seq + 1)
            cache.attention(0, put(q_row), None, rope, out, 1, seq + 1, heads, kv_heads, hd,
                            put(k_row), put(v_row), mass)
            masses.append((out.to_numpy(), mass.to_numpy().reshape(heads, -1)))
        np.testing.assert_array_equal(masses[0][0], masses[1][0])
        np.testing.assert_array_equal(masses[0][1], masses[1][1])
        assert (masses[0][1][:, [0, 2, 3]] == 0).all()
        np.testing.assert_allclose(masses[0][1].sum(axis=1), 1.0, rtol=1e-5)

    for page_tokens in PAGE_TOKENS:
        evicted(page_tokens)


def without(pair, k, v, q, kept, start, page_tokens):
    """Each query alone over the keys it may read, those of the pages before
    its own that `kept` marks, and its own page's, from a cache holding
    them all: the masked cache's own reference. A query's own page is
    always read."""
    allocator, _ = pair
    heads, kv_heads, hd = CFG.num_attention_heads, CFG.num_key_value_heads, CFG.head_dim
    sealed_k, sealed_v = sealed(allocator, CFG, 0, len(k) // page_tokens, page_tokens)
    rope = device.RopeTable(hd, CFG.rope_theta)
    rope.cover(len(k))
    rows = []
    for i, pos in enumerate(range(start, len(k))):
        own = pos // page_tokens * page_tokens
        read = kept[:pos + 1] | (np.arange(pos + 1) >= own)
        kk = np.concatenate([sealed_k[:own], k[own:pos + 1]])[read]
        vv = np.concatenate([sealed_v[:own], v[own:pos + 1]])[read]
        out = device.empty(heads * hd)
        device.attention(put(q[i:i + 1]), put(kk), put(vv), None, rope, out, 1, len(kk),
                         heads, kv_heads, hd)
        rows.append(out.to_numpy())
    return np.concatenate(rows)


def test_an_evicted_pages_memory_returns_to_the_driver():
    """Evicting pages that fill granules of an FP16 range gives the
    granules back: the range maps all but what the pages left need, the
    evicted pages are no longer in the page table, and the rest hold what
    they held."""
    page_tokens = PAGE_TOKENS[0]
    seq = 640 * page_tokens
    (pair, _), k, v, _ = stored(page_tokens, seq)
    allocator, cache = pair
    before = allocator.mapped_bytes(Tier.FP16)
    held = {key: allocator.read(*key) for key in allocator.pages(Tier.FP16)}
    gone = [(layer, page) for layer in range(LAYERS) for page in range(400)]
    for layer, page in gone:
        cache.evict(layer, page)
    granule = allocator.granule_bytes
    left = len(allocator.pages(Tier.FP16)) * allocator.page_bytes(Tier.FP16)
    assert allocator.mapped_bytes(Tier.FP16) == -(-left // granule) * granule < before
    assert not set(gone) & set(allocator.pages(Tier.FP16))
    for key in allocator.pages(Tier.FP16):
        np.testing.assert_array_equal(allocator.read(*key), held[key])
    assert cache.evicted()[:, :400].all() and not cache.evicted()[:, 400:].any()
    tiers = cache.page_tiers()
    assert (tiers[:, :400] == 4).all() and (tiers[:, 400:] == int(Tier.FP16)).all()


def test_what_an_eviction_refuses():
    """A page not sealed, one evicted already, or masked, is not evicted;
    an evicted page neither moves nor is masked; and a cache that does not
    seal evicts nothing."""
    page_tokens = PAGE_TOKENS[0]
    seq = 3 * page_tokens + 5
    ((_, cache), (_, twin)), _, _, _ = stored(page_tokens, seq)
    cache.evict(0, 1)
    twin.mask(0, 1)
    for call in (lambda: cache.evict(0, 3), lambda: cache.evict(0, 1),
                 lambda: cache.mask(0, 1), lambda: cache.downgrade(0, 1, Tier.INT4),
                 lambda: cache.upgrade(0, 1, Tier.FP16), lambda: twin.evict(0, 1),
                 lambda: cache.evict(LAYERS, 0)):
        with pytest.raises((ValueError, IndexError)):
            call()
    with pytest.raises(Exception, match="not in the page table"):
        cache.page_tier(0, 1)
    twin.downgrade(0, 1, Tier.INT4)  # a masked page still moves
    capacity = [0] * 4
    capacity[int(Tier.FP16)] = 64
    allocator = PagedKVCache(tier_page_bytes(CFG, page_tokens), capacity)
    plain = device.KVPages(allocator, LAYERS, page_tokens, CFG.num_key_value_heads,
                           CFG.head_dim, Tier.FP16)
    plain.reserve(2 * page_tokens)
    with pytest.raises(ValueError, match="seals"):
        plain.evict(0, 0)
