"""The KV cache on pages at a quantised tier (#18, Seam B).

A page at INT8, INT4 or INT2 is allocated at that tier and written once, when
its last position arrives; until then its positions are in the layer's FP16
open page (ADR-0005, kv_pages.h). No page changes tier.

As in test_kv_pages.py the proof is equality, not tolerance, in two steps:

- what the cache stores for a full page is, byte for byte, what quantise_page
  gives for that page's positions, however the positions arrived: in one
  store, one at a time, or in runs that end mid-page;
- attention over the cache is, to the bit, attention over contiguous keys and
  values holding what the cache holds: each full page as dequantise_page
  returns it, and the open page's rows as they are.

So everything quantisation loses was measured already, by test_quant.py and
tools/quant_roundtrip.py, and nothing here may lose more. P runs at 16 and
48, as in test_kv_pages.py.
"""

import numpy as np
import pytest
from test_kv_pages import CFG, LAYERS, MODELS, PAGE_TOKENS, attend_both, normal, put

from microinfer import _microinfer
from microinfer._microinfer import PagedKVCache, Tier
from microinfer.config import ModelConfig

device = _microinfer.device
QUANTISED = (Tier.INT8, Tier.INT4, Tier.INT2)


@pytest.fixture(params=QUANTISED, ids=lambda t: t.name)
def tier(request):
    return request.param


def pages_for(cfg, page_tokens, tier, capacity_pages=4096):
    """A cache at `tier`, and an allocator with a range for that tier's pages
    and one for the open pages at FP16."""
    kv_heads, hd = cfg.num_key_value_heads, cfg.head_dim
    page_bytes = [device.page_bytes(page_tokens, kv_heads * hd)] + [
        _microinfer.quantised_page_layout(t, kv_heads, hd, page_tokens)["page_bytes"]
        for t in QUANTISED]
    capacity = [0] * 4
    capacity[int(tier)] = capacity_pages
    capacity[int(Tier.FP16)] = LAYERS
    allocator = PagedKVCache(page_bytes, capacity)
    return allocator, device.KVPages(allocator, LAYERS, page_tokens, kv_heads, hd, tier)


def held(allocator, cfg, layer, seq, page_tokens, tier):
    """What the cache holds for positions [0, seq) of `layer`, as rows: each
    full page as dequantise_page returns it, then the open page's rows."""
    kv_heads, hd = cfg.num_key_value_heads, cfg.head_dim
    keys, values = [], []
    for i in range(seq // page_tokens):
        k, v = _microinfer.dequantise_page(allocator.read(layer, i), tier, kv_heads, hd,
                                           page_tokens)
        keys.append(k)
        values.append(v)
    if tail := seq % page_tokens:
        open_page = allocator.read(layer, device.open_page).view(np.float16).astype(np.float32)
        open_page = open_page.reshape(2, page_tokens, kv_heads, hd)
        keys.append(open_page[0, :tail])
        values.append(open_page[1, :tail])
    return np.concatenate(keys), np.concatenate(values)


def store_in_runs(cache, layer, k, v, runs):
    """Stores k and v in consecutive runs of the given lengths, reserving as
    the engine does, before each."""
    start = 0
    for n in runs:
        cache.reserve(start + n)
        cache.store(layer, put(k[start:start + n]), put(v[start:start + n]), start, n)
        start += n
    assert start == len(k)


# -- what a page holds ------------------------------------------------------------


@pytest.mark.parametrize("page_tokens", PAGE_TOKENS)
def test_a_full_page_holds_quantise_page_of_its_positions(tier, page_tokens):
    """Runs that cover a page whole, that end mid-page, that finish a page the
    open page began, and single positions: each full page is quantise_page's
    bytes for its positions, and the open page holds the rest as given."""
    rng = np.random.default_rng(page_tokens)
    kv_heads, hd = CFG.num_key_value_heads, CFG.head_dim
    seq = 4 * page_tokens + 5
    k, v = normal(rng, seq, kv_heads, hd), normal(rng, seq, kv_heads, hd)
    runs = [7, page_tokens, 1, 1, 2 * page_tokens - 9]
    runs.append(seq - sum(runs))
    allocator, cache = pages_for(CFG, page_tokens, tier)
    store_in_runs(cache, 1, k, v, runs)

    for i in range(seq // page_tokens):
        span = slice(i * page_tokens, (i + 1) * page_tokens)
        np.testing.assert_array_equal(
            allocator.read(1, i), _microinfer.quantise_page(k[span], v[span], tier, page_tokens),
            err_msg=f"page {i}")
    got_k, got_v = held(allocator, CFG, 1, seq, page_tokens, tier)
    tail = slice(seq // page_tokens * page_tokens, seq)
    np.testing.assert_array_equal(got_k[tail], k[tail])
    np.testing.assert_array_equal(got_v[tail], v[tail])


def test_pages_are_taken_at_the_tier_and_never_change_it(tier):
    """Every allocation the cache makes, followed position by position: a page
    of positions appears at the cache's tier once its last position is
    reserved, the open pages appear at FP16 once, and no page, once seen, is
    ever at another tier or gone."""
    page_tokens = PAGE_TOKENS[0]
    rng = np.random.default_rng(1)
    kv_heads, hd = CFG.num_key_value_heads, CFG.head_dim
    seq = 3 * page_tokens + 4
    k, v = normal(rng, seq, kv_heads, hd), normal(rng, seq, kv_heads, hd)
    allocator, cache = pages_for(CFG, page_tokens, tier)

    seen = {}
    for t in range(seq):
        cache.reserve(t + 1)
        for layer in range(LAYERS):
            cache.store(layer, put(k[t]), put(v[t]), t, 1)
        now = {key: t_ for t_ in Tier.__members__.values() for key in allocator.pages(t_)}
        for key, was in seen.items():
            assert now.get(key) == was, f"{key} was at {was}, now {now.get(key)}"
        seen.update(now)
        full = (t + 1) // page_tokens
        assert sorted(allocator.pages(tier)) == [(layer, i) for layer in range(LAYERS)
                                                 for i in range(full)]
        assert sorted(allocator.pages(Tier.FP16)) == [(layer, device.open_page)
                                                      for layer in range(LAYERS)]
    assert cache.pages_per_layer == seq // page_tokens and cache.tier == tier


# -- attention over it --------------------------------------------------------------


@pytest.mark.parametrize("name", MODELS)
@pytest.mark.parametrize("page_tokens", PAGE_TOKENS)
def test_attention_is_attention_over_what_the_pages_hold(name, tier, page_tokens):
    """Prefill and decode, with and without the key bias, below a page, at one,
    one past it, and several with a ragged last."""
    cfg = ModelConfig.from_card(name)
    rng = np.random.default_rng(page_tokens)
    heads, kv_heads, hd = cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim
    for seq_k in (1, page_tokens - 1, page_tokens, page_tokens + 1, 3 * page_tokens + 5):
        allocator, cache = pages_for(cfg, page_tokens, tier)
        k, v = normal(rng, seq_k, kv_heads, hd), normal(rng, seq_k, kv_heads, hd)
        cache.reserve(seq_k)
        cache.store(1, put(k), put(v), 0, seq_k)
        want_k, want_v = held(allocator, cfg, 1, seq_k, page_tokens, tier)
        for bias in (None, normal(rng, kv_heads, hd) * 30):
            for seq_q in {seq_k, 1}:
                q = normal(rng, seq_q, heads, hd)
                paged, whole = attend_both(cfg, cache, 1, q, want_k, want_v, bias, seq_k)
                np.testing.assert_array_equal(paged, whole,
                                              err_msg=f"{tier.name} P={page_tokens} seq_k={seq_k}")


def test_a_quantised_page_moved_between_launches_is_still_read_correctly(tier):
    """As test_kv_pages.py's, at a quantised tier: the allocator moves one of
    the cache's pages into a slot another page left, and that slot is then
    filled with another page's garbage, so a stale address cannot pass."""
    page_tokens = PAGE_TOKENS[0]
    rng = np.random.default_rng(3)
    heads, kv_heads, hd = CFG.num_attention_heads, CFG.num_key_value_heads, CFG.head_dim
    allocator, cache = pages_for(CFG, page_tokens, tier)
    seq = 3 * page_tokens + 2
    k, v = normal(rng, seq, kv_heads, hd), normal(rng, seq, kv_heads, hd)
    q = normal(rng, 1, heads, hd)

    cache.reserve(page_tokens)
    allocator.allocate(99, 0, tier)  # someone else's page, mid-range
    cache.reserve(seq)
    cache.store(0, put(k), put(v), 0, seq)
    want_k, want_v = held(allocator, CFG, 0, seq, page_tokens, tier)
    before, _ = attend_both(CFG, cache, 0, q, want_k, want_v, None, seq)

    tail = allocator.pages(tier)[-1]
    slot_before = allocator.locate(*tail)[1]
    allocator.free(99, 0)
    assert allocator.locate(*tail)[1] < slot_before, "no page of the cache moved"
    allocator.allocate(98, 0, tier)
    assert allocator.locate(98, 0)[1] == slot_before
    allocator.write(98, 0, np.full(cache.page_bytes, 0xFF, np.uint8))

    after, whole = attend_both(CFG, cache, 0, q, want_k, want_v, None, seq)
    np.testing.assert_array_equal(after, whole)
    np.testing.assert_array_equal(after, before)


def test_the_cache_returns_every_page_when_it_goes(tier):
    allocator, cache = pages_for(CFG, PAGE_TOKENS[0], tier)
    cache.reserve(100)
    assert allocator.pages(tier) and allocator.pages(Tier.FP16)
    del cache
    for t in Tier.__members__.values():
        assert allocator.pages(t) == [] and allocator.mapped_bytes(t) == 0


def test_a_quantised_cache_needs_room_for_its_open_pages(tier):
    kv_heads, hd = CFG.num_key_value_heads, CFG.head_dim
    page_bytes = [device.page_bytes(16, kv_heads * hd)] + [
        _microinfer.quantised_page_layout(t, kv_heads, hd, 16)["page_bytes"] for t in QUANTISED]
    wrong = list(page_bytes)
    wrong[int(Tier.FP16)] += 2
    with pytest.raises(ValueError, match="open page"):
        device.KVPages(PagedKVCache(wrong, [8] * 4), LAYERS, 16, kv_heads, hd, tier)
