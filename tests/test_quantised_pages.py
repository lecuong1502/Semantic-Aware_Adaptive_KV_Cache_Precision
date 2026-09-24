"""The KV cache on pages at a quantised tier (#18, ADR-0011, Seam B).

A page at INT8, INT4 or INT2 is allocated at that tier and sealed once, when
its last position arrives; until then its positions are in one of the
layer's two FP16 open pages. No page changes tier.

Attention is causal as decode is: a query reads every page before its own as
sealed, and its own page at FP16, however many positions arrived with it.
The proof is equality, not tolerance, in two steps:

- what the cache holds for a full page is, byte for byte, what quantise_page
  gives for that page's positions, however the positions arrived: in one
  store, one at a time, or in runs that begin and end mid-page;
- each query's output is, to the bit, attention over contiguous keys and
  values holding the pages before its own as dequantise_page returns them,
  and its own page's positions as they came. Attention's output for one
  query depends on no other query, so a query run alone is the reference
  for the same query run among many.

So everything quantisation loses was measured already, by test_quant.py and
tools/quant_roundtrip.py, and nothing here may lose more, or less. P runs at
16 and 48, as in test_kv_pages.py.
"""

import numpy as np
import pytest
from test_kv_pages import CFG, LAYERS, MODELS, PAGE_TOKENS, normal, put

from microinfer import _microinfer
from microinfer._microinfer import PagedKVCache, Tier
from microinfer.config import ModelConfig
from microinfer.model import tier_page_bytes

device = _microinfer.device
Halves = device.Halves
QUANTISED = (Tier.INT8, Tier.INT4, Tier.INT2)


@pytest.fixture(params=QUANTISED, ids=lambda t: t.name)
def tier(request):
    return request.param


def pages_for(cfg, page_tokens, tier, halves=Halves.Both, capacity_pages=4096):
    """A cache at `tier`, and an allocator with a range for its pages of
    positions and one for its open pages at FP16."""
    storage = tier if halves == Halves.Both else Tier.FP16
    capacity = [0] * 4
    capacity[int(storage)] = capacity_pages
    capacity[int(Tier.FP16)] += LAYERS * len(device.open_pages)
    allocator = PagedKVCache(tier_page_bytes(cfg, page_tokens), capacity)
    return allocator, device.KVPages(allocator, LAYERS, page_tokens, cfg.num_key_value_heads,
                                     cfg.head_dim, tier, halves)


def sealed(allocator, cfg, layer, pages, page_tokens, tier, halves=Halves.Both):
    """The first `pages` pages of `layer`, as rows: dequantised as
    dequantise_page returns them, or, for a diagnostic, as the FP16 page
    holds them."""
    kv_heads, hd = cfg.num_key_value_heads, cfg.head_dim
    keys, values = [], []
    for i in range(pages):
        raw = allocator.read(layer, i)
        if halves == Halves.Both:
            k, v = _microinfer.dequantise_page(raw, tier, kv_heads, hd, page_tokens)
        else:
            k, v = raw.view(np.float16).astype(np.float32).reshape(2, page_tokens, kv_heads, hd)
        keys.append(k)
        values.append(v)
    empty = np.zeros((0, kv_heads, hd), np.float32)
    return np.concatenate(keys or [empty]), np.concatenate(values or [empty])


def store_in_runs(cache, layer, k, v, runs):
    """Stores k and v in consecutive runs of the given lengths, reserving as
    the engine does, before each. Returns where the last run began."""
    start = 0
    for n in runs:
        cache.reserve(start + n)
        cache.store(layer, put(k[start:start + n]), put(v[start:start + n]), start, n)
        start += n
    assert start == len(k)
    return start - runs[-1]


def causal_reference(cfg, allocator, layer, q, k, v, bias, start, page_tokens, tier,
                     halves=Halves.Both):
    """Each query alone, over what it may read: the pages before its own as
    sealed, and its own page's positions as they came."""
    heads, kv_heads, hd = cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim
    seq_k = len(k)
    rope = device.RopeTable(hd, cfg.rope_theta)
    rope.cover(seq_k)
    b = put(bias) if bias is not None else None
    sealed_k, sealed_v = sealed(allocator, cfg, layer, seq_k // page_tokens, page_tokens,
                                tier, halves)
    rows = []
    for i, pos in enumerate(range(start, seq_k)):
        own = pos // page_tokens * page_tokens
        kk = np.concatenate([sealed_k[:own], k[own:pos + 1]])
        vv = np.concatenate([sealed_v[:own], v[own:pos + 1]])
        out = device.empty(heads * hd)
        device.attention(put(q[i:i + 1]), put(kk), put(vv), b, rope, out, 1, pos + 1,
                         heads, kv_heads, hd)
        rows.append(out.to_numpy())
    return np.concatenate(rows)


def attend(cfg, cache, layer, q, k, v, bias, start):
    """Attention through the cache for the queries at [start, len(k)), which
    are the rows the last store brought."""
    heads, kv_heads, hd = cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim
    seq_k = len(k)
    rope = device.RopeTable(hd, cfg.rope_theta)
    rope.cover(seq_k)
    out = device.empty(q.size)
    cache.attention(layer, put(q), put(bias) if bias is not None else None, rope, out,
                    seq_k - start, seq_k, heads, kv_heads, hd, put(k[start:]), put(v[start:]))
    return out.to_numpy()


# -- what a page holds ------------------------------------------------------------


@pytest.mark.parametrize("page_tokens", PAGE_TOKENS)
def test_a_full_page_holds_quantise_page_of_its_positions(tier, page_tokens):
    """Runs that cover a page whole, that end mid-page, that finish a page an
    open page began, and single positions: each full page is quantise_page's
    bytes for its positions."""
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


def test_pages_are_taken_at_the_tier_and_never_change_it(tier):
    """Every allocation the cache makes, followed position by position: a page
    of positions appears at the cache's tier once its last position is
    reserved, the two open pages appear at FP16 once, and no page, once seen,
    is ever at another tier or gone."""
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
        assert sorted(allocator.pages(Tier.FP16)) == sorted(
            (layer, p) for layer in range(LAYERS) for p in device.open_pages)
    assert cache.pages_per_layer == seq // page_tokens and cache.tier == tier


# -- attention over it --------------------------------------------------------------


@pytest.mark.parametrize("name", MODELS)
@pytest.mark.parametrize("page_tokens", PAGE_TOKENS)
def test_each_query_reads_earlier_pages_sealed_and_its_own_as_it_came(name, tier, page_tokens):
    """A prompt in one step, then more positions in a step that begins
    mid-page and crosses pages, then single positions: after each, every
    query's output is the causal reference's, with and without the key bias."""
    cfg = ModelConfig.from_card(name)
    rng = np.random.default_rng(page_tokens)
    heads, kv_heads, hd = cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim
    seq = 3 * page_tokens + 7
    k, v = normal(rng, seq, kv_heads, hd), normal(rng, seq, kv_heads, hd)
    for runs in ([seq], [page_tokens + 5, seq - page_tokens - 5],
                 [page_tokens - 1, 1, 2 * page_tokens + 4, 1, 1, 1]):
        allocator, cache = pages_for(cfg, page_tokens, tier)
        start = store_in_runs(cache, 1, k, v, runs)
        for bias in (None, normal(rng, kv_heads, hd) * 30):
            q = normal(rng, seq - start, heads, hd)
            want = causal_reference(cfg, allocator, 1, q, k, v, bias, start, page_tokens, tier)
            np.testing.assert_array_equal(attend(cfg, cache, 1, q, k, v, bias, start), want,
                                          err_msg=f"{tier.name} P={page_tokens} runs={runs}")


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
    start = store_in_runs(cache, 0, k, v, [seq - 1, 1])
    before = attend(CFG, cache, 0, q, k, v, None, start)

    tail = allocator.pages(tier)[-1]
    slot_before = allocator.locate(*tail)[1]
    allocator.free(99, 0)
    assert allocator.locate(*tail)[1] < slot_before, "no page of the cache moved"
    allocator.allocate(98, 0, tier)
    assert allocator.locate(98, 0)[1] == slot_before
    allocator.write(98, 0, np.full(cache.page_bytes, 0xFF, np.uint8))

    after = attend(CFG, cache, 0, q, k, v, None, start)
    np.testing.assert_array_equal(after, before)
    np.testing.assert_array_equal(
        after, causal_reference(CFG, allocator, 0, q, k, v, None, start, page_tokens, tier))


# -- the diagnostic: one half at a time ---------------------------------------------


@pytest.mark.parametrize("halves", [Halves.Keys, Halves.Values], ids=lambda h: h.name)
def test_a_diagnostic_page_rounds_one_half_and_keeps_the_other(tier, halves):
    """Stored at FP16, with the named half replaced by the tier's round trip
    and the other as it came; and attention reads it causally all the same."""
    page_tokens = PAGE_TOKENS[0]
    rng = np.random.default_rng(4)
    heads, kv_heads, hd = CFG.num_attention_heads, CFG.num_key_value_heads, CFG.head_dim
    seq = 2 * page_tokens + 3
    k, v = normal(rng, seq, kv_heads, hd), normal(rng, seq, kv_heads, hd)
    allocator, cache = pages_for(CFG, page_tokens, tier, halves)
    assert cache.storage_tier == Tier.FP16 and cache.halves == halves
    start = store_in_runs(cache, 0, k, v, [page_tokens + 2, seq - page_tokens - 2])

    got_k, got_v = sealed(allocator, CFG, 0, 2, page_tokens, tier, halves)
    for i in range(2):
        span = slice(i * page_tokens, (i + 1) * page_tokens)
        rk, rv = _microinfer.dequantise_page(
            _microinfer.quantise_page(k[span], v[span], tier, page_tokens), tier, kv_heads, hd,
            page_tokens)
        np.testing.assert_array_equal(got_k[span], rk if halves == Halves.Keys else k[span])
        np.testing.assert_array_equal(got_v[span], rv if halves == Halves.Values else v[span])
    q = normal(rng, seq - start, heads, hd)
    np.testing.assert_array_equal(
        attend(CFG, cache, 0, q, k, v, None, start),
        causal_reference(CFG, allocator, 0, q, k, v, None, start, page_tokens, tier, halves))


# -- the rest of the contract ----------------------------------------------------------


def test_the_cache_returns_every_page_when_it_goes(tier):
    allocator, cache = pages_for(CFG, PAGE_TOKENS[0], tier)
    cache.reserve(100)
    assert allocator.pages(tier) and allocator.pages(Tier.FP16)
    del cache
    for t in Tier.__members__.values():
        assert allocator.pages(t) == [] and allocator.mapped_bytes(t) == 0


def test_a_failed_reserve_gives_back_the_open_pages_it_took(tier):
    """The first reserve takes the open pages and then the pages of positions;
    if the second part fails, the first is given back too."""
    page_tokens = PAGE_TOKENS[0]
    allocator, cache = pages_for(CFG, page_tokens, tier, capacity_pages=LAYERS)
    with pytest.raises(ValueError, match="full"):
        cache.reserve(2 * page_tokens)
    assert all(allocator.pages(t) == [] for t in Tier.__members__.values())
    cache.reserve(page_tokens)
    assert cache.pages_per_layer == 1


def test_a_quantised_cache_needs_room_for_its_open_pages(tier):
    wrong = tier_page_bytes(CFG, 16)
    wrong[int(Tier.FP16)] += 2
    with pytest.raises(ValueError, match="open page"):
        device.KVPages(PagedKVCache(wrong, [8] * 4), LAYERS, 16, CFG.num_key_value_heads,
                       CFG.head_dim, tier)


def test_attention_at_a_quantised_tier_needs_the_queries_rows(tier):
    _, cache = pages_for(CFG, PAGE_TOKENS[0], tier)
    cache.reserve(4)
    heads, kv_heads, hd = CFG.num_attention_heads, CFG.num_key_value_heads, CFG.head_dim
    with pytest.raises(ValueError, match="rows"):
        cache.attention(0, device.empty(heads * hd), None, None, device.empty(heads * hd),
                        1, 4, heads, kv_heads, hd)
