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
from conftest import each
from test_kv_pages import CFG, LAYERS, MODELS, PAGE_TOKENS, normal, put

from microinfer import _microinfer
from microinfer._microinfer import PagedKVCache, Tier
from microinfer.config import ModelConfig
from microinfer.model import tier_page_bytes

device = _microinfer.device
Halves = device.Halves
QUANTISED = (Tier.INT8, Tier.INT4, Tier.INT2)


def pages_for(cfg, page_tokens, tier, halves=Halves.Both, capacity_pages=4096, tier_map=None):
    """A cache at `tier`, or built from `tier_map`, and an allocator with a
    range for its pages of positions, at every tier for a map, and one for
    its open pages at FP16."""
    storage = tier if halves == Halves.Both else Tier.FP16
    capacity = [0] * 4
    for t in (Tier.__members__.values() if tier_map is not None else (storage,)):
        capacity[int(t)] = capacity_pages
    capacity[int(Tier.FP16)] += LAYERS * len(device.open_pages)
    allocator = PagedKVCache(tier_page_bytes(cfg, page_tokens), capacity)
    return allocator, device.KVPages(allocator, LAYERS, page_tokens, cfg.num_key_value_heads,
                                     cfg.head_dim, tier, halves, tier_map or [])


def sealed(allocator, cfg, layer, pages, page_tokens):
    """The first `pages` pages of `layer`, as rows, each read at the tier the
    page table records for it: dequantised as dequantise_page returns it, or,
    at FP16 (a diagnostic's pages, or a page a map placed there), as the page
    holds it."""
    kv_heads, hd = cfg.num_key_value_heads, cfg.head_dim
    keys, values = [], []
    for i in range(pages):
        raw = allocator.read(layer, i)
        tier = allocator.locate(layer, i)[0]
        if tier == Tier.FP16:
            k, v = raw.view(np.float16).astype(np.float32).reshape(2, page_tokens, kv_heads, hd)
        else:
            k, v = _microinfer.dequantise_page(raw, tier, kv_heads, hd, page_tokens)
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


def causal_reference(cfg, allocator, layer, q, k, v, bias, start, page_tokens):
    """Each query alone, over what it may read: the pages before its own as
    sealed, each at its own tier, and its own page's positions as they
    came."""
    heads, kv_heads, hd = cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim
    seq_k = len(k)
    rope = device.RopeTable(hd, cfg.rope_theta)
    rope.cover(seq_k)
    b = put(bias) if bias is not None else None
    sealed_k, sealed_v = sealed(allocator, cfg, layer, seq_k // page_tokens, page_tokens)
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


def test_a_page_holds_quantise_page_of_its_positions_at_its_tier_for_good():
    """At every tier and both page sizes, runs that cover a page whole, that
    end mid-page, that finish a page an open page began, and single positions:
    each full page is quantise_page's bytes for its positions. And every
    allocation the cache makes, followed position by position: a page of
    positions appears at the cache's tier once its last position is reserved,
    the two open pages appear at FP16 once, and no page, once seen, is ever
    at another tier or gone."""
    def holds(tier, page_tokens):
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
                allocator.read(1, i),
                _microinfer.quantise_page(k[span], v[span], tier, page_tokens),
                err_msg=f"page {i}")

    def never_changes(tier, page_tokens):
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

    each([(holds, t, p) for t in QUANTISED for p in PAGE_TOKENS]
         + [(never_changes, t, PAGE_TOKENS[0]) for t in QUANTISED],
         lambda f, t, p: f(t, p), name=lambda c: f"{c[0].__name__} {c[1].name} P={c[2]}")


# -- attention over it --------------------------------------------------------------


def test_each_query_reads_earlier_pages_sealed_and_its_own_as_it_came():
    """At every tier, both models and both page sizes: a prompt in one step,
    then more positions in a step that begins mid-page and crosses pages,
    then single positions; after each, every query's output is the causal
    reference's, with and without the key bias. And as test_kv_pages.py's,
    at a quantised tier: the allocator moves one of the cache's pages into a
    slot another page left, and that slot is then filled with another page's
    garbage, so a stale address cannot pass."""
    def causal(name, tier, page_tokens):
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
                want = causal_reference(cfg, allocator, 1, q, k, v, bias, start, page_tokens)
                np.testing.assert_array_equal(attend(cfg, cache, 1, q, k, v, bias, start), want,
                                              err_msg=f"runs={runs}")

    def moved(tier):
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
            after, causal_reference(CFG, allocator, 0, q, k, v, None, start, page_tokens))

    each([(n, t, p) for n in MODELS for t in QUANTISED for p in PAGE_TOKENS], causal,
         name=lambda c: f"{c[0]} {c[1].name} P={c[2]}")
    each(QUANTISED, moved, name=lambda t: f"moved {t.name}")


# -- the diagnostic: one half at a time ---------------------------------------------


def test_a_diagnostic_page_rounds_one_half_and_keeps_the_other():
    """Stored at FP16, with the named half replaced by the tier's round trip
    and the other as it came; and attention reads it causally all the same."""
    def diagnostic(tier, halves):
        page_tokens = PAGE_TOKENS[0]
        rng = np.random.default_rng(4)
        heads, kv_heads, hd = CFG.num_attention_heads, CFG.num_key_value_heads, CFG.head_dim
        seq = 2 * page_tokens + 3
        k, v = normal(rng, seq, kv_heads, hd), normal(rng, seq, kv_heads, hd)
        allocator, cache = pages_for(CFG, page_tokens, tier, halves)
        assert cache.storage_tier == Tier.FP16 and cache.halves == halves
        start = store_in_runs(cache, 0, k, v, [page_tokens + 2, seq - page_tokens - 2])
        got_k, got_v = sealed(allocator, CFG, 0, 2, page_tokens)
        for i in range(2):
            span = slice(i * page_tokens, (i + 1) * page_tokens)
            rk, rv = _microinfer.dequantise_page(
                _microinfer.quantise_page(k[span], v[span], tier, page_tokens), tier, kv_heads,
                hd, page_tokens)
            np.testing.assert_array_equal(got_k[span], rk if halves == Halves.Keys else k[span])
            np.testing.assert_array_equal(got_v[span], rv if halves == Halves.Values else v[span])
        q = normal(rng, seq - start, heads, hd)
        np.testing.assert_array_equal(
            attend(CFG, cache, 0, q, k, v, None, start),
            causal_reference(CFG, allocator, 0, q, k, v, None, start, page_tokens))

    each([(t, h) for t in QUANTISED for h in (Halves.Keys, Halves.Values)], diagnostic,
         name=lambda c: f"{c[0].name} {c[1].name}")


# -- the rest of the contract ----------------------------------------------------------


def test_the_rest_of_the_contract():
    """At every tier: the cache returns every page when it goes; a failed
    first reserve, which takes the open pages and then the pages of
    positions, gives the open pages back too; a quantised cache needs room
    for its open pages; attention at a quantised tier needs the queries'
    rows."""
    heads, kv_heads, hd = CFG.num_attention_heads, CFG.num_key_value_heads, CFG.head_dim
    for tier in QUANTISED:
        allocator, cache = pages_for(CFG, PAGE_TOKENS[0], tier)
        cache.reserve(100)
        assert allocator.pages(tier) and allocator.pages(Tier.FP16)
        del cache
        for t in Tier.__members__.values():
            assert allocator.pages(t) == [] and allocator.mapped_bytes(t) == 0

        page_tokens = PAGE_TOKENS[0]
        allocator, cache = pages_for(CFG, page_tokens, tier, capacity_pages=LAYERS)
        with pytest.raises(ValueError, match="full"):
            cache.reserve(2 * page_tokens)
        assert all(allocator.pages(t) == [] for t in Tier.__members__.values())
        cache.reserve(page_tokens)
        assert cache.pages_per_layer == 1

        wrong = tier_page_bytes(CFG, 16)
        wrong[int(Tier.FP16)] += 2
        with pytest.raises(ValueError, match="open page"):
            device.KVPages(PagedKVCache(wrong, [8] * 4), LAYERS, 16, kv_heads, hd, tier)

        _, cache = pages_for(CFG, PAGE_TOKENS[0], tier)
        cache.reserve(4)
        with pytest.raises(ValueError, match="rows"):
            cache.attention(0, device.empty(heads * hd), None, None, device.empty(heads * hd),
                            1, 4, heads, kv_heads, hd)


def test_each_page_records_its_own_tier_in_the_page_table():
    """A page's tier is the page table's record of it, not the cache's: each
    page of positions at the tier it was stored at, the open pages at FP16,
    and the pages of a diagnostic at FP16. A page the table does not hold
    has no tier."""
    def records(page_tokens, tier, halves=Halves.Both):
        allocator, cache = pages_for(CFG, page_tokens, tier, halves)
        positions = 2 * page_tokens + 3  # two pages sealed, one being filled
        rng = np.random.default_rng(89)
        cache.reserve(positions)
        for layer in range(LAYERS):
            k, v = (normal(rng, positions, CFG.num_key_value_heads, CFG.head_dim)
                    for _ in range(2))
            cache.store(layer, put(k), put(v), 0, positions)
        stored = tier if halves == Halves.Both else Tier.FP16
        for layer in range(LAYERS):
            for page in range(cache.pages_per_layer):
                assert cache.page_tier(layer, page) == stored, (layer, page)
                assert allocator.locate(layer, page)[0] == stored, (layer, page)
            if tier != Tier.FP16:
                assert all(cache.page_tier(layer, p) == Tier.FP16 for p in device.open_pages)
        with pytest.raises(_microinfer.PageNotFound):
            cache.page_tier(0, cache.pages_per_layer + 5)

    each([(p, *case) for p in PAGE_TOKENS
          for case in ((Tier.FP16,), (Tier.INT8,), (Tier.INT4,), (Tier.INT2,),
                       (Tier.INT4, Halves.Keys))], records)


def test_a_cache_built_from_a_tier_map_seals_each_page_at_its_tier():
    """Page i of layer l is born and sealed at the map's tier for it, or at
    the cache's beyond the map: its bytes are quantise_page's at that tier,
    or, born at FP16, its rows as they came (quantise_page has no FP16
    layout; the FP16 page is the rows). A map with one tier everywhere builds
    what the static path builds, byte for byte, open pages included. A map that does not name every
    layer, and one with a diagnostic, are refused."""
    kv_heads, hd = CFG.num_key_value_heads, CFG.head_dim
    mixed = [[Tier.INT8, Tier.FP16, Tier.INT4, Tier.INT2], [Tier.FP16, Tier.INT2]]

    def stored(page_tokens, tier, tier_map):
        rng = np.random.default_rng(page_tokens)
        seq = 5 * page_tokens + 3
        k = [normal(rng, seq, kv_heads, hd) for _ in range(LAYERS)]
        v = [normal(rng, seq, kv_heads, hd) for _ in range(LAYERS)]
        allocator, cache = pages_for(CFG, page_tokens, tier, tier_map=tier_map)
        runs = [7, page_tokens, 1, 2 * page_tokens - 9]
        runs.append(seq - sum(runs))
        for layer in range(LAYERS):
            store_in_runs(cache, layer, k[layer], v[layer], runs)
        return allocator, cache, k, v

    def seals(page_tokens, tier):
        allocator, cache, k, v = stored(page_tokens, tier, mixed)
        assert cache.seals and not cache.born_at_one_tier
        for layer in range(LAYERS):
            for i in range(cache.pages_per_layer):
                want = mixed[layer][i] if i < len(mixed[layer]) else tier
                assert cache.birth_tier(layer, i) == want, (layer, i)
                assert cache.page_tier(layer, i) == want, (layer, i)
                span = slice(i * page_tokens, (i + 1) * page_tokens)
                expected = (np.concatenate([k[layer][span], v[layer][span]]).astype(np.float16)
                            .view(np.uint8).ravel() if want == Tier.FP16 else
                            _microinfer.quantise_page(k[layer][span], v[layer][span], want,
                                                      page_tokens))
                np.testing.assert_array_equal(allocator.read(layer, i), expected,
                                              err_msg=f"layer {layer} page {i} at {want.name}")

    def as_static(page_tokens, tier):
        uniform = [[tier] * 6 for _ in range(LAYERS)]
        mapped, cache, _, _ = stored(page_tokens, tier, uniform)
        static, kept, _, _ = stored(page_tokens, tier, None)  # kept: it frees its pages
        assert cache.born_at_one_tier
        for what in ("pages_per_layer", "capacity_tokens", "seals", "page_bytes"):
            assert getattr(cache, what) == getattr(kept, what), what
        open_pages = list(device.open_pages) if cache.seals else []
        for layer in range(LAYERS):
            for i in list(range(cache.pages_per_layer)) + open_pages:
                assert cache.page_tier(layer, i) == kept.page_tier(layer, i), (layer, i)
                np.testing.assert_array_equal(mapped.read(layer, i), static.read(layer, i),
                                              err_msg=f"layer {layer} page {i}")

    each([(f, p, t) for f in (seals, as_static) for p in PAGE_TOKENS
          for t in (Tier.FP16, Tier.INT4)],
         lambda f, p, t: f(p, t), name=lambda c: f"{c[0].__name__} P={c[1]} {c[2].name}")

    with pytest.raises(ValueError, match="layer"):
        pages_for(CFG, 16, Tier.FP16, tier_map=[[Tier.INT8]])
    with pytest.raises(ValueError, match="diagnostic"):
        pages_for(CFG, 16, Tier.INT4, Halves.Keys, tier_map=mixed)


def test_attention_reads_each_page_at_its_own_tier():
    """Over a cache built from a map of every tier, at both models and both
    page sizes, over caches at FP16 and at INT4: each query reads every page
    before its own at the tier the page table records for it, and its own
    page at FP16, as ADR-0011 has it. A prompt in one step, a step that
    begins mid-page and crosses pages, and single positions; after each,
    every query's output is, to the bit, the causal reference's, which
    dequantises each page at its own tier, with and without the key bias.
    And a map with one tier everywhere reads, to the bit, as the static
    cache at that tier does, at every tier."""
    def mixed_map(page_tokens, seq):
        pages = seq // page_tokens + 1
        order = [Tier.INT8, Tier.FP16, Tier.INT2, Tier.INT4]
        return [[order[(layer + i) % 4] for i in range(pages)] for layer in range(LAYERS)]

    def per_page(name, tier, page_tokens):
        cfg = ModelConfig.from_card(name)
        rng = np.random.default_rng(page_tokens)
        heads, kv_heads, hd = cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim
        seq = 4 * page_tokens + 7
        k, v = normal(rng, seq, kv_heads, hd), normal(rng, seq, kv_heads, hd)
        for runs in ([seq], [page_tokens + 5, seq - page_tokens - 5],
                     [page_tokens - 1, 1, 3 * page_tokens + 4, 1, 1, 1]):
            allocator, cache = pages_for(cfg, page_tokens, tier,
                                         tier_map=mixed_map(page_tokens, seq))
            start = store_in_runs(cache, 1, k, v, runs)
            assert {cache.page_tier(1, i) for i in range(cache.pages_per_layer)} == set(
                Tier.__members__.values())
            for bias in (None, normal(rng, kv_heads, hd) * 30):
                q = normal(rng, seq - start, heads, hd)
                np.testing.assert_array_equal(
                    attend(cfg, cache, 1, q, k, v, bias, start),
                    causal_reference(cfg, allocator, 1, q, k, v, bias, start, page_tokens),
                    err_msg=f"runs={runs}")

    def as_static(tier, page_tokens):
        rng = np.random.default_rng(91)
        heads, kv_heads, hd = CFG.num_attention_heads, CFG.num_key_value_heads, CFG.head_dim
        seq = 3 * page_tokens + 5
        k, v = normal(rng, seq, kv_heads, hd), normal(rng, seq, kv_heads, hd)
        bias = normal(rng, kv_heads, hd) * 30
        runs = [page_tokens + 3, seq - page_tokens - 3]
        outs = []
        for tier_map in ([[tier] * (seq // page_tokens + 1) for _ in range(LAYERS)], None):
            _, cache = pages_for(CFG, page_tokens, tier, tier_map=tier_map)
            start = store_in_runs(cache, 0, k, v, runs)
            q = normal(np.random.default_rng(0), seq - start, heads, hd)
            outs.append(attend(CFG, cache, 0, q, k, v, bias, start))
        np.testing.assert_array_equal(outs[0], outs[1])

    each([(n, t, p) for n in MODELS for t in (Tier.FP16, Tier.INT4) for p in PAGE_TOKENS],
         per_page, name=lambda c: f"{c[0]} {c[1].name} P={c[2]}")
    each([(t, p) for t in Tier.__members__.values() for p in PAGE_TOKENS], as_static,
         name=lambda c: f"as static {c[0].name} P={c[1]}")


# -- runtime tier changes (#93) -------------------------------------------------------


def snapshot(allocator):
    """Every page the allocator holds, by key: its bytes."""
    return {key: allocator.read(*key) for t in Tier.__members__.values()
            for key in allocator.pages(t)}


def test_a_sealed_fp16_page_downgrades_at_runtime():
    """A page of positions sealed at FP16 is moved to a lower tier mid-session:
    allocated at the target tier, quantised from the FP16 page, put in its
    place in the page table, and the FP16 page freed, its tier's tail moving
    into the slot (ADR-0007). Afterwards each moved page holds quantise_page's
    bytes for its positions at its target tier, every other page holds what
    it held, and attention reads, to the bit, what a cache built at those
    tiers reads. At both page sizes, over caches at FP16 and INT4."""
    kv_heads, hd = CFG.num_key_value_heads, CFG.head_dim
    heads = CFG.num_attention_heads

    def downgrades(page_tokens, tier):
        rng = np.random.default_rng(93)
        seq = 5 * page_tokens + 3
        pages = seq // page_tokens
        k = [normal(rng, seq, kv_heads, hd) for _ in range(LAYERS)]
        v = [normal(rng, seq, kv_heads, hd) for _ in range(LAYERS)]
        born = [[Tier.FP16] * pages for _ in range(LAYERS)]
        born[0][0] = Tier.INT8  # so that a cache at FP16 seals
        moves = {(0, 1): Tier.INT4, (0, 3): Tier.INT2, (1, 0): Tier.INT8, (1, 4): Tier.INT4}
        final = [row[:] for row in born]
        for (layer, i), target in moves.items():
            final[layer][i] = target
        runs = [page_tokens + 5, seq - page_tokens - 5]

        def built(tier_map):
            allocator, cache = pages_for(CFG, page_tokens, tier, tier_map=tier_map)
            for layer in range(LAYERS):
                start = store_in_runs(cache, layer, k[layer], v[layer], runs)
            return allocator, cache, start

        allocator, cache, start = built(born)
        before = snapshot(allocator)
        for (layer, i), target in moves.items():
            cache.downgrade(layer, i, target)
        for layer in range(LAYERS):
            for i in range(pages):
                assert cache.page_tier(layer, i) == final[layer][i], (layer, i)
                if (layer, i) in moves:
                    span = slice(i * page_tokens, (i + 1) * page_tokens)
                    want = _microinfer.quantise_page(k[layer][span], v[layer][span],
                                                     final[layer][i], page_tokens)
                else:
                    want = before[(layer, i)]
                np.testing.assert_array_equal(allocator.read(layer, i), want,
                                              err_msg=f"layer {layer} page {i}")
        for p in device.open_pages:
            np.testing.assert_array_equal(allocator.read(0, p), before[(0, p)])

        reference, cache_at_final, _ = built(final)
        for layer in range(LAYERS):
            for bias in (None, normal(rng, kv_heads, hd) * 30):
                q = normal(rng, seq - start, heads, hd)
                got = attend(CFG, cache, layer, q, k[layer], v[layer], bias, start)
                np.testing.assert_array_equal(
                    got, attend(CFG, cache_at_final, layer, q, k[layer], v[layer], bias, start))
                np.testing.assert_array_equal(
                    got, causal_reference(CFG, allocator, layer, q, k[layer], v[layer], bias,
                                          start, page_tokens))

    each([(p, t) for p in PAGE_TOKENS for t in (Tier.FP16, Tier.INT4)], downgrades,
         name=lambda c: f"P={c[0]} {c[1].name}")


def test_a_downgrade_whose_slot_takes_the_tail_keeps_every_other_page():
    """The FP16 page a downgrade frees is not its tier's tail, so the tail, a
    page of another layer, moves into its slot, and keeps its bytes; and a
    page of another cache sharing the allocator keeps its bytes and its slot
    in the target tier, where the new page goes after it."""
    page_tokens = PAGE_TOKENS[0]
    rng = np.random.default_rng(7)
    kv_heads, hd = CFG.num_key_value_heads, CFG.head_dim
    seq = 3 * page_tokens
    born = [[Tier.INT8, Tier.FP16, Tier.FP16]] * LAYERS
    allocator, cache = pages_for(CFG, page_tokens, Tier.FP16, tier_map=born)
    allocator.allocate(99, 0, Tier.INT4)
    other = rng.integers(0, 256, allocator.page_bytes(Tier.INT4), dtype=np.uint8)
    allocator.write(99, 0, other)
    for layer in range(LAYERS):
        store_in_runs(cache, layer, normal(rng, seq, kv_heads, hd), normal(rng, seq, kv_heads, hd),
                      [seq])
    tail = allocator.pages(Tier.FP16)[-1]
    assert tail != (0, 1)
    slot = allocator.locate(0, 1)[1]
    before = snapshot(allocator)
    cache.downgrade(0, 1, Tier.INT4)
    assert allocator.locate(*tail) == (Tier.FP16, slot), "the tail did not move into the slot"
    assert allocator.locate(99, 0) == (Tier.INT4, 0) and allocator.locate(0, 1) == (Tier.INT4, 1)
    np.testing.assert_array_equal(allocator.read(99, 0), other)
    for key, was in before.items():
        if key != (0, 1):
            np.testing.assert_array_equal(allocator.read(*key), was, err_msg=str(key))


def test_what_a_downgrade_refuses():
    """Only a sealed page of positions at FP16 downgrades, and only to a
    quantised tier, in a cache that seals: not an open page, not a page
    whose positions are reserved but not yet stored, not a page already
    quantised, not to FP16, not in a cache that does not seal, nor under a
    diagnostic Halves, whose pages are FP16 by definition. A downgrade that
    cannot allocate its target page leaves the cache as it was."""
    page_tokens = PAGE_TOKENS[0]
    rng = np.random.default_rng(8)
    kv_heads, hd = CFG.num_key_value_heads, CFG.head_dim
    born = [[Tier.INT8, Tier.FP16, Tier.FP16]] * LAYERS
    allocator, cache = pages_for(CFG, page_tokens, Tier.FP16, tier_map=born)
    seq = 2 * page_tokens
    k, v = normal(rng, seq, kv_heads, hd), normal(rng, seq, kv_heads, hd)
    store_in_runs(cache, 0, k, v, [seq])
    cache.reserve(3 * page_tokens)  # page 2 is held, and not sealed
    for layer, page, target, match in ((0, device.open_pages[0], Tier.INT4, "page of positions"),
                                       (0, 2, Tier.INT4, "sealed"),
                                       (1, 1, Tier.INT4, "sealed"),
                                       (0, 0, Tier.INT4, "FP16"),
                                       (0, 1, Tier.FP16, "quantised tier"),
                                       (0, 9, Tier.INT4, "page of positions")):
        with pytest.raises(ValueError, match=match):
            cache.downgrade(layer, page, target)

    _, plain = pages_for(CFG, page_tokens, Tier.FP16)
    store_in_runs(plain, 0, k, v, [seq])
    with pytest.raises(ValueError, match="seal"):
        plain.downgrade(0, 0, Tier.INT4)
    _, diagnostic = pages_for(CFG, page_tokens, Tier.INT4, Halves.Keys)
    store_in_runs(diagnostic, 0, k, v, [seq])
    with pytest.raises(ValueError, match="diagnostic"):
        diagnostic.downgrade(0, 0, Tier.INT2)

    capacity = [0] * 4
    capacity[int(Tier.FP16)] = 4096
    capacity[int(Tier.INT8)] = LAYERS * 3
    full = PagedKVCache(tier_page_bytes(CFG, page_tokens), capacity)
    small = device.KVPages(full, LAYERS, page_tokens, kv_heads, hd, Tier.FP16, Halves.Both,
                           born)
    store_in_runs(small, 0, k, v, [seq])
    for j in range(capacity[int(Tier.INT8)] - len(full.pages(Tier.INT8))):
        full.allocate(98, j, Tier.INT8)  # until the INT8 range is full
    before = full.read(0, 1)
    with pytest.raises(ValueError, match="full"):
        small.downgrade(0, 1, Tier.INT8)
    assert small.page_tier(0, 1) == Tier.FP16
    np.testing.assert_array_equal(full.read(0, 1), before)


def test_a_cache_at_fp16_can_be_asked_to_seal():
    """always_seal makes a cache seal with every page born at FP16, as one
    whose map names a quantised tier does, so that its pages can be
    downgraded (#93): each page is sealed as its rows came, its positions in
    the open pages until then, and attention reads it as the causal
    reference does, before and after a downgrade. seals_for says so too."""
    page_tokens = PAGE_TOKENS[0]
    rng = np.random.default_rng(17)
    heads, kv_heads, hd = CFG.num_attention_heads, CFG.num_key_value_heads, CFG.head_dim
    assert device.KVPages.seals_for(Tier.FP16, [], True)
    assert not device.KVPages.seals_for(Tier.FP16, [], False)
    storage = [0] * 4
    storage[int(Tier.FP16)] = 4096
    storage[int(Tier.INT4)] = 4096
    allocator = PagedKVCache(tier_page_bytes(CFG, page_tokens), storage)
    cache = device.KVPages(allocator, LAYERS, page_tokens, kv_heads, hd, Tier.FP16,
                           Halves.Both, [], True)
    assert cache.seals and cache.born_at_one_tier
    seq = 3 * page_tokens + 5
    k, v = normal(rng, seq, kv_heads, hd), normal(rng, seq, kv_heads, hd)
    start = store_in_runs(cache, 0, k, v, [page_tokens + 3, seq - page_tokens - 3])
    assert sorted(allocator.pages(Tier.FP16)) == sorted(
        [(layer, i) for layer in range(LAYERS) for i in range(3)]
        + [(layer, p) for layer in range(LAYERS) for p in device.open_pages])
    for i in range(3):
        span = slice(i * page_tokens, (i + 1) * page_tokens)
        np.testing.assert_array_equal(
            allocator.read(0, i),
            np.concatenate([k[span], v[span]]).astype(np.float16).view(np.uint8).ravel())
    for downgrade in (False, True):
        if downgrade:
            cache.downgrade(0, 1, Tier.INT4)
        q = normal(rng, seq - start, heads, hd)
        np.testing.assert_array_equal(
            attend(CFG, cache, 0, q, k, v, None, start),
            causal_reference(CFG, allocator, 0, q, k, v, None, start, page_tokens))
