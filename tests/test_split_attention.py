"""Attention that splits a long launch's keys across blocks (#204, Seam B).

A launch with too few blocks to fill the device, as a decode step's one
query is at long contexts, splits each query's keys into runs: each block
makes a partial softmax, m, l and an unnormalised output, over its run, and
a second kernel folds them. A decode step's block also serves every query
head of a KV head's group, so that each key tile is loaded and dequantised
once for all of them. Splitting changes the order in which the output is
summed, so a split launch is held to ADR-0006's per-kernel gate against
float64, not to the bits of an unsplit one. It splits only when given
partials, which the engine's workspace holds; without them it is the
unsplit kernel, which every other attention test holds to its references.

What holds to the bit is what does not depend on the order of the sums:
- the plan depends on the launch's shape alone, so a split launch through
  pages, at any tier, gives the bits it gives over contiguous keys holding
  what the pages hold;
- asking for the masses changes no bit of the output;
- an evicted page is a masked one, to the bit, even where a whole run of
  keys is unread.

The masses (#98) are made per page within a run and folded with the runs'
m and l, a page that spans two runs taking a share from each. They are held
to the float64 attention matrix's per-page sums within the tolerance the
unsplit kernel is held to, MASS_TOLERANCE, and so to the unsplit kernel's
within twice that.
"""

import numpy as np
from conftest import each
from test_attention import check, make_qkv, reference_attention
from test_attention_mass import MASS_TOLERANCE, SUM_TOLERANCE
from test_kv_pages import CFG, LAYERS, MODELS, normal, put
from test_kv_pages import pages_for as fp16_pages_for
from test_quantised_pages import pages_for, sealed, store_in_runs
from ulp_gate import assert_within_gate, floor_from_bound

from microinfer import _microinfer
from microinfer._microinfer import Tier
from microinfer.config import ModelConfig

device = _microinfer.device
TILE_Q = _microinfer.attention_tiles["query"]
TILE_K = _microinfer.attention_tiles["key"]
CFGS = [ModelConfig.from_card(n) for n in MODELS]
#: Page sizes: two the kernel's key tile divides and two it does not, one of
#: them, 48, dividing no run either, so that pages span two runs.
PAGE_TOKENS = [16, 32, 48, 64]
#: Every tier, cycled page by page.
MAP = [Tier.INT8, Tier.FP16, Tier.INT2, Tier.INT4]


def splits(seq_q, seq_k, cfg, page_tokens=0):
    return device.attention_split(seq_q, seq_k, cfg.num_attention_heads,
                                  cfg.num_key_value_heads, page_tokens)


def partials(cfg):
    return device.empty_f32(device.attention_partials(cfg.head_dim))


def test_the_cases_here_split_and_a_page_spans_two_runs():
    """Not a test of the kernel: of this file's reach. Each launch length
    below splits on this device, and at P = 48 some page lies across the
    boundary of two runs, which is the masses' hardest case."""
    for cfg in CFGS:
        for seq_q, seq_k in ((1, 700), (1, 3000), (1, 5000), (TILE_Q, 3000), (20, 2500)):
            assert splits(seq_q, seq_k, cfg)[0] > 1, (cfg.head_dim, seq_q, seq_k)
    runs, keys = splits(1, mass_seq(48), CFG)
    assert runs > 1 and keys % 48 != 0


def test_a_split_launch_matches_the_float64_reference():
    """Decode at three lengths and two prefill launches of few queries, at
    both models' heads and head_dims, and the running maximum across runs:
    a key towering over every other in the last run, so that every earlier
    run is rescaled at the fold, and in the first, so that every later run
    weighs almost nothing."""
    def plain(cfg, seq_q, seq_k):
        check(*make_qkv(seq_q, seq_k, cfg.num_attention_heads, cfg.head_dim, seed=seq_k,
                        kv_heads=cfg.num_key_value_heads))

    def towering(cfg, where):
        seq = 3000
        q, k, v = make_qkv(1, seq, cfg.num_attention_heads, cfg.head_dim, seed=5,
                           kv_heads=cfg.num_key_value_heads)
        q[..., -1] = 8.0
        k[..., -1] = 0.0
        k[where, :, -1] = 60.0
        got = _microinfer.attention(q, k, v)
        assert np.all(np.isfinite(got))
        check(q, k, v, got=got)

    each([(plain, c, q, s) for c in CFGS
          for q, s in ((1, 700), (1, 3000), (1, 5000), (TILE_Q, 3000), (20, 2500))]
         + [(towering, c, w) for c in CFGS for w in (0, 2999)],
         lambda f, *a: f(*a), name=lambda c: f"{c[0].__name__} hd={c[1].head_dim} {c[2:]}")


def test_through_pages_a_split_launch_is_bit_identical_to_contiguous():
    """At both models and at page sizes that do and do not divide a run,
    with and without the key bias: decode, and a tile of prefill queries,
    over pages give the bits they give over the same keys and values laid
    out contiguously."""
    def identical(cfg, page_tokens):
        rng = np.random.default_rng(page_tokens)
        heads, kv_heads, hd = cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim
        seq_k = 3000
        _, cache = fp16_pages_for(cfg, page_tokens)
        k, v = normal(rng, seq_k, kv_heads, hd), normal(rng, seq_k, kv_heads, hd)
        cache.reserve(seq_k)
        cache.store(1, put(k), put(v), 0, seq_k)
        rope = device.RopeTable(hd, cfg.rope_theta)
        rope.cover(seq_k)
        for bias in (None, normal(rng, kv_heads, hd) * 30):
            b = put(bias) if bias is not None else None
            for seq_q in (1, TILE_Q):
                q = put(normal(rng, seq_q, heads, hd))
                paged, whole = device.empty(seq_q * heads * hd), device.empty(seq_q * heads * hd)
                cache.attention(1, q, b, rope, paged, seq_q, seq_k, heads, kv_heads, hd,
                                partials=partials(cfg))
                device.attention(q, put(k), put(v), b, rope, whole, seq_q, seq_k, heads,
                                 kv_heads, hd, partials(cfg))
                np.testing.assert_array_equal(paged.to_numpy(), whole.to_numpy(),
                                              err_msg=f"seq_q={seq_q}")

    each([(c, p) for c in CFGS for p in (16, 48)], identical,
         name=lambda c: f"hd={c[0].head_dim} P={c[1]}")


def causal_cache(cfg, page_tokens, tier, mapped, seq, last, seed):
    """A cache that seals, at `tier` or born from a map of every tier, of
    `seq` positions of one layer, the last `last` stored in a run of their
    own, as a step stores them; its allocator, and the keys and values."""
    rng = np.random.default_rng(seed)
    kv_heads, hd = cfg.num_key_value_heads, cfg.head_dim
    k, v = normal(rng, seq, kv_heads, hd), normal(rng, seq, kv_heads, hd)
    tier_map = [MAP * (seq // page_tokens // 4 + 1)] * LAYERS if mapped else None
    allocator, cache = pages_for(cfg, page_tokens, tier, tier_map=tier_map)
    store_in_runs(cache, 0, k, v, [seq - last, last])
    return allocator, cache, k, v


def test_a_split_decode_reads_each_page_at_its_own_tier_to_the_bit():
    """A decode step over a long cache at INT4, and over one born from a
    map of every tier, with and without the key bias: the one query's
    output is, to the bit, split attention over contiguous keys holding the
    pages before its own as sealed, at each one's tier, and its own page's
    positions as they came."""
    def matches(cfg, tier, mapped):
        heads, kv_heads, hd = cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim
        seq = 3000
        allocator, cache, k, v = causal_cache(cfg, 32, tier, mapped, seq, 1, 11)
        own = (seq - 1) // 32 * 32
        sealed_k, sealed_v = sealed(allocator, cfg, 0, own // 32, 32)
        kk, vv = np.concatenate([sealed_k, k[own:]]), np.concatenate([sealed_v, v[own:]])
        rope = device.RopeTable(hd, cfg.rope_theta)
        rope.cover(seq)
        rng = np.random.default_rng(3)
        for bias in (None, normal(rng, kv_heads, hd) * 30):
            b = put(bias) if bias is not None else None
            q = put(normal(rng, 1, heads, hd))
            got, want = device.empty(heads * hd), device.empty(heads * hd)
            cache.attention(0, q, b, rope, got, 1, seq, heads, kv_heads, hd, put(k[-1:]),
                            put(v[-1:]), partials=partials(cfg))
            device.attention(q, put(kk), put(vv), b, rope, want, 1, seq, heads, kv_heads, hd,
                             partials(cfg))
            np.testing.assert_array_equal(got.to_numpy(), want.to_numpy())

    each([(c, t, m) for c in CFGS for t, m in ((Tier.INT4, False), (Tier.FP16, True))],
         matches, name=lambda c: f"hd={c[0].head_dim} {c[1].name}{' mapped' if c[2] else ''}")


def test_a_split_prefill_tile_over_sealed_pages_matches_the_float64_reference():
    """The queries of one prefill tile over a long cache born from a map of
    every tier, starting where no tile does, so that the launch has two:
    each query's output is within the gate of float64
    attention over what it reads, the pages before its own as sealed and
    its own as it came."""
    def matches(cfg):
        heads, kv_heads, hd = cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim
        seq, n = 3000 + 5, TILE_Q
        allocator, cache, k, v = causal_cache(cfg, 32, Tier.FP16, True, seq, n, 12)
        q = normal(np.random.default_rng(4), n, heads, hd)
        rope = device.RopeTable(hd, cfg.rope_theta)
        rope.cover(seq)
        out = device.empty(q.size)
        cache.attention(0, put(q), None, rope, out, n, seq, heads, kv_heads, hd,
                        put(k[-n:]), put(v[-n:]), partials=partials(cfg))
        got = out.to_numpy().reshape(n, heads, hd)
        sealed_k, sealed_v = sealed(allocator, cfg, 0, seq // 32, 32)
        refs, bounds = [], []
        for i, pos in enumerate(range(seq - n, seq)):
            own = pos // 32 * 32
            kk = np.concatenate([sealed_k[:own], k[own:pos + 1]])
            vv = np.concatenate([sealed_v[:own], v[own:pos + 1]])
            ref, bound = reference_attention(q[i:i + 1], kk, vv, positions=np.array([pos]))
            refs.append(ref)
            bounds.append(bound)
        assert_within_gate(got, np.concatenate(refs), floor_from_bound(np.concatenate(bounds)))

    each(CFGS, matches, name=lambda c: f"hd={c.head_dim}")


def mass_seq(page_tokens):
    """A decode length of 60 pages or more, and 9 positions, that splits;
    for a page no larger than a key tile, the first, and for a larger one
    the first at which a page spans two runs, its mass then folded from
    both."""
    for pages in range(60, 2000):
        seq = pages * page_tokens + 9
        runs, keys = splits(1, seq, CFG, page_tokens)
        # A run is whole key tiles, so a page that divides a tile divides it.
        if runs > 1 and (keys % page_tokens != 0 or TILE_K % page_tokens == 0):
            return seq
    raise AssertionError(f"no length here splits with a page of {page_tokens} across runs")


def decode_masses(cfg, cache, k, v, q, split, bias=None, with_mass=True):
    """One decode step's attention through the cache, split or not: its
    output, and the masses, (heads, pages), if asked for."""
    heads, kv_heads, hd = cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim
    seq_k, pages = len(k), -(-len(k) // cache.page_tokens)
    rope = device.RopeTable(hd, cfg.rope_theta)
    rope.cover(seq_k)
    out = device.empty(heads * hd)
    mass = device.empty_f32(heads * pages) if with_mass else None
    cache.attention(0, put(q), put(bias) if bias is not None else None, rope, out, 1, seq_k,
                    heads, kv_heads, hd, put(k[-1:]), put(v[-1:]), mass,
                    partials(cfg) if split else None)
    return out.to_numpy(), (mass.to_numpy().reshape(heads, pages) if with_mass else None)


def hand_made(cfg, allocator, cache, k, q):
    """Each head's float64 softmax over the keys its query reads, summed per
    page (test_attention_mass.py's reference, at any length)."""
    heads, kv_heads, hd = cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim
    page_tokens, seq_k = cache.page_tokens, len(k)
    own = (seq_k - 1) // page_tokens * page_tokens
    read = k.astype(np.float64).copy()
    read[:own] = sealed(allocator, cfg, 0, own // page_tokens, page_tokens)[0]
    group, pages = heads // kv_heads, -(-seq_k // page_tokens)
    want = np.zeros((heads, pages))
    for h in range(heads):
        scores = read[:, h // group, :] @ q[0, h].astype(np.float64) / np.sqrt(hd)
        p = np.exp(scores - scores.max())
        p /= p.sum()
        want[h] = np.add.reduceat(p, np.arange(0, seq_k, page_tokens))
    return want


def test_split_masses_are_a_hand_made_attention_matrix_s_per_page():
    """At every page size, over a cache at FP16, at INT4 and born from a map
    of every tier: each page's split mass, for each query head, is its share
    of a float64 softmax over the keys the query reads, within the unsplit
    kernel's tolerance, and the unsplit kernel's mass within twice it."""
    def matches(page_tokens, tier, mapped):
        seq = mass_seq(page_tokens)
        allocator, cache, k, v = causal_cache(CFG, page_tokens, tier, mapped, seq, 1, page_tokens)
        q = normal(np.random.default_rng(page_tokens + 1), 1, CFG.num_attention_heads,
                   CFG.head_dim)
        _, got = decode_masses(CFG, cache, k, v, q, split=True)
        _, unsplit = decode_masses(CFG, cache, k, v, q, split=False)
        np.testing.assert_allclose(got, hand_made(CFG, allocator, cache, k, q),
                                   rtol=0, atol=MASS_TOLERANCE)
        np.testing.assert_allclose(got, unsplit, rtol=0, atol=2 * MASS_TOLERANCE)

    each([(p, t, m) for p in PAGE_TOKENS for t, m in ((Tier.FP16, False), (Tier.INT4, False),
                                                    (Tier.FP16, True))],
         lambda p, t, m: matches(p, t, m),
         name=lambda c: f"P={c[0]} {c[1].name}{' mapped' if c[2] else ''}")


def test_split_masses_sum_to_one_and_change_no_bit_of_the_output():
    """For every query head of both models, with and without the key bias,
    at page sizes that do and do not divide a run: the masses are
    non-negative and sum to 1, and the output is the same to the bit with
    them asked for or not."""
    def sums(cfg, page_tokens):
        seq = mass_seq(page_tokens)
        _, cache, k, v = causal_cache(cfg, page_tokens, Tier.INT4, False, seq, 1, 3)
        q = normal(np.random.default_rng(6), 1, cfg.num_attention_heads, cfg.head_dim)
        for bias in (None, normal(np.random.default_rng(5), cfg.num_key_value_heads,
                                  cfg.head_dim) * 30):
            on, got = decode_masses(cfg, cache, k, v, q, True, bias)
            off, _ = decode_masses(cfg, cache, k, v, q, True, bias, with_mass=False)
            np.testing.assert_array_equal(on, off)
            assert (got >= 0).all()
            np.testing.assert_allclose(got.sum(axis=1), 1.0, rtol=0, atol=SUM_TOLERANCE)

    each([(c, p) for c in CFGS for p in (32, 48)], sums,
         name=lambda c: f"hd={c[0].head_dim} P={c[1]}")


def test_a_whole_run_of_evicted_pages_is_masked_to_the_bit_and_has_no_mass():
    """Pages evicted from one cache and masked in its twin, every page of
    the first run and more, and a page spanning two runs: the split decode
    step's output and masses are the same to the bit, the pages have no
    mass, and the rest sums to 1."""
    heads, kv_heads, hd = CFG.num_attention_heads, CFG.num_key_value_heads, CFG.head_dim
    page_tokens = 48
    seq = mass_seq(page_tokens)
    runs, keys = splits(1, seq, CFG, page_tokens)
    gone = list(range(keys // page_tokens + 2))
    twins = []
    for _ in range(2):
        twins.append(causal_cache(CFG, page_tokens, Tier.FP16, True, seq, 1, 7))
    _, evicting, k, v = twins[0]
    _, masking, _, _ = twins[1]
    for page in gone:
        evicting.evict(0, page)
        masking.mask(0, page)
    q = normal(np.random.default_rng(8), 1, heads, hd)
    a = decode_masses(CFG, evicting, k, v, q, split=True)
    b = decode_masses(CFG, masking, k, v, q, split=True)
    np.testing.assert_array_equal(a[0], b[0])
    np.testing.assert_array_equal(a[1], b[1])
    assert runs > 1 and (a[1][:, gone] == 0).all()
    np.testing.assert_allclose(a[1].sum(axis=1), 1.0, rtol=0, atol=SUM_TOLERANCE)
