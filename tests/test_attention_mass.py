"""Per-page attention mass from the attention kernel (#98, Seam B).

During decode, the attention kernel can write each page's share of each query
head's attention: the softmax's mass over the page's keys. It accumulates
each page's partial sum of exp(s - m) as the tiles go by, and normalises it
once the softmax's m and l are final. The scorer (#88) reads these.

The proof is a hand-made attention matrix: the scores of the one query
against every key it reads, a page before its own at the tier the page
table records for it and its own page as it came, softmaxed on the host,
and summed over each page's keys. The kernel's arithmetic is fp32 and its
exp is not numpy's, so the comparison is to a tolerance; a mass is a score
for ranking pages, not an output the model reads. The output the model does
read is bit-identical whether the masses are asked for or not.
"""

import numpy as np
import pytest
from conftest import each
from test_kv_pages import CFG, LAYERS, MODELS, PAGE_TOKENS, normal, put
from test_quantised_pages import pages_for, sealed, store_in_runs

from microinfer import _microinfer
from microinfer._microinfer import Tier
from microinfer.config import ModelConfig

device = _microinfer.device
#: Masses are fp32 in [0, 1], so a bound on them is in ulps of fp32 at 1.0
#: (ADR-0006): the kernel's fp32 dot products, exp and log against numpy's
#: float64. 2e-6 was the largest difference measured here; 32 ulps is 3.8e-6.
EPS = float(np.finfo(np.float32).eps)
MASS_TOLERANCE = 32 * EPS
#: A head's sum adds a page's error per page: at most 64 pages here.
SUM_TOLERANCE = 64 * MASS_TOLERANCE


def decode_state(cfg, page_tokens, tier, tier_map, seed):
    """A cache of `seq` positions of one layer, the last stored alone, as
    decode stores it; the rows, and the query of that last position."""
    rng = np.random.default_rng(seed)
    heads, kv_heads, hd = cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim
    seq = 3 * page_tokens + 9
    k, v = normal(rng, seq, kv_heads, hd), normal(rng, seq, kv_heads, hd)
    allocator, cache = pages_for(cfg, page_tokens, tier, tier_map=tier_map)
    store_in_runs(cache, 0, k, v, [seq - 1, 1])
    return allocator, cache, k, v, normal(rng, 1, heads, hd)


def decode_attention(cfg, cache, k, v, q, bias=None, with_mass=True):
    """One decode step's attention through the cache: its output, and the
    masses, (heads, pages), if asked for."""
    heads, kv_heads, hd = cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim
    seq_k, page_tokens = len(k), cache.page_tokens
    pages = -(-seq_k // page_tokens)
    rope = device.RopeTable(hd, cfg.rope_theta)
    rope.cover(seq_k)
    out = device.empty(heads * hd)
    mass = device.empty_f32(heads * pages) if with_mass else None
    cache.attention(0, put(q), put(bias) if bias is not None else None, rope, out, 1, seq_k,
                    heads, kv_heads, hd, put(k[-1:]), put(v[-1:]), mass)
    return out.to_numpy(), (mass.to_numpy().reshape(heads, pages) if with_mass else None)


def hand_made(cfg, allocator, cache, k, q):
    """Each head's softmax over the keys its query reads, summed per page:
    every page before its own as the cache holds it, dequantised at the
    tier the page table records, and its own page's keys as they came."""
    heads, kv_heads, hd = cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim
    page_tokens, seq_k = cache.page_tokens, len(k)
    own = (seq_k - 1) // page_tokens * page_tokens
    read = k.astype(np.float64).copy()
    if cache.seals:
        read[:own] = sealed(allocator, cfg, 0, own // page_tokens, page_tokens)[0]
    group = heads // kv_heads
    pages = -(-seq_k // page_tokens)
    want = np.zeros((heads, pages))
    for h in range(heads):
        scores = read[:, h // group, :] @ q[0, h].astype(np.float64) / np.sqrt(hd)
        p = np.exp(scores - scores.max())
        p /= p.sum()
        want[h] = [p[i * page_tokens:(i + 1) * page_tokens].sum() for i in range(pages)]
    return want


def test_the_masses_are_a_hand_made_attention_matrix_s_per_page():
    """At both page sizes, over a cache at FP16, one at INT4, and one built
    from a map of every tier: each page's mass, for each query head, is its
    share of a hand-made softmax over the keys the query reads."""
    def matches(page_tokens, tier, mapped):
        tier_map = ([[Tier.INT8, Tier.FP16, Tier.INT2, Tier.INT4]] * LAYERS if mapped else None)
        allocator, cache, k, v, q = decode_state(CFG, page_tokens, tier, tier_map, page_tokens)
        _, got = decode_attention(CFG, cache, k, v, q)
        np.testing.assert_allclose(got, hand_made(CFG, allocator, cache, k, q),
                                   rtol=0, atol=MASS_TOLERANCE)

    each([(p, t, m) for p in PAGE_TOKENS for t, m in ((Tier.FP16, False), (Tier.INT4, False),
                                                      (Tier.FP16, True))],
         lambda p, t, m: matches(p, t, m),
         name=lambda c: f"P={c[0]} {c[1].name}{' mapped' if c[2] else ''}")


def test_each_head_s_masses_sum_to_one():
    """For every query head of both models, with and without the key bias,
    at both page sizes: the masses are non-negative and sum to 1."""
    def sums(name, page_tokens):
        cfg = ModelConfig.from_card(name)
        allocator, cache, k, v, q = decode_state(cfg, page_tokens, Tier.INT4, None, 3)
        for bias in (None, normal(np.random.default_rng(5), cfg.num_key_value_heads,
                                  cfg.head_dim) * 30):
            _, got = decode_attention(cfg, cache, k, v, q, bias)
            assert (got >= 0).all()
            np.testing.assert_allclose(got.sum(axis=1), 1.0, rtol=0, atol=SUM_TOLERANCE)

    each([(n, p) for n in MODELS for p in PAGE_TOKENS], sums)


def test_with_the_masses_off_the_output_is_what_it_was():
    """The output is bit-identical with the masses asked for or not, at
    every tier and both page sizes. (Without them the kernel is compiled as
    it was before #98, and every other test of attention's output holds it
    to the references it held before.)"""
    def same(page_tokens, tier):
        allocator, cache, k, v, q = decode_state(CFG, page_tokens, tier, None, 11)
        bias = normal(np.random.default_rng(6), CFG.num_key_value_heads, CFG.head_dim) * 30
        on, _ = decode_attention(CFG, cache, k, v, q, bias)
        off, _ = decode_attention(CFG, cache, k, v, q, bias, with_mass=False)
        np.testing.assert_array_equal(on, off)

    each([(p, t) for p in PAGE_TOKENS for t in Tier.__members__.values()],
         lambda p, t: same(p, t), name=lambda c: f"P={c[0]} {c[1].name}")


def test_the_masses_are_refused_where_they_are_not_defined():
    """For more than one query, or into too small a buffer."""

    heads, kv_heads, hd = CFG.num_attention_heads, CFG.num_key_value_heads, CFG.head_dim
    _, cache, k, v, _ = decode_state(CFG, PAGE_TOKENS[0], Tier.INT4, None, 1)
    rope = device.RopeTable(hd, CFG.rope_theta)
    rope.cover(len(k))
    pages = -(-len(k) // PAGE_TOKENS[0])
    with pytest.raises(ValueError, match="one query"):
        cache.attention(0, put(normal(np.random.default_rng(0), 2, heads, hd)), None, rope,
                        device.empty(2 * heads * hd), 2, len(k), heads, kv_heads, hd,
                        put(k[-2:]), put(v[-2:]), device.empty_f32(heads * pages))
    with pytest.raises(ValueError, match="mass"):
        cache.attention(0, put(normal(np.random.default_rng(0), 1, heads, hd)), None, rope,
                        device.empty(heads * hd), 1, len(k), heads, kv_heads, hd,
                        put(k[-1:]), put(v[-1:]), device.empty_f32(heads * pages - 1))
