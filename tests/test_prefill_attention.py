"""Prefill's attention on the tensor cores (#213, Seam B).

A launch of more than one query, at the models' head_dims, multiplies on the
tensor cores: fp16 operands accumulated in fp32. The operands that are not
fp16 numbers, a key completed by its bias (ADR-0009) and a softmax weight,
are each multiplied as two fp16 ones. The output is held to ADR-0006's
per-kernel gate, which a weight rounded to one fp16 fails (test_attention's
precision contract). test_attention and test_split_attention hold it at every
shape they have; test_quantised_pages and test_eviction to the bits of the
same query over contiguous keys, through pages at every tier. Here is what is
this kernel's own: what its fp16 operands must not lose, and the blocks of
one and of two tiles of queries it may be launched with.
"""

import numpy as np
from conftest import each
from test_attention import check, make_qkv, qwen_like_keys, rotated_bias
from test_kv_pages import MODELS, normal, put
from test_quantised_pages import attend, causal_reference, pages_for, store_in_runs

from microinfer import _microinfer
from microinfer._microinfer import Tier
from microinfer.config import ModelConfig

device = _microinfer.device
TILE_Q = _microinfer.attention_tiles["query"]
CFGS = [ModelConfig.from_card(n) for n in MODELS]
#: Every tier, cycled page by page.
MAP = [Tier.INT8, Tier.FP16, Tier.INT2, Tier.INT4]


def launch(cfg, q, k, v, bias=None, split=False):
    """One launch over contiguous keys, splitting them or not."""
    seq_q, heads, hd = q.shape
    rope = device.RopeTable(hd, cfg.rope_theta)
    rope.cover(len(k))
    out = device.empty(q.size)
    partials = device.empty_f32(device.attention_partials(hd)) if split else None
    device.attention(put(q), put(k), put(v), put(bias) if bias is not None else None, rope, out,
                     seq_q, len(k), heads, k.shape[1], hd, partials=partials)
    return out.to_numpy().reshape(q.shape)


def test_a_chunk_over_a_long_cache_matches_the_float64_reference():
    """Chunks of one tile of queries, of two and of several, over thousands
    of keys, with and without the key bias, split and not: where a prefill
    under pressure spends its time, and the launches whose shapes differ."""
    def matches(cfg, seq_q, split, biased):
        heads, kv_heads, hd = cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim
        seq_k = 1600
        if split:
            assert device.attention_split(seq_q, seq_k, heads, kv_heads, hd)[0] > 1
        if biased:
            q, k, v, bias = qwen_like_keys(cfg, seq_k, seed=seq_q)
            q = q[-seq_q:]
            whole = k.astype(np.float64) + rotated_bias(bias, seq_k, cfg.rope_theta)
        else:
            q, k, v = make_qkv(seq_q, seq_k, heads, hd, seed=seq_q, kv_heads=kv_heads)
            whole, bias = k, None
        check(q, whole, v, got=launch(cfg, q, k, v, bias, split))

    each([(c, n, s, b) for c in CFGS
          for n, s, b in ((2, False, True), (TILE_Q, True, True), (2 * TILE_Q, False, False),
                          (2 * TILE_Q + 3, True, True))], matches,
         name=lambda c: f"hd={c[0].head_dim} {c[1]} queries split={c[2]} bias={c[3]}")


def test_many_small_weights_beside_a_large_one_are_kept():
    """One key far above the rest, and thousands whose weights are under fp16's
    smallest number beside it but add up to as much: in one fp16 they would be
    flushed or coarse, and the output would be that one key's value."""
    def kept(cfg):
        heads, kv_heads, hd = cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim
        seq_k, seq_q = 4096, 20
        q, k, v = make_qkv(seq_q, seq_k, heads, hd, seed=3, kv_heads=kv_heads)
        k *= 0.01
        q *= 0.01
        q[..., 0] = 8.0
        # Score 8 = ln(4096) - 0.3 for key 0, near 0 for every other, whose
        # weights are then 2^-11.6 each and 0.7 in all ...
        k[:, :, 0] = 0.0
        k[0, :, 0] = np.float16(np.sqrt(hd))
        v[0] = 1.0
        ref = launch(cfg, q, k, v)
        check(q, k, v, got=ref)
        assert abs(ref.mean()) < 0.7, "the rest should weigh as much as key 0"
        # ... and 17, 6e-8 of the largest each, fp16's smallest number.
        k[0, :, 0] = np.float16(17 / 8 * np.sqrt(hd))
        check(q, k, v, got=launch(cfg, q, k, v))

    each(CFGS, kept, name=lambda c: f"hd={c.head_dim}")


def test_large_values_do_not_overflow_the_fp16_sums():
    """What a weight's first fp16 leaves is multiplied by V in fp16 sums: V at
    thousands, under weights that are nowhere one fp16, and whose sum over a
    tile is largest, stays finite and within the gate."""
    def finite(cfg):
        heads, kv_heads, hd = cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim
        q, k, v = make_qkv(40, 400, heads, hd, seed=5, kv_heads=kv_heads)
        k *= 0.01  # near-equal scores, so every weight is near its tile's largest
        v = np.where(v > 0, 8000.0, 7000.0).astype(np.float32)
        got = launch(cfg, q, k, v)
        assert np.isfinite(got).all()
        check(q, k, v, got=got)

    each(CFGS, finite, name=lambda c: f"hd={c.head_dim}")


def test_a_block_of_two_tiles_gives_each_query_what_a_block_of_one_does():
    """Over a cache that seals, two tiles of queries share a block where a page
    holds both, at P = 32 and 64, and not at P = 16 or 48; a launch may begin
    at any position, mid-tile or mid-page. Every query's output is, to the
    bit, its own over contiguous keys, at every tier."""
    def same(cfg, page_tokens, start):
        heads, kv_heads, hd = cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim
        seq = start + 2 * page_tokens + 9
        rng = np.random.default_rng(start)
        mapped = [[MAP[(i + layer) % 4] for i in range(seq // page_tokens + 1)]
                  for layer in range(2)]
        allocator, cache = pages_for(cfg, page_tokens, Tier.FP16, tier_map=mapped)
        k, v = normal(rng, seq, kv_heads, hd), normal(rng, seq, kv_heads, hd)
        begun = store_in_runs(cache, 0, k, v, [start, seq - start])
        q = normal(rng, seq - start, heads, hd)
        bias = normal(rng, kv_heads, hd) * 30
        np.testing.assert_array_equal(
            attend(cfg, cache, 0, q, k, v, bias, begun),
            causal_reference(cfg, allocator, 0, q, k, v, bias, begun, page_tokens))

    each([(c, p, s) for c in CFGS for p in (16, 32, 48, 64)
          for s in (1, TILE_Q, p + 3, p + TILE_Q + 5)], same,
         name=lambda c: f"hd={c[0].head_dim} P={c[1]} from {c[2]}")


def test_a_head_dim_it_is_not_compiled_for_is_still_attended():
    """Other head_dims take the kernel that computes a decode step, and give
    what they always did: within the gate."""
    each([(32,), (96,), (256,)], lambda d: check(*make_qkv(50, 300, 4, d, seed=d, kv_heads=2)))


def test_the_kernel_before_can_be_asked_for_and_a_decode_step_is_always_its():
    """prefill_on_tensor_cores says which kernel computes a chunk, and returns
    what was asked before. Both are within the gate. A launch of one query
    gives the same bits whichever is asked for."""
    cfg = CFGS[0]
    q, k, v = make_qkv(40, 300, cfg.num_attention_heads, cfg.head_dim, seed=9,
                       kv_heads=cfg.num_key_value_heads)
    on, one = launch(cfg, q, k, v), launch(cfg, q[-1:], k, v)
    assert device.prefill_on_tensor_cores(False) is True
    try:
        off, one_off = launch(cfg, q, k, v), launch(cfg, q[-1:], k, v)
    finally:
        assert device.prefill_on_tensor_cores(True) is False
    assert not np.array_equal(on, off), "two kernels round in different places"
    check(q, k, v, got=off)
    np.testing.assert_array_equal(one, one_off)
    np.testing.assert_array_equal(launch(cfg, q, k, v), on)
