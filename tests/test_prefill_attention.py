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
    """One key far above the rest, and thousands that weigh little each and
    much together: at score 8 above them, 3e-4 of it each and 1.4 of it in
    all; at 12, 6e-6 each, among fp16's subnormals, and 2.5% in all. One fp16
    would hold the second to a percent, all of them alike."""
    def kept(cfg, score):
        heads, kv_heads, hd = cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim
        seq_k, seq_q = 4096, 20
        q, k, v = make_qkv(seq_q, seq_k, heads, hd, seed=3, kv_heads=kv_heads)
        k *= 0.01
        q *= 0.01
        q[..., 0] = 8.0
        k[:, :, 0] = 0.0
        k[0, :, 0] = np.float16(score / 8 * np.sqrt(hd))
        v[0] = 1.0
        check(q, k, v, got=launch(cfg, q, k, v))

    each([(c, s) for c in CFGS for s in (8, 12)], kept,
         name=lambda c: f"hd={c[0].head_dim} score {c[1]}")


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
    """set_prefill_on_tensor_cores says which kernel computes a chunk, and
    returns what was said before. Both are within the gate. A decode step
    gives the same output and the same per-page masses, to the bit, whichever
    it says: over one cache, an importance score's tolerance is zero."""
    cfg = CFGS[0]
    q, k, v = make_qkv(40, 300, cfg.num_attention_heads, cfg.head_dim, seed=9,
                       kv_heads=cfg.num_key_value_heads)
    def decode_step():
        """One query over a cache of every tier, with its masses."""
        seq = 300
        rng = np.random.default_rng(2)
        _, cache = pages_for(cfg, 32, Tier.FP16,
                             tier_map=[[MAP[i % 4] for i in range(seq // 32 + 1)]] * 2)
        keys = normal(rng, seq, cfg.num_key_value_heads, cfg.head_dim)
        store_in_runs(cache, 0, keys, keys, [seq - 1, 1])
        rope = device.RopeTable(cfg.head_dim, cfg.rope_theta)
        rope.cover(seq)
        out = device.empty(cfg.num_attention_heads * cfg.head_dim)
        mass = device.empty_f32(cfg.num_attention_heads * (seq // 32 + 1))
        cache.attention(0, put(q[-1:]), None, rope, out, 1, seq, cfg.num_attention_heads,
                        cfg.num_key_value_heads, cfg.head_dim, put(keys[-1:]), put(keys[-1:]),
                        mass)
        return out.to_numpy(), mass.to_numpy()

    on, step = launch(cfg, q, k, v), decode_step()
    assert device.set_prefill_on_tensor_cores(False) is True
    try:
        off, step_off = launch(cfg, q, k, v), decode_step()
    finally:
        assert device.set_prefill_on_tensor_cores(True) is False
    check(q, k, v, got=off)
    for a, b in zip(step, step_off):
        np.testing.assert_array_equal(a, b)
    np.testing.assert_array_equal(launch(cfg, q, k, v), on)
