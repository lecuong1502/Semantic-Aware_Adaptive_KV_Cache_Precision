"""The KV cache on pages, against the contiguous one it replaces (#14, Seam B).

Paging changes where keys and values live, not what attention computes. So the
proof is equality, not tolerance: attention through the page table returns the
same bits as attention over a contiguous buffer holding the same keys and
values. The kernel is one template over both layouts, and only its addressing
differs.

P is the build's (`device.page_tokens`, ADR-0004), and the engine's tests run
it. `KVPages` takes P as a parameter, and these tests run two others: 16,
vLLM's, and 48, which is not a power of two. No code under test may work only
because P is 32.

Pages move. The allocator swaps a tier's tail page into a freed slot
(ADR-0007), so a test here frees a page between two launches, moving one of
the cache's own pages, and the second launch must still read the right
memory. Only a page table resolved after every allocator operation can pass
that.
"""

import numpy as np
import pytest
from ulp_gate import fp16_exact

from microinfer import _microinfer
from microinfer._microinfer import PagedKVCache, Tier
from microinfer.config import ModelConfig
from microinfer.models import VERIFIED

device = _microinfer.device
MODELS = sorted(VERIFIED)
PAGE_TOKENS = [16, 48]
LAYERS = 2
CFG = ModelConfig.from_card(MODELS[0])


def put(host):
    return _microinfer.upload_fp16(np.ascontiguousarray(host, np.float32).reshape(-1))


def normal(rng, *shape):
    return fp16_exact(rng.standard_normal(shape).astype(np.float32))


def pages_for(cfg, page_tokens, capacity_pages=4096):
    kv_width = cfg.num_key_value_heads * cfg.head_dim
    allocator = PagedKVCache([device.page_bytes(page_tokens, kv_width)] * 4,
                             [capacity_pages, 0, 0, 0])
    return allocator, device.KVPages(allocator, LAYERS, page_tokens, cfg.num_key_value_heads,
                                     cfg.head_dim, Tier.FP16)


def attend_both(cfg, cache, layer, q, k, v, bias, seq_k):
    """Attention through the pages and over contiguous copies of the same keys
    and values: the last seq_q queries against the first seq_k keys."""
    heads, kv_heads, hd = cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim
    rope = device.RopeTable(hd, cfg.rope_theta)
    rope.cover(seq_k)
    b = put(bias) if bias is not None else None
    paged, whole = device.empty(q.size), device.empty(q.size)
    cache.attention(layer, put(q), b, rope, paged, len(q), seq_k, heads, kv_heads, hd)
    device.attention(put(q), put(k[:seq_k]), put(v[:seq_k]), b, rope, whole,
                     len(q), seq_k, heads, kv_heads, hd)
    return paged.to_numpy(), whole.to_numpy()


# -- equality with the contiguous path ---------------------------------------


@pytest.mark.parametrize("name", MODELS)
@pytest.mark.parametrize("page_tokens", PAGE_TOKENS)
def test_paged_attention_is_bit_identical_to_contiguous(name, page_tokens):
    """Prefill and decode, with and without the key bias, at lengths below a
    page, at one page, one past it, and several pages with a ragged last one."""
    cfg = ModelConfig.from_card(name)
    rng = np.random.default_rng(page_tokens)
    heads, kv_heads, hd = cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim
    for seq_k in (1, page_tokens - 1, page_tokens, page_tokens + 1, 3 * page_tokens + 5):
        _, cache = pages_for(cfg, page_tokens)
        k, v = normal(rng, seq_k, kv_heads, hd), normal(rng, seq_k, kv_heads, hd)
        cache.reserve(seq_k)
        cache.store(1, put(k), put(v), 0, seq_k)
        for bias in (None, normal(rng, kv_heads, hd) * 30):
            for seq_q in {seq_k, 1}:
                q = normal(rng, seq_q, heads, hd)
                paged, whole = attend_both(cfg, cache, 1, q, k, v, bias, seq_k)
                np.testing.assert_array_equal(paged, whole, err_msg=f"P={page_tokens} seq_k={seq_k}")


@pytest.mark.parametrize("page_tokens", PAGE_TOKENS)
def test_storing_in_steps_is_storing_at_once(page_tokens):
    """Decode writes one position at a time, crossing page boundaries as it
    goes; the pages must end up holding what one store of the whole would."""
    rng = np.random.default_rng(7)
    kv_heads, hd = CFG.num_key_value_heads, CFG.head_dim
    seq = 2 * page_tokens + 3
    k, v = normal(rng, seq, kv_heads, hd), normal(rng, seq, kv_heads, hd)
    _, stepped = pages_for(CFG, page_tokens)
    for t in range(seq):
        stepped.reserve(t + 1)
        stepped.store(0, put(k[t]), put(v[t]), t, 1)
    q = normal(rng, 1, CFG.num_attention_heads, hd)
    paged, whole = attend_both(CFG, stepped, 0, q, k, v, None, seq)
    np.testing.assert_array_equal(paged, whole)


# -- pages move ---------------------------------------------------------------


@pytest.mark.parametrize("page_tokens", PAGE_TOKENS)
def test_a_page_moved_between_launches_is_still_read_correctly(page_tokens):
    """A page belonging to someone else is allocated between the cache's pages
    and then freed. The allocator moves its tier's tail, one of the cache's
    pages, into the freed slot. The next launch must find it there."""
    rng = np.random.default_rng(3)
    heads, kv_heads, hd = CFG.num_attention_heads, CFG.num_key_value_heads, CFG.head_dim
    allocator, cache = pages_for(CFG, page_tokens)
    seq = 3 * page_tokens
    k, v = normal(rng, seq, kv_heads, hd), normal(rng, seq, kv_heads, hd)
    q = normal(rng, 1, heads, hd)

    cache.reserve(page_tokens)
    allocator.allocate(99, 0, Tier.FP16)  # someone else's page, mid-range
    cache.reserve(seq)
    cache.store(0, put(k), put(v), 0, seq)
    before, _ = attend_both(CFG, cache, 0, q, k, v, None, seq)

    tail = allocator.pages(Tier.FP16)[-1]
    slot_before = allocator.locate(*tail)[1]
    allocator.free(99, 0)
    assert allocator.locate(*tail)[1] < slot_before, "no page of the cache moved"
    assert tail[0] < LAYERS, "the page that moved should be the cache's own"
    # The slot it left still holds its old bytes, which a stale address would
    # read back correctly by luck. Another page takes the slot and fills it
    # with NaN, so a stale read cannot pass.
    allocator.allocate(98, 0, Tier.FP16)
    assert allocator.locate(98, 0)[1] == slot_before
    allocator.write(98, 0, np.full(cache.page_bytes // 2, np.nan, np.float16))

    after, whole = attend_both(CFG, cache, 0, q, k, v, None, seq)
    np.testing.assert_array_equal(after, whole)
    np.testing.assert_array_equal(after, before)


# -- on demand, failure, and clean up ------------------------------------------


def test_pages_are_allocated_on_demand():
    page_tokens = PAGE_TOKENS[-1]
    allocator, cache = pages_for(CFG, page_tokens)
    held = lambda: len(allocator.pages(Tier.FP16))  # noqa: E731
    assert held() == 0
    cache.reserve(1)
    assert held() == LAYERS
    cache.reserve(page_tokens)
    assert held() == LAYERS, "a page already held is not allocated again"
    cache.reserve(page_tokens + 1)
    assert held() == 2 * LAYERS
    cache.reserve(3)
    assert held() == 2 * LAYERS, "reserving less never frees"
    assert cache.pages_per_layer == 2 and cache.capacity_tokens == 2 * page_tokens


def test_a_failed_reserve_gives_back_what_it_took():
    """Under contention an allocation will fail partway through a position
    range. The pages already taken for that range are returned, so nothing
    leaks, and the cache is as it was: a later reserve fails for the same
    reason, not because it finds half a range already allocated."""
    page_tokens = PAGE_TOKENS[0]
    allocator, cache = pages_for(CFG, page_tokens, capacity_pages=LAYERS + 1)
    cache.reserve(page_tokens)
    for _ in range(2):
        with pytest.raises(ValueError, match="full"):
            cache.reserve(page_tokens + 1)
        assert len(allocator.pages(Tier.FP16)) == LAYERS
        assert cache.pages_per_layer == 1


def test_the_cache_returns_its_pages_when_it_goes():
    allocator, cache = pages_for(CFG, PAGE_TOKENS[0])
    cache.reserve(100)
    assert len(allocator.pages(Tier.FP16)) > 0
    del cache
    assert allocator.pages(Tier.FP16) == []
    assert allocator.mapped_bytes(Tier.FP16) == 0


def test_misuse_is_refused():
    kv_width = CFG.num_key_value_heads * CFG.head_dim
    _, cache = pages_for(CFG, 16)
    cache.reserve(16)
    kv = device.empty(2 * kv_width)
    with pytest.raises(IndexError, match="reserved"):
        cache.store(0, kv, kv, 15, 2)
    with pytest.raises(IndexError, match="seq_k"):
        cache.attention(0, device.empty(CFG.hidden_size), None, None, device.empty(CFG.hidden_size),
                        1, 17, CFG.num_attention_heads, CFG.num_key_value_heads, CFG.head_dim)
    wrong = PagedKVCache([device.page_bytes(16, kv_width)] * 4, [8, 0, 0, 0])
    with pytest.raises(ValueError, match="page"):
        device.KVPages(wrong, LAYERS, 32, CFG.num_key_value_heads, CFG.head_dim, Tier.FP16)


# -- the RoPE table -------------------------------------------------------------


@pytest.mark.parametrize("name", MODELS)
def test_the_rope_table_is_the_fp64_value_rounded_once(name):
    """Attention completes keys with cos and sin from this table (ADR-0009,
    note from #14): each entry is the float64 value rounded once to fp32, out
    to the model's whole context window. It grows by doubling, so a decode
    step does not rebuild it."""
    cfg = ModelConfig.from_card(name)
    hd = cfg.head_dim
    table = device.RopeTable(hd, cfg.rope_theta)
    table.cover(5)
    table.cover(6)
    assert table.positions == 10, "grows by doubling"
    table.cover(cfg.max_position_embeddings)
    got = table.to_numpy()
    assert table.nbytes == got.nbytes

    angle = (np.arange(got.shape[0], dtype=np.float64)[:, None]
             * cfg.rope_theta ** (-2.0 * np.arange(hd // 2) / hd))
    for value, reference in ((got[..., 0], np.cos(angle)), (got[..., 1], np.sin(angle))):
        rounded = reference.astype(np.float32)
        ulps = np.abs(value - rounded) / np.spacing(np.abs(rounded).astype(np.float32))
        assert ulps.max() <= 1, f"{ulps.max()} fp32 ulp from the float64 value"


def test_a_key_bias_needs_a_table_that_reaches_every_key():
    rng = np.random.default_rng(1)
    heads, kv_heads, hd = CFG.num_attention_heads, CFG.num_key_value_heads, CFG.head_dim
    q, k = normal(rng, 1, heads, hd), normal(rng, 8, kv_heads, hd)
    short = device.RopeTable(hd, CFG.rope_theta)
    short.cover(4)
    for rope in (None, short):
        with pytest.raises(ValueError, match="RoPE table"):
            device.attention(put(q), put(k), put(k), put(normal(rng, kv_heads, hd)), rope,
                             device.empty(q.size), 1, 8, heads, kv_heads, hd)
