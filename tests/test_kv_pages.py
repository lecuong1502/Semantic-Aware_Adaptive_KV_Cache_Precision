"""The KV cache on pages, against the contiguous one it replaces (#14, Seam B).

Paging changes where keys and values live, not what attention computes. So the
proof is equality, not tolerance: attention through the page table returns the
same bits as attention over a contiguous buffer holding the same keys and
values. The kernel is one template over both layouts, and only its addressing
differs.

P is the build's (`device.page_tokens`, ADR-0004), but `KVPages` takes it as a
parameter, and every test here runs at 8, 16, 32 and 64. No test here, and no
code under test, may pass only because P is 32.

Pages move. The allocator swaps a tier's tail page into a freed slot
(ADR-0007), so a test here frees a page between two launches, moving one of
the cache's own pages, and the second launch must still read the right
memory. Only a page table resolved at each launch can pass that.
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
PAGE_TOKENS = [8, 16, 32, 64]
LAYERS = 2


@pytest.fixture(params=MODELS)
def cfg(request) -> ModelConfig:
    return ModelConfig.from_card(request.param)


def put(host):
    return _microinfer.upload_fp16(np.ascontiguousarray(host, np.float32).reshape(-1))


def normal(rng, *shape):
    return fp16_exact(rng.standard_normal(shape).astype(np.float32))


def allocator_for(page_tokens, row, capacity_pages=4096):
    page_bytes = 2 * page_tokens * row * 2  # keys then values, fp16
    return PagedKVCache([page_bytes] * 4, [capacity_pages, 0, 0, 0])


def pages_for(cfg, page_tokens, allocator=None):
    row = cfg.num_key_value_heads * cfg.head_dim
    allocator = allocator or allocator_for(page_tokens, row)
    return allocator, device.KVPages(allocator, LAYERS, page_tokens, row, Tier.FP16)


def attend_both(cfg, cache, layer, q, k, v, bias, seq_k):
    """Attention through the pages and over contiguous copies of the same
    keys and values, the last seq_q queries against the first seq_k keys."""
    heads, kv_heads, hd = cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim
    seq_q = len(q)
    paged = device.empty(q.size)
    cache.attention(layer, put(q), put(bias) if bias is not None else None, paged,
                    seq_q, seq_k, heads, kv_heads, hd, cfg.rope_theta)
    whole = device.empty(q.size)
    device.attention(put(q), put(k[:seq_k]), put(v[:seq_k]),
                     put(bias) if bias is not None else None, whole,
                     seq_q, seq_k, heads, kv_heads, hd, cfg.rope_theta)
    return paged.to_numpy(), whole.to_numpy()


# -- equality with the contiguous path ---------------------------------------


@pytest.mark.parametrize("page_tokens", PAGE_TOKENS)
@pytest.mark.parametrize("with_bias", [False, True])
def test_paged_attention_is_bit_identical_to_contiguous(cfg, page_tokens, with_bias):
    """Prefill and decode, at lengths below a page, at one page, one past it,
    and several pages with a ragged last one."""
    rng = np.random.default_rng(page_tokens)
    heads, kv_heads, hd = cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim
    bias = normal(rng, kv_heads, hd) * 30 if with_bias else None
    for seq_k in (1, page_tokens - 1, page_tokens, page_tokens + 1, 3 * page_tokens + 5):
        if seq_k < 1:
            continue
        _, cache = pages_for(cfg, page_tokens)
        k, v = normal(rng, seq_k, kv_heads, hd), normal(rng, seq_k, kv_heads, hd)
        cache.reserve(seq_k)
        cache.store(1, put(k), put(v), 0, seq_k)
        for seq_q in {seq_k, 1}:
            q = normal(rng, seq_q, heads, hd)
            paged, whole = attend_both(cfg, cache, 1, q, k, v, bias, seq_k)
            np.testing.assert_array_equal(paged, whole, err_msg=f"P={page_tokens} seq_k={seq_k}")
        assert cache.capacity_tokens >= seq_k


@pytest.mark.parametrize("page_tokens", PAGE_TOKENS)
def test_storing_in_steps_is_storing_at_once(cfg, page_tokens):
    """Decode writes one position at a time, crossing page boundaries as it
    goes; the pages must end up holding what one store of the whole would."""
    rng = np.random.default_rng(7)
    kv_heads, hd = cfg.num_key_value_heads, cfg.head_dim
    seq = 2 * page_tokens + 3
    k, v = normal(rng, seq, kv_heads, hd), normal(rng, seq, kv_heads, hd)
    _, stepped = pages_for(cfg, page_tokens)
    for t in range(seq):
        stepped.reserve(t + 1)
        stepped.store(0, put(k[t]), put(v[t]), t, 1)
    q = normal(rng, 1, cfg.num_attention_heads, hd)
    paged, whole = attend_both(cfg, stepped, 0, q, k, v, None, seq)
    np.testing.assert_array_equal(paged, whole)


# -- pages move ---------------------------------------------------------------


@pytest.mark.parametrize("page_tokens", PAGE_TOKENS)
def test_a_page_moved_between_launches_is_still_read_correctly(cfg, page_tokens):
    """A page belonging to someone else is allocated between the cache's pages
    and then freed. The allocator moves its tier's tail, one of the cache's
    pages, into the freed slot. The next launch must find it there."""
    rng = np.random.default_rng(3)
    heads, kv_heads, hd = cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim
    allocator, cache = pages_for(cfg, page_tokens)
    seq = 3 * page_tokens
    k, v = normal(rng, seq, kv_heads, hd), normal(rng, seq, kv_heads, hd)
    q = normal(rng, 1, heads, hd)

    cache.reserve(page_tokens)
    allocator.allocate(99, 0, Tier.FP16)  # someone else's page, mid-range
    cache.reserve(seq)
    cache.store(0, put(k), put(v), 0, seq)
    before, _ = attend_both(cfg, cache, 0, q, k, v, None, seq)

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

    after, whole = attend_both(cfg, cache, 0, q, k, v, None, seq)
    np.testing.assert_array_equal(after, whole)
    np.testing.assert_array_equal(after, before)


# -- on demand, and clean up ---------------------------------------------------


@pytest.mark.parametrize("page_tokens", PAGE_TOKENS)
def test_pages_are_allocated_on_demand(cfg, page_tokens):
    allocator, cache = pages_for(cfg, page_tokens)
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


def test_the_cache_returns_its_pages_when_it_goes(cfg):
    allocator, cache = pages_for(cfg, 16)
    cache.reserve(100)
    assert len(allocator.pages(Tier.FP16)) > 0
    del cache
    assert allocator.pages(Tier.FP16) == []
    assert allocator.mapped_bytes(Tier.FP16) == 0


def test_positions_off_the_pages_are_refused(cfg):
    _, cache = pages_for(cfg, 16)
    cache.reserve(16)
    row = cfg.num_key_value_heads * cfg.head_dim
    kv = device.empty(2 * row)
    with pytest.raises(IndexError, match="reserved"):
        cache.store(0, kv, kv, 15, 2)
    with pytest.raises(IndexError, match="seq_k"):
        cache.attention(0, device.empty(cfg.hidden_size), None, device.empty(cfg.hidden_size),
                        1, 17, cfg.num_attention_heads, cfg.num_key_value_heads,
                        cfg.head_dim, cfg.rope_theta)


def test_an_allocator_with_the_wrong_page_size_is_refused(cfg):
    row = cfg.num_key_value_heads * cfg.head_dim
    with pytest.raises(ValueError, match="page"):
        device.KVPages(allocator_for(16, row), LAYERS, 32, row, Tier.FP16)


def test_p_is_the_build_s_parameter():
    """ADR-0004: P is set by the build, and this build's is what the engine
    uses. Its value is not asserted: any P is a valid build."""
    assert isinstance(device.page_tokens, int) and device.page_tokens > 0
