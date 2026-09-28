"""Causal attention with online softmax, against a float64 NumPy reference.

**Grouped-query attention.** k and v may carry fewer heads than q: with
`group = num_attention_heads / num_key_value_heads`, query head `h` reads KV
head `h // group`, as HuggingFace's `repeat_kv` arranges it. Both models this
project runs use two KV heads (groups of 7 and 6), so grouping is the case that
matters. `group == 1` is ordinary multi-head attention and keeps every test
written for it before grouping existed (#8, #9).

**Causal alignment is bottom-right.** With `seq_q` queries over `seq_k` keys,
query `i` sits at absolute position `seq_k - seq_q + i` and attends keys at
positions `<=` that. For prefill `seq_q == seq_k` and this is the familiar lower
triangle; for decode `seq_q == 1` and the single query sees every key. One rule
serves both, so neither is a special case in the kernel.

The output is a weighted sum of value rows, so it cancels where values of mixed
sign are averaged, and the gate carries a floor (ADR-0006's amendment on
cancellation). Attention's floor admits two error sources, not one:

- the fp32 sum over the visible keys, as for any projection: sqrt(n) * u32 *
  terms, where terms = sum_j p_j |v_j|;
- the fp32 *scores*. A score is itself a dot product over head_dim, and its
  rounding, sqrt(d) * u32 * sum_d |q_d k_d| / sqrt(d), becomes a relative error
  in every softmax weight, which then weighs `terms`.

With that bound the floor judges 0.8% of outputs at head_dim 64 and 1.4% at
128, and the worst error is 1.04 ulp: the fp16 store. The score term is
needed. Without it the kernel reads 5-7 ulp at head_dim 128, and a simulation
in which only the scores are fp32 already reads 5.4 ulp. It is also not
generous. That simulation uses about a fifth of the term, the same slack the
accumulation term has.

The reference materialises the whole score matrix and an explicit mask, which
the kernel is forbidden to do. That is the point of it being a reference.
"""

import numpy as np
import pytest
from conftest import each
from ulp_gate import FP32_REL_ULP, assert_within_gate, floor_from_bound, fp16_exact

from microinfer import _microinfer
from microinfer.config import ModelConfig
from microinfer.models import VERIFIED

MODELS = sorted(VERIFIED)

#: The kernel's own tile sizes, read from the module so that "below one tile"
#: and "not a multiple of the tile" track the kernel rather than a guess at it.
TILE_Q = _microinfer.attention_tiles["query"]
TILE_K = _microinfer.attention_tiles["key"]


def query_positions(seq_q: int, seq_k: int) -> np.ndarray:
    return np.arange(seq_q) + (seq_k - seq_q)


def expand_kv(q, kv):
    """Give every query head its KV head, repeated in HuggingFace's order:
    query head h gets KV head h // group. The reference materialises this; the
    kernel only indexes it."""
    return np.repeat(kv, q.shape[1] // kv.shape[1], axis=1)


def reference_weights(q, k, positions=None):
    """float64 causal softmax weights, (heads, seq_q, seq_k), and the mask used.

    `positions` are the queries' absolute positions, bottom-right aligned by
    default. k may carry fewer heads than q."""
    k = expand_kv(q, k)
    seq_q, _, head_dim = q.shape
    seq_k = k.shape[0]
    if positions is None:
        positions = query_positions(seq_q, seq_k)
    scores = np.einsum("qhd,khd->hqk", q.astype(np.float64), k.astype(np.float64)) / np.sqrt(head_dim)
    masked = np.arange(seq_k)[None, :] > positions[:, None]  # (seq_q, seq_k)
    scores[:, masked] = -np.inf
    p = np.exp(scores - scores.max(axis=-1, keepdims=True))
    return p / p.sum(axis=-1, keepdims=True), masked


def reference_attention(q, k, v, positions=None):
    """float64 softmax(q k^T / sqrt(d) + mask) v, per head.

    `positions` are the queries' absolute positions, bottom-right aligned by
    default. Passing them explicitly lets a test check a few rows of a sequence
    too long to materialise whole. Returns the output and, per element, the
    absolute error bound the floor is built from (module docstring)."""
    seq_q, _, head_dim = q.shape
    if positions is None:
        positions = query_positions(seq_q, k.shape[0])
    p, masked = reference_weights(q, k, positions)
    k, v = expand_kv(q, k), expand_kv(q, v)
    q64, k64, v64 = (a.astype(np.float64) for a in (q, k, v))

    out = np.einsum("hqk,khd->qhd", p, v64)
    terms = np.einsum("hqk,khd->qhd", p, np.abs(v64))

    # The largest score magnitude-of-terms each query computes: bounds the
    # fp32 rounding of its scores.
    score_terms = np.einsum("qhd,khd->hqk", np.abs(q64), np.abs(k64)) / np.sqrt(head_dim)
    score_terms[:, masked] = 0.0
    score_terms = score_terms.max(axis=-1).T[..., None]  # (seq_q, heads, 1)

    visible = (positions + 1)[:, None, None]
    bound = (np.sqrt(visible) + np.sqrt(head_dim) * score_terms) * FP32_REL_ULP * terms
    return out, bound


def check(q, k, v, positions=None, got=None):
    """The gate, with the floor from the module docstring."""
    if got is None:
        got = _microinfer.attention(q, k, v)
    ref, bound = reference_attention(q, k, v, positions)
    return assert_within_gate(got, ref, floor_from_bound(bound))


def make_qkv(seq_q, seq_k, heads, head_dim, seed=0, kv_heads=None):
    """kv_heads defaults to heads: ordinary multi-head attention."""
    kv_heads = heads if kv_heads is None else kv_heads
    rng = np.random.default_rng(seed)
    q = fp16_exact(rng.standard_normal((seq_q, heads, head_dim)))
    k = fp16_exact(rng.standard_normal((seq_k, kv_heads, head_dim)))
    v = fp16_exact(rng.standard_normal((seq_k, kv_heads, head_dim)))
    return q, k, v


def model_with_head_dim(head_dim: int) -> ModelConfig:
    """Selected by the property a test needs, not by position in a sorted list."""
    (cfg,) = [c for c in map(ModelConfig.from_card, MODELS) if c.head_dim == head_dim]
    return cfg


def sequence_lengths():
    """1; below one tile; exactly one; one past; and several lengths that are
    not a multiple of either tile, so a partial tile appears on both axes."""
    return sorted({1, TILE_K // 2 - 1, TILE_Q, TILE_K, TILE_K + 1, 3 * TILE_K - 5, 100, 257})


def test_matches_the_float64_reference_at_every_shape():
    """Prefill at both models' head counts and head_dims (64 and 128, one per
    model: ADR-0003) over every awkward length; one query decoding over many
    keys; fewer queries than keys aligned to the end, as chunked prefill
    has; and head_dim as a parameter."""
    cfgs = [ModelConfig.from_card(n) for n in MODELS]
    assert {c.head_dim for c in cfgs} == {64, 128}  # the cards still supply both
    cases = ([(c.num_attention_heads, c.head_dim, s, s) for c in cfgs for s in sequence_lengths()]
             + [(c.num_attention_heads, c.head_dim, 1, s) for c in cfgs
                for s in (1, TILE_K - 1, TILE_K + 1, 300)]
             + [(cfgs[0].num_attention_heads, cfgs[0].head_dim, q, k)
                for q, k in ((5, 40), (TILE_Q + 3, 3 * TILE_K + 1), (64, 65))]
             + [(2, d, TILE_K + 7, TILE_K + 7) for d in (16, 32, 64, 96, 128, 256)])
    each(cases, lambda heads, head_dim, seq_q, seq_k: check(
        *make_qkv(seq_q, seq_k, heads, head_dim, seed=seq_q + seq_k)))


def test_the_precision_contract_is_fp32_scores_and_weights():
    """fp32 input costs exactly one rounding into fp16, and then the kernel
    runs as it would on that fp16 input, bit for bit: rounding q and k moves
    every score through an exponential, so no floor could bound it instead.

    And the floor admits fp32 scores and sums, and no more: holding the
    softmax weights in fp16 for P @ V, as tensor-core kernels do, is
    simulated and fails the gate. That trade would go through ADR-0006."""
    cfg = ModelConfig.from_card(MODELS[0])
    rng = np.random.default_rng(4)
    q, k, v = (rng.standard_normal((50, cfg.num_attention_heads, cfg.head_dim)).astype(np.float32)
               for _ in range(3))
    np.testing.assert_array_equal(
        _microinfer.attention(q, k, v),
        _microinfer.attention(fp16_exact(q), fp16_exact(k), fp16_exact(v)))

    cfg = model_with_head_dim(128)
    seq = 3 * TILE_K
    q, k, v = make_qkv(seq, seq, cfg.num_attention_heads, cfg.head_dim, seed=10)
    p16 = fp16_exact(reference_weights(q, k)[0]).astype(np.float64)
    fp16_weights = fp16_exact(np.einsum("hqk,khd->qhd", p16, v.astype(np.float64)))
    with pytest.raises(AssertionError, match="relative error"):
        check(q, k, v, got=fp16_weights)


def test_grouped_query_attention_reads_each_query_heads_own_kv_head():
    """The configuration the engine runs: every query head of each model over
    its two KV heads (groups of 7 and 6), prefilling and decoding; group sizes
    the models do not use, one KV head for all (MQA) and none shared, so
    nothing assumes 7 or 6. And the mapping itself: change one KV head, and
    exactly its group of query heads moves. HuggingFace gives query head h KV
    head h // group; h % kv_heads is the plausible mistake, and it passes any
    test whose KV heads are identical."""
    cfgs = [ModelConfig.from_card(n) for n in MODELS]
    assert all(c.num_key_value_heads < c.num_attention_heads for c in cfgs)
    cases = ([(c.num_attention_heads, c.num_key_value_heads, c.head_dim, q, k) for c in cfgs
              for q, k in ((1, 1), (TILE_K - 1, TILE_K - 1), (TILE_K + 1, TILE_K + 1),
                           (257, 257), (1, 300))]
             + [(h, kv, c.head_dim, TILE_K + 5, TILE_K + 5) for c in cfgs
                for h, kv in ((8, 1), (8, 2), (8, 4), (6, 3), (7, 1), (12, 12))])
    each(cases, lambda heads, kv_heads, head_dim, seq_q, seq_k: check(
        *make_qkv(seq_q, seq_k, heads, head_dim, seed=heads * 10 + kv_heads + seq_k,
                  kv_heads=kv_heads)))

    for cfg in cfgs:
        heads, kv_heads = cfg.num_attention_heads, cfg.num_key_value_heads
        group = heads // kv_heads
        q, k, v = make_qkv(TILE_K + 3, TILE_K + 3, heads, cfg.head_dim, seed=4, kv_heads=kv_heads)
        base = _microinfer.attention(q, k, v)
        for changed in range(kv_heads):
            k2, v2 = k.copy(), v.copy()
            k2[:, changed] = -k2[:, changed]
            v2[:, changed] = -v2[:, changed]
            moved = _microinfer.attention(q, k2, v2)
            differs = [not np.array_equal(moved[:, h], base[:, h]) for h in range(heads)]
            assert differs == [h // group == changed for h in range(heads)], (cfg.name, changed)


def test_attention_is_causal():
    """Change everything after position i, and query i's output does not move,
    in any bit. A NaN in a masked key does not leak, as it would through a
    mask applied by multiplying by zero or adding a large negative number. And
    position 0 attends to itself: with the diagonal masked by an off-by-one
    its output would be NaN or zero, not exactly v[0]."""
    cfg = ModelConfig.from_card(MODELS[0])
    rng = np.random.default_rng(2)
    for seq in (TILE_K - 3, 2 * TILE_K + 5):
        q, k, v = make_qkv(seq, seq, cfg.num_attention_heads, cfg.head_dim, seed=1)
        base = _microinfer.attention(q, k, v)
        for i in (0, 1, TILE_K // 2, seq - 2):
            k2, v2 = k.copy(), v.copy()
            k2[i + 1 :] = fp16_exact(1000 * rng.standard_normal(k2[i + 1 :].shape))
            v2[i + 1 :] = fp16_exact(1000 * rng.standard_normal(v2[i + 1 :].shape))
            moved = _microinfer.attention(q, k2, v2)
            np.testing.assert_array_equal(moved[: i + 1], base[: i + 1], err_msg=f"{seq}, <= {i}")
            assert not np.array_equal(moved[i + 1 :], base[i + 1 :]), "the change should show"

    seq = TILE_K + 9
    q, k, v = make_qkv(seq, seq, cfg.num_attention_heads, cfg.head_dim, seed=5)
    base = _microinfer.attention(q, k, v)
    k2 = k.copy()
    k2[-1] = np.nan
    np.testing.assert_array_equal(_microinfer.attention(q, k2, v)[:-1], base[:-1])

    q, k, v = make_qkv(3, 3, cfg.num_attention_heads, cfg.head_dim, seed=6)
    np.testing.assert_array_equal(_microinfer.attention(q, k, v)[0], v[0])


def test_the_running_maximum_survives_large_and_late_scores():
    """Softmax does not change when every score moves by a constant, but exp()
    overflows even fp64 near 1000: an offset built into one dimension of every
    key survives to the kernel, and only a tracked running maximum survives
    it in turn. A stability test, and a weak precision one by construction:
    an fp32 score near 1000 has an ulp of 6e-5.

    A maximum set only from the first tile would pass that, every key sharing
    the offset. So one late key also towers over all before it: the maximum
    changes after tiles have been accumulated, and they must be rescaled.
    There the floor judges under 1% of outputs."""
    def offset(name):
        cfg = ModelConfig.from_card(name)
        seq = 3 * TILE_K + 5
        q, k, v = make_qkv(seq, seq, cfg.num_attention_heads, cfg.head_dim, seed=7)
        q[..., -1], k[..., -1] = 32.0, 352.0  # scaled score offset ~ 32*352/sqrt(d)
        assert 32.0 * 352.0 / np.sqrt(cfg.head_dim) > 700, "too small to overflow exp()"
        got = _microinfer.attention(q, k, v)
        assert np.all(np.isfinite(got))
        check(q, k, v, got=got)

    def late(name):
        cfg = ModelConfig.from_card(name)
        seq = 4 * TILE_K
        q, k, v = make_qkv(seq, seq, cfg.num_attention_heads, cfg.head_dim, seed=8)
        q[..., -1] = 16.0
        k[..., -1] = np.linspace(0.0, 400.0, seq, dtype=np.float32)[:, None].astype(np.float16)
        got = _microinfer.attention(q, k, v)
        assert np.all(np.isfinite(got))
        check(q, k, v, got=got)

    each([(offset, n) for n in MODELS] + [(late, n) for n in MODELS], lambda f, n: f(n),
         name=lambda c: f"{c[0].__name__} {c[1]}")


# --- no mask tensor -------------------------------------------------------


def test_runs_at_a_length_whose_mask_alone_would_not_fit():
    """No explicit mask tensor: tested by what it would cost. At this length a
    boolean mask for one head is larger than the whole device, so a kernel that
    materialised one — or the score matrix, four times larger again — could not
    run. The inputs themselves are a few tens of MiB.

    The length is derived from this device's memory, so the claim holds on a
    larger card too. A few rows are checked against a reference built for
    those rows alone; the full reference would not fit either."""
    total = _microinfer.device_memory_info()["total"]
    seq = int(np.ceil(np.sqrt(total) / TILE_K) + 1) * TILE_K
    assert seq * seq > total

    head_dim = ModelConfig.from_card(MODELS[0]).head_dim
    rng = np.random.default_rng(9)
    q, k, v = (fp16_exact(rng.standard_normal((seq, 1, head_dim), dtype=np.float32)) for _ in range(3))

    got = _microinfer.attention(q, k, v)

    rows = np.array([0, TILE_K, seq // 2, seq - 1])
    check(q[rows], k, v, positions=rows, got=got[rows])


# --- the surface -----------------------------------------------------------


def test_the_surface_returns_the_query_shape_and_refuses_what_it_cannot_do():
    """The output has the queries' shape. Refused: more queries than keys,
    which bottom-right alignment would put before position 0; head counts
    that do not group (a remainder, or more KV heads than query heads); zero
    heads on one side only (none on both is an empty computation); keys and
    values of different shapes; a head_dim that differs between q and k."""
    q, k, v = make_qkv(5, 9, 3, 64)
    out = _microinfer.attention(q, k, v)
    assert isinstance(out, np.ndarray) and out.shape == q.shape
    q, k, v = make_qkv(4, 4, 0, 64, kv_heads=0)
    assert _microinfer.attention(q, k, v).shape == (4, 0, 64)

    refused = [(make_qkv(10, 4, 2, 64), "10")]
    refused += [(make_qkv(4, 4, h, 64, kv_heads=kv), "head") for h, kv in ((4, 3), (7, 2), (2, 4))]
    refused += [(make_qkv(4, 4, h, 64, kv_heads=kv), "both are zero or neither")
                for h, kv in ((0, 2), (2, 0))]
    q, k, v = make_qkv(4, 6, 2, 64)
    refused.append(((q, k, v[:5].copy()), None))
    q, _, _ = make_qkv(4, 4, 2, 64)
    _, k, v = make_qkv(4, 4, 2, 128)
    refused.append(((q, k, v), "head_dim"))
    for (q, k, v), message in refused:
        with pytest.raises(ValueError, match=message):
            _microinfer.attention(q, k, v)


# --- keys stored without their bias (ADR-0009) -------------------------------


def rotated_bias(bias, seq_k, theta):
    """float64 RoPE of the key bias at every key position 0..seq_k-1, in the
    rotate-half pairing: (seq_k, kv_heads, head_dim). Written as complex
    multiplication, as test_rope's reference is, rather than in the kernel's
    cos/sin shape."""
    half = bias.shape[-1] // 2
    inv_freq = theta ** (-np.arange(half, dtype=np.float64) * 2.0 / bias.shape[-1])
    rot = np.exp(1j * np.arange(seq_k)[:, None] * inv_freq)[:, None, :]  # (seq_k, 1, half)
    b = bias.astype(np.float64)
    z = (b[..., :half] + 1j * b[..., half:])[None] * rot
    return np.concatenate([z.real, z.imag], axis=-1)


def qwen_like_keys(cfg, seq, seed):
    """Keys shaped as Qwen2.5's are: a small input-dependent part under a large
    per-channel bias. Measured on the 0.5B's first layers, the bias reaches 147
    while the rest stays within about 5-11, and queries reach 80."""
    rng = np.random.default_rng(seed)
    kv_heads, hd = cfg.num_key_value_heads, cfg.head_dim
    q = fp16_exact(8 * rng.standard_normal((seq, cfg.num_attention_heads, hd)))
    k_part = fp16_exact(0.3 * rng.standard_normal((seq, kv_heads, hd)))
    v = fp16_exact(rng.standard_normal((seq, kv_heads, hd)))
    bias = fp16_exact(60 * rng.standard_normal((kv_heads, hd)))
    return q, k_part, v, bias


def test_keys_completed_by_their_bias_match_the_reference_and_beat_whole_fp16_keys():
    """k_bias completes each stored key as k + RoPE(bias, j) at its row j:
    the reference forms the whole key in float64, the kernel in fp32 from an
    fp16 part and an fp16 bias, held to the same gate as ever. And ADR-0009's
    reason: storing the whole key in fp16 rounds it at the bias's magnitude,
    where fp16's ulp is as large as what tells one key from another, and on
    the same keys that fails the gate. k_bias is one row per KV head."""
    def completed(name, seq_q, seq_k):
        cfg = ModelConfig.from_card(name)
        q, k_part, v, bias = qwen_like_keys(cfg, seq_k, seed=seq_q)
        q = q[-seq_q:]
        k_whole = k_part.astype(np.float64) + rotated_bias(bias, seq_k, cfg.rope_theta)
        check(q, k_whole, v, got=_microinfer.attention(q, k_part, v, k_bias=bias,
                                                       theta=cfg.rope_theta))

    each([(n, q, k) for n in MODELS
          for q, k in ((TILE_K + 7, TILE_K + 7), (1, 3 * TILE_K + 2))], completed)

    for name in MODELS:
        cfg = ModelConfig.from_card(name)
        seq = 2 * TILE_K + 3
        q, k_part, v, bias = qwen_like_keys(cfg, seq, seed=9)
        k_whole = k_part.astype(np.float64) + rotated_bias(bias, seq, cfg.rope_theta)
        with pytest.raises(AssertionError):
            check(q, k_whole, v, got=_microinfer.attention(q, fp16_exact(k_whole), v))

    q, k, v = make_qkv(4, 4, 4, 64, kv_heads=2)
    with pytest.raises(ValueError, match="k_bias"):
        _microinfer.attention(q, k, v, k_bias=np.zeros((4, 64), np.float32), theta=1e4)
