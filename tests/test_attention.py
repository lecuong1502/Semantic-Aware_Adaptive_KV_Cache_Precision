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


@pytest.mark.parametrize("name", MODELS)
@pytest.mark.parametrize("seq", sequence_lengths())
def test_prefill_matches_float64_reference(name, seq):
    """head_dim 64 and 128 both come from here: one per model (ADR-0003)."""
    cfg = ModelConfig.from_card(name)
    check(*make_qkv(seq, seq, cfg.num_attention_heads, cfg.head_dim, seed=seq))


def test_head_dim_covers_both_adr_0003_sizes():
    """The criterion names 64 and 128; this fails if the cards ever stop
    supplying both, rather than letting the parametrisation quietly shrink."""
    assert {ModelConfig.from_card(n).head_dim for n in MODELS} == {64, 128}


@pytest.mark.parametrize("name", MODELS)
@pytest.mark.parametrize("seq_k", [1, TILE_K - 1, TILE_K + 1, 300])
def test_decode_one_query_sees_every_key(name, seq_k):
    cfg = ModelConfig.from_card(name)
    check(*make_qkv(1, seq_k, cfg.num_attention_heads, cfg.head_dim, seed=seq_k))


@pytest.mark.parametrize("seq_q,seq_k", [(5, 40), (TILE_Q + 3, 3 * TILE_K + 1), (64, 65)])
def test_fewer_queries_than_keys_align_to_the_end(seq_q, seq_k):
    """Chunked prefill: the queries are the last seq_q positions."""
    cfg = ModelConfig.from_card(MODELS[0])
    check(*make_qkv(seq_q, seq_k, cfg.num_attention_heads, cfg.head_dim, seed=seq_q))


@pytest.mark.parametrize("head_dim", [16, 32, 64, 96, 128, 256])
def test_head_dim_is_a_parameter(head_dim):
    check(*make_qkv(TILE_K + 7, TILE_K + 7, 2, head_dim, seed=head_dim))


def test_arbitrary_fp32_input_is_rounded_to_fp16_and_nothing_else():
    """Seam B's advertised contract, stated exactly rather than as a tolerance.

    RoPE and the projections compare fp32 input against a float64 reference
    with a floor for the input rounding. Attention cannot: rounding q and k
    perturbs every score, and that perturbation is exponentiated, so no floor
    derived from the output's terms bounds it. What can be said exactly is that
    the fp32 input costs one round-to-nearest into fp16 and then the kernel runs
    as it would have on that fp16 input — bit for bit."""
    cfg = ModelConfig.from_card(MODELS[0])
    rng = np.random.default_rng(4)
    q, k, v = (rng.standard_normal((50, cfg.num_attention_heads, cfg.head_dim)).astype(np.float32)
               for _ in range(3))
    np.testing.assert_array_equal(
        _microinfer.attention(q, k, v),
        _microinfer.attention(fp16_exact(q), fp16_exact(k), fp16_exact(v)))


def test_the_gate_would_reject_fp16_softmax_weights():
    """The floor admits fp32 rounding of scores and sums, and must not admit
    more. The obvious next optimisation — holding the softmax weights in fp16
    for the P @ V product, as tensor-core kernels do — is simulated in NumPy
    and checked to fail. It is a legitimate trade for a later ticket to make,
    but it would have to be made through ADR-0006, not past it."""
    cfg = model_with_head_dim(128)
    seq = 3 * TILE_K
    q, k, v = make_qkv(seq, seq, cfg.num_attention_heads, cfg.head_dim, seed=10)
    p16 = fp16_exact(reference_weights(q, k)[0]).astype(np.float64)
    fp16_weights = fp16_exact(np.einsum("hqk,khd->qhd", p16, v.astype(np.float64)))
    with pytest.raises(AssertionError, match="relative error"):
        check(q, k, v, got=fp16_weights)


# --- grouped-query attention ---------------------------------------------


@pytest.mark.parametrize("name", MODELS)
@pytest.mark.parametrize("seq", [1, TILE_K - 1, TILE_K + 1, 257])
def test_grouped_query_attention_at_model_shapes(name, seq):
    """The configuration the engine actually runs: every query head of the
    model, sharing the model's two KV heads. Both grouping factors, 7 and 6,
    come from the cards."""
    cfg = ModelConfig.from_card(name)
    assert cfg.num_key_value_heads < cfg.num_attention_heads
    check(*make_qkv(seq, seq, cfg.num_attention_heads, cfg.head_dim, seed=seq,
                    kv_heads=cfg.num_key_value_heads))


@pytest.mark.parametrize("name", MODELS)
def test_grouped_decode_one_query_sees_every_key(name):
    cfg = ModelConfig.from_card(name)
    check(*make_qkv(1, 300, cfg.num_attention_heads, cfg.head_dim, seed=3,
                    kv_heads=cfg.num_key_value_heads))


@pytest.mark.parametrize("name", MODELS)
@pytest.mark.parametrize("heads,kv_heads", [(8, 1), (8, 2), (8, 4), (6, 3), (7, 1), (12, 12)])
def test_grouping_factor_is_a_parameter(name, heads, kv_heads):
    """Group sizes the models do not use, including one KV head for all (MQA)
    and none shared at all, so nothing can quietly assume 7 or 6.

    The head counts are literals because not being the cards' is the point.
    head_dim is not: it comes from each card, as in every other numerics test."""
    head_dim = ModelConfig.from_card(name).head_dim
    check(*make_qkv(TILE_K + 5, TILE_K + 5, heads, head_dim, seed=heads * 10 + kv_heads,
                    kv_heads=kv_heads))


@pytest.mark.parametrize("name", MODELS)
def test_each_query_head_reads_its_own_group_kv_head(name):
    """Change one KV head, and exactly its group of query heads moves.

    The mapping is the thing grouping can get wrong while still matching a
    reference that happens to be symmetric. HuggingFace assigns query head h to
    KV head h // group (contiguous groups); h % kv_heads (interleaved) is the
    plausible mistake, and it passes any test whose KV heads are identical."""
    cfg = ModelConfig.from_card(name)
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
        assert differs == [h // group == changed for h in range(heads)], f"KV head {changed}"


# --- causality -------------------------------------------------------------


@pytest.mark.parametrize("seq", [TILE_K - 3, 2 * TILE_K + 5])
def test_a_query_is_unaffected_by_later_keys_and_values(seq):
    """The criterion as stated: change everything at positions > i, and query
    i's output does not move — not approximately, but in every bit."""
    cfg = ModelConfig.from_card(MODELS[0])
    q, k, v = make_qkv(seq, seq, cfg.num_attention_heads, cfg.head_dim, seed=1)
    base = _microinfer.attention(q, k, v)
    rng = np.random.default_rng(2)
    for i in (0, 1, TILE_K // 2, seq - 2):
        k2, v2 = k.copy(), v.copy()
        k2[i + 1 :] = fp16_exact(1000 * rng.standard_normal(k2[i + 1 :].shape))
        v2[i + 1 :] = fp16_exact(1000 * rng.standard_normal(v2[i + 1 :].shape))
        moved = _microinfer.attention(q, k2, v2)
        np.testing.assert_array_equal(moved[: i + 1], base[: i + 1], err_msg=f"rows <= {i}")
        assert not np.array_equal(moved[i + 1 :], base[i + 1 :]), "the change should be visible"


def test_masked_keys_do_not_reach_the_arithmetic():
    """A NaN in a masked key must not leak. A mask applied by multiplying by
    zero, or by adding a large negative number, lets it through; a mask applied
    by selection does not."""
    cfg = ModelConfig.from_card(MODELS[0])
    seq = TILE_K + 9
    q, k, v = make_qkv(seq, seq, cfg.num_attention_heads, cfg.head_dim, seed=5)
    base = _microinfer.attention(q, k, v)
    k2 = k.copy()
    k2[-1] = np.nan
    got = _microinfer.attention(q, k2, v)
    np.testing.assert_array_equal(got[:-1], base[:-1])


def test_a_query_attends_to_itself():
    """With the diagonal masked by an off-by-one, position 0 would see nothing
    and the output would be NaN or zero. It must be exactly v[0]'s row."""
    cfg = ModelConfig.from_card(MODELS[0])
    q, k, v = make_qkv(3, 3, cfg.num_attention_heads, cfg.head_dim, seed=6)
    out = _microinfer.attention(q, k, v)
    np.testing.assert_array_equal(out[0], v[0])


# --- stability -------------------------------------------------------------


@pytest.mark.parametrize("name", MODELS)
def test_a_large_common_score_offset_changes_nothing(name):
    """Softmax is invariant to adding a constant to every score, but exp() is
    not: a score near 1000 overflows even fp64. The offset is built into the
    inputs — one dimension of every key carries a large shared component — so
    it survives as far as the kernel. Only a tracked running maximum survives
    it in turn; an assumed one (0, or the first tile's) overflows or underflows
    to NaN.

    This is a stability test, and a weak precision test by construction. An
    fp32 score near 1000 carries an ulp of 6e-5, so the floor — which is honest
    about that — judges most outputs against their terms. The precision claim
    at large scores is test_the_maximum_can_arrive_late's, where the floor
    judges under 1% of outputs."""
    cfg = ModelConfig.from_card(name)
    seq = 3 * TILE_K + 5
    q, k, v = make_qkv(seq, seq, cfg.num_attention_heads, cfg.head_dim, seed=7)
    q[..., -1], k[..., -1] = 32.0, 352.0  # scaled score offset ~ 32*352/sqrt(d)
    offset = 32.0 * 352.0 / np.sqrt(cfg.head_dim)
    assert offset > 700, "not large enough to overflow an unshifted exp()"
    got = _microinfer.attention(q, k, v)
    assert np.all(np.isfinite(got))
    check(q, k, v, got=got)


@pytest.mark.parametrize("name", MODELS)
def test_the_maximum_can_arrive_late(name):
    """A running maximum that is only ever set from the first tile passes the
    offset test, because every key shares the offset. Here one late key towers
    over everything before it, so the maximum changes after tiles have been
    accumulated, and their contribution has to be rescaled correctly."""
    cfg = ModelConfig.from_card(name)
    seq = 4 * TILE_K
    q, k, v = make_qkv(seq, seq, cfg.num_attention_heads, cfg.head_dim, seed=8)
    q[..., -1] = 16.0
    k[..., -1] = np.linspace(0.0, 400.0, seq, dtype=np.float32)[:, None].astype(np.float16)
    got = _microinfer.attention(q, k, v)
    assert np.all(np.isfinite(got))
    check(q, k, v, got=got)


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


def test_returns_the_query_shape():
    q, k, v = make_qkv(5, 9, 3, 64)
    out = _microinfer.attention(q, k, v)
    assert isinstance(out, np.ndarray)
    assert out.shape == q.shape


def test_rejects_more_queries_than_keys():
    """Bottom-right alignment would put the first query before position 0."""
    q, k, v = make_qkv(10, 4, 2, 64)
    with pytest.raises(ValueError, match="10"):
        _microinfer.attention(q, k, v)


@pytest.mark.parametrize("heads,kv_heads", [(4, 3), (7, 2), (2, 4)])
def test_rejects_head_counts_that_do_not_group(heads, kv_heads):
    """Every KV head must serve the same number of query heads. A remainder has
    no defined grouping, and more KV heads than query heads has none either."""
    q, k, v = make_qkv(4, 4, heads, 64, kv_heads=kv_heads)
    with pytest.raises(ValueError, match="head"):
        _microinfer.attention(q, k, v)


def test_zero_heads_is_empty_only_when_both_sides_have_none():
    """No heads on either side is an empty computation, as zero tokens is. No
    heads on one side only has no grouping: 0 query heads over 2 KV heads would
    make every group empty, and 2 over 0 would divide by zero in the kernel."""
    q, k, v = make_qkv(4, 4, 0, 64, kv_heads=0)
    assert _microinfer.attention(q, k, v).shape == (4, 0, 64)
    for heads, kv_heads in [(0, 2), (2, 0)]:
        q, k, v = make_qkv(4, 4, heads, 64, kv_heads=kv_heads)
        with pytest.raises(ValueError, match="both are zero or neither"):
            _microinfer.attention(q, k, v)


def test_rejects_mismatched_key_and_value_shapes():
    q, k, v = make_qkv(4, 6, 2, 64)
    with pytest.raises(ValueError):
        _microinfer.attention(q, k, v[:5].copy())


def test_rejects_mismatched_head_dim():
    q, _, _ = make_qkv(4, 4, 2, 64)
    _, k, v = make_qkv(4, 4, 2, 128)
    with pytest.raises(ValueError, match="head_dim"):
        _microinfer.attention(q, k, v)
