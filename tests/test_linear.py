"""The cuBLAS projection wrapper against a float64 NumPy reference.

ADR-0001 draws the boundary: the dense projections call cuBLAS, which is a BLAS
and not an inference engine. What this wrapper owns is everything around the
call — the row-major/column-major transposition, the weight layout HuggingFace
stores (`(out_features, in_features)`, so `y = x @ W.T`), the bias, and the
choice of fp32 accumulation. Those are the parts that can be wrong.

Shapes are derived from the model cards, never written out: q/o are square,
k/v are strongly rectangular (hidden -> num_key_value_heads * head_dim, a
factor of seven down for the 0.5B), gate/up widen and down narrows by the
intermediate ratio, and lm_head widens by two orders of magnitude.
"""

import numpy as np
import pytest
from conftest import each
from ulp_gate import accumulation_floor, assert_within_gate, fp16_exact, input_rounding_floor

from microinfer import _microinfer
from microinfer.config import ModelConfig
from microinfer.models import VERIFIED

MODELS = sorted(VERIFIED)


def projections(cfg: ModelConfig) -> dict[str, tuple[int, int, bool]]:
    """Every projection in one decoder layer: (in_features, out_features, bias).

    Qwen2 carries a bias on q, k and v only. That is easy to get wrong in both
    directions, so it is part of the shape table rather than a separate case."""
    kv = cfg.num_key_value_heads * cfg.head_dim
    return {
        "q_proj": (cfg.hidden_size, cfg.hidden_size, True),
        "k_proj": (cfg.hidden_size, kv, True),
        "v_proj": (cfg.hidden_size, kv, True),
        "o_proj": (cfg.hidden_size, cfg.hidden_size, False),
        "gate_proj": (cfg.hidden_size, cfg.intermediate_size, False),
        "up_proj": (cfg.hidden_size, cfg.intermediate_size, False),
        "down_proj": (cfg.intermediate_size, cfg.hidden_size, False),
    }


CASES = [(name, proj) for name in MODELS for proj in projections(ModelConfig.from_card(name))]


def reference_linear(x, w, bias=None, chunk: int = 16384):
    """float64 `x @ w.T + bias`, taken a slab of output features at a time so
    that lm_head's weight is never held in float64 whole (1 GiB).

    Returns the values and, per output, `|x| @ |w|.T + |bias|`: the magnitude
    of the terms each dot product sums, which bounds its error where the terms
    cancel (`ulp_gate`)."""
    x64 = x.astype(np.float64)
    out = np.empty((x.shape[0], w.shape[0]), dtype=np.float64)
    terms = np.empty_like(out)
    for start in range(0, w.shape[0], chunk):
        slab = w[start : start + chunk].astype(np.float64).T
        out[:, start : start + chunk] = x64 @ slab
        terms[:, start : start + chunk] = np.abs(x64) @ np.abs(slab)
    if bias is not None:
        out += bias.astype(np.float64)
        terms += np.abs(bias.astype(np.float64))
    return out, terms


def make_case(rows: int, in_features: int, out_features: int, bias: bool, seed: int = 0):
    """Weights scaled by 1/sqrt(in_features), as a trained layer's roughly are,
    so outputs sit near unit scale instead of growing with the reduction."""
    rng = np.random.default_rng(seed)
    x = fp16_exact(rng.standard_normal((rows, in_features), dtype=np.float32))
    w = fp16_exact(rng.standard_normal((out_features, in_features), dtype=np.float32)
                   / np.sqrt(in_features))
    b = fp16_exact(rng.standard_normal(out_features, dtype=np.float32)) if bias else None
    return x, w, b


def term_count(x, b) -> int:
    """How many terms each output adds: in_features products, plus the bias."""
    return x.shape[1] + (b is not None)


def within_gate(got, x, w, b=None):
    """The gate for fp16-exact input."""
    ref, terms = reference_linear(x, w, b)
    return assert_within_gate(got, ref, accumulation_floor(terms, term_count(x, b)))


def check(x, w, b):
    return within_gate(_microinfer.linear(x, w, b), x, w, b)


@pytest.mark.slow
def test_matches_float64_reference_at_every_shape():
    """Every projection of both models, at rows=1, a decode step, and rows=64,
    a prefill, since cuBLAS picks different algorithms for the two; the LM
    head at the full vocabulary, the most rectangular shape the model has,
    for the smaller model only (the larger one's weight is a 0.9 GiB host
    array for no more coverage); and shapes that are not multiples of eight,
    which tensor cores prefer and the wrapper may not require."""
    def projection(name, proj, rows):
        in_f, out_f, bias = projections(ModelConfig.from_card(name))[proj]
        check(*make_case(rows, in_f, out_f, bias, seed=in_f + out_f + rows))

    each([(n, p, r) for n, p in CASES for r in (1, 64)], projection)
    cfg = ModelConfig.from_card(MODELS[0])
    check(*make_case(4, cfg.hidden_size, cfg.vocab_size, bias=False, seed=11))
    each([(3, 5, 7), (1, 1, 1), (17, 33, 65), (2, 896, 3)], lambda rows, in_f, out_f: check(
        *make_case(rows, in_f, out_f, bias=True, seed=rows * in_f * out_f)))


def test_the_gate_holds_fp32_input_and_would_reject_fp16_accumulation():
    """fp32 input meets the gate with its rounding floored. And the
    accumulation floor forgives fp32's rounding and no more: partial sums
    kept in fp16, which is what reduced-precision split-K does and why the
    wrapper disallows it, are simulated and fail at the widest reduction the
    model has (measured, 370 ulp max; were the floor `terms` itself they
    would pass, at 3.0 ulp max and 0.33 mean)."""
    cfg = ModelConfig.from_card(MODELS[0])
    rng = np.random.default_rng(5)
    x = rng.standard_normal((8, cfg.hidden_size), dtype=np.float32)
    w = rng.standard_normal((cfg.hidden_size, cfg.hidden_size), dtype=np.float32) / np.float32(
        np.sqrt(cfg.hidden_size))
    ref, terms = reference_linear(x, w)
    assert_within_gate(_microinfer.linear(x, w), ref, input_rounding_floor(terms))

    x, w, _ = make_case(2, cfg.intermediate_size, 256, bias=False, seed=13)
    products = (x[:, None, :] * w[None, :, :]).astype(np.float16)
    fp16_accumulated = np.cumsum(products, axis=-1, dtype=np.float16)[..., -1].astype(np.float32)
    with pytest.raises(AssertionError):
        within_gate(fp16_accumulated, x, w)


def test_the_weight_is_out_by_in_the_bias_added_once_and_rows_independent():
    """A transposition bug on a square weight still yields the right shape:
    only a comparison against `x @ w.T` specifically catches it. The bias is
    added once per row, and each row's result is the same computed alone."""
    x, w, _ = make_case(4, 128, 128, bias=False)
    got = _microinfer.linear(x, w)
    within_gate(got, x, w)
    assert not np.allclose(got, x.astype(np.float64) @ w.astype(np.float64), atol=1e-2)

    x, w, b = make_case(5, 64, 32, bias=True)
    with_bias = _microinfer.linear(x, w, b)
    within_gate(with_bias, x, w, b)
    assert not np.array_equal(with_bias, _microinfer.linear(x, w))

    x, w, b = make_case(16, 256, 96, bias=True)
    all_rows = _microinfer.linear(x, w, b)
    for i in (0, 9, 15):
        within_gate(_microinfer.linear(x[i : i + 1].copy(), w, b), x[i : i + 1], w, b)
        within_gate(all_rows[i : i + 1], x[i : i + 1], w, b)


def test_malformed_operands_are_refused():
    x, w, b = make_case(2, 64, 32, bias=True)
    with pytest.raises(ValueError, match="64"):
        _microinfer.linear(x, w[:, :48].copy())
    with pytest.raises(ValueError, match="32"):
        _microinfer.linear(x, w, b[:16].copy())
    with pytest.raises(ValueError):
        _microinfer.linear(np.zeros(64, dtype=np.float32), w)
