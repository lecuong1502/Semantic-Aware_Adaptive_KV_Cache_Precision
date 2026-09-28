"""RoPE against a float64 NumPy reference.

Qwen2.5 uses plain RoPE: no scaling, and the "rotate half" pairing, in which
dimension `j` rotates with dimension `j + head_dim/2` rather than with its
neighbour. Getting the pairing wrong still produces a rotation, still preserves
norms, and still passes any test that only checks those — so the reference here
fixes the pairing by construction and the kernel is compared value by value.

The reference is written as complex multiplication, deliberately not in the
kernel's cos/sin shape: `(x1 + i*x2) * exp(i * pos * inv_freq)`, whose real and
imaginary parts are the two halves of the output.

Every dimension and theta in a test of correctness comes from the model cards,
never a literal (issue #7). The input-validation tests at the end use literals,
because what they check is a shape the model never has.
"""

import numpy as np
import pytest
from conftest import each
from ulp_gate import accumulation_floor, assert_within_gate, fp16_exact, input_rounding_floor

from microinfer import _microinfer
from microinfer.config import ModelConfig
from microinfer.models import VERIFIED

MODELS = sorted(VERIFIED)


def positions_for(cfg: ModelConfig) -> np.ndarray:
    """0 is the identity, and the ticket asks for positions past 2048.

    The far end is what makes this list worth having. An fp32 angle
    `pos * inv_freq` carries an absolute error that grows with the position.
    Simulated at the fastest frequency (test_the_gate_would_reject_an_fp32_angle),
    it costs 18 ulp at 2049 and 60 ulp at max_position_embeddings - 1, the last
    position the model ever sees. Near zero it costs nothing, which is why a
    kernel that gets this wrong passes a test of small positions."""
    return np.array([0, 1, 2, 31, 32, 1000, 2047, 2048, 2049, 4097, 16384,
                     cfg.max_position_embeddings - 1], dtype=np.int32)


def reference_rope(x: np.ndarray, positions: np.ndarray, theta: float):
    """x: (seq, heads, head_dim). Float64 throughout, angles included.

    Returns the rotated values and, per output, the magnitude of the two terms
    summed to make it (`ulp_gate`): `x1*cos - x2*sin` cancels, and near a
    cancellation only the terms say how precise the result can be."""
    head_dim = x.shape[-1]
    half = head_dim // 2
    x64 = x.astype(np.float64)
    inv_freq = theta ** (-np.arange(half, dtype=np.float64) * 2.0 / head_dim)
    angle = positions.astype(np.float64)[:, None] * inv_freq[None, :]  # (seq, half)
    rot = np.exp(1j * angle)[:, None, :]
    x1, x2 = x64[..., :half], x64[..., half:]
    z = (x1 + 1j * x2) * rot
    c, s = np.abs(rot.real), np.abs(rot.imag)
    terms = np.concatenate([np.abs(x1) * c + np.abs(x2) * s, np.abs(x2) * c + np.abs(x1) * s], axis=-1)
    return np.concatenate([z.real, z.imag], axis=-1), terms


def check(x: np.ndarray, positions: np.ndarray, theta: float) -> np.ndarray:
    """The gate for fp16-exact input. Each output is a sum of two terms."""
    got = _microinfer.rope(x, positions, theta)
    ref, terms = reference_rope(x, positions, theta)
    assert_within_gate(got, ref, accumulation_floor(terms, 2))
    return got


def make_x(seq: int, heads: int, head_dim: int, seed: int = 0) -> np.ndarray:
    return fp16_exact(np.random.default_rng(seed).standard_normal((seq, heads, head_dim)))


def test_matches_float64_reference_at_every_shape_and_position():
    """At both models' shapes, queries and keys alike though their head counts
    differ; and each position on its own, since a pooled gate lets one bad
    position hide among good ones: at 12 positions a single wholly wrong row
    moves the mean by only a twelfth. Positions need not be contiguous or
    sorted, as decode rotates one token at any position; theta and head_dim
    are parameters. Position 0 is the identity, and the input is left as it
    was."""
    cfgs = [ModelConfig.from_card(n) for n in MODELS]

    def shape(cfg, heads):
        positions = positions_for(cfg)
        check(make_x(len(positions), heads, cfg.head_dim, seed=heads), positions,
              cfg.rope_theta)

    def one_position(cfg, index):
        pos = positions_for(cfg)[index:index + 1]
        check(make_x(1, cfg.num_attention_heads, cfg.head_dim, seed=int(pos[0])), pos,
              cfg.rope_theta)

    each([(shape, c, h) for c in cfgs for h in (c.num_attention_heads, c.num_key_value_heads)]
         + [(one_position, c, i) for c in cfgs for i in range(12)],
         lambda f, c, arg: f(c, arg), name=lambda c: f"{c[0].__name__} {c[1].name} {c[2]}")

    cfg = cfgs[0]
    positions = np.array([4097, 3, 3, 0, 2048], dtype=np.int32)
    check(make_x(len(positions), cfg.num_key_value_heads, cfg.head_dim), positions,
          cfg.rope_theta)
    positions = np.array([5, 3000], dtype=np.int32)
    x = make_x(2, cfg.num_key_value_heads, cfg.head_dim)
    other = check(x, positions, cfg.rope_theta / 100)
    assert not np.array_equal(other, _microinfer.rope(x, positions, cfg.rope_theta))
    each([2, 32, 64, 96, 128, 256], lambda head_dim: check(
        make_x(3, 3, head_dim, seed=head_dim), np.array([0, 17, 2049], dtype=np.int32),
        cfg.rope_theta))

    x = make_x(1, cfg.num_attention_heads, cfg.head_dim)
    np.testing.assert_array_equal(
        _microinfer.rope(x, np.zeros(1, dtype=np.int32), cfg.rope_theta), x)
    x = make_x(2, cfg.num_key_value_heads, cfg.head_dim)
    before = x.copy()
    _microinfer.rope(x, np.array([9, 4000], dtype=np.int32), cfg.rope_theta)
    np.testing.assert_array_equal(x, before)


def test_the_gate_holds_fp32_input_and_would_reject_an_fp32_angle():
    """The contract Seam B advertises: input rounded to fp16, output rounded.
    Unlike RMSNorm, not to the fp16-exact floor: rounding x1 and x2 moves
    `x1*cos - x2*sin` by up to half an ulp of its terms, and where the two
    cancel that is a relative error no kernel could avoid (measured, 1084 ulp
    at an output of -9.7e-5 whose terms sum to 1.64).

    And the far positions are only worth testing if the gate can see the error
    they exist to expose: the obvious fp32 implementation, inv_freq and angle
    both in fp32, is simulated and fails."""
    for name in MODELS:
        cfg = ModelConfig.from_card(name)
        positions = positions_for(cfg)
        x = np.random.default_rng(1).standard_normal(
            (len(positions), cfg.num_attention_heads, cfg.head_dim)).astype(np.float32)
        ref, terms = reference_rope(x, positions, cfg.rope_theta)
        assert_within_gate(_microinfer.rope(x, positions, cfg.rope_theta), ref,
                           input_rounding_floor(terms))

    cfg = ModelConfig.from_card(MODELS[0])
    pos = np.array([cfg.max_position_embeddings - 1], dtype=np.int32)
    x = make_x(1, cfg.num_attention_heads, cfg.head_dim, seed=3)
    half = cfg.head_dim // 2
    inv_freq = (np.float32(cfg.rope_theta)
                ** (-np.arange(half, dtype=np.float32) * np.float32(2.0 / cfg.head_dim)))
    angle = (pos.astype(np.float32)[:, None] * inv_freq).astype(np.float64)
    z = (x[..., :half] + 1j * x[..., half:]).astype(np.complex128) * np.exp(1j * angle)[:, None, :]
    fp32_angle = fp16_exact(np.concatenate([z.real, z.imag], axis=-1))
    ref, terms = reference_rope(x, pos, cfg.rope_theta)
    with pytest.raises(AssertionError):
        assert_within_gate(fp32_angle, ref, accumulation_floor(terms, 2))


def test_attention_score_depends_only_on_relative_position():
    """The property RoPE exists for: <R(m)q, R(n)k> is a function of m - n."""
    cfg = ModelConfig.from_card(MODELS[0])
    rng = np.random.default_rng(7)
    q = fp16_exact(rng.standard_normal((1, 1, cfg.head_dim)))
    k = fp16_exact(rng.standard_normal((1, 1, cfg.head_dim)))

    def score(m, n):
        rq = _microinfer.rope(q, np.array([m], dtype=np.int32), cfg.rope_theta)
        rk = _microinfer.rope(k, np.array([n], dtype=np.int32), cfg.rope_theta)
        return float(np.dot(rq.ravel().astype(np.float64), rk.ravel()))

    base = score(10, 3)
    for shift in (100, 2048, 5000):
        assert score(10 + shift, 3 + shift) == pytest.approx(base, rel=1e-2, abs=1e-2)


def test_malformed_input_is_refused():
    """Positions that do not match the sequence; an odd head_dim, which
    rotate-half cannot pair (j with j + head_dim/2); negative positions; input
    that is not (seq, heads, head_dim)."""
    with pytest.raises(ValueError, match="3"):
        _microinfer.rope(make_x(3, 2, 64), np.zeros(2, dtype=np.int32), 1e6)
    with pytest.raises(ValueError, match="even"):
        _microinfer.rope(np.zeros((1, 1, 63), dtype=np.float32),
                         np.zeros(1, dtype=np.int32), 1e6)
    with pytest.raises(ValueError, match="negative"):
        _microinfer.rope(make_x(1, 1, 64), np.array([-1], dtype=np.int32), 1e6)
    with pytest.raises(ValueError):
        _microinfer.rope(np.zeros((2, 64), dtype=np.float32),
                         np.zeros(2, dtype=np.int32), 1e6)
