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


def reference_rope(x: np.ndarray, positions: np.ndarray, theta: float,
                   inv_freq: np.ndarray | None = None):
    """x: (seq, heads, head_dim). Float64 throughout, angles included;
    `inv_freq`, if given, in place of theta's plain frequencies.

    Returns the rotated values and, per output, the magnitude of the two terms
    summed to make it (`ulp_gate`): `x1*cos - x2*sin` cancels, and near a
    cancellation only the terms say how precise the result can be."""
    head_dim = x.shape[-1]
    half = head_dim // 2
    x64 = x.astype(np.float64)
    if inv_freq is None:
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


# -- llama3 scaling (#209) ----------------------------------------------------

#: Llama-3.2's rope_scaling and rope_theta. Literals until its card is
#: committed with the checkpoint (ADR-0003: verified at download), then read
#: from the card like every other model's constants.
LLAMA3 = {"factor": 32.0, "low_freq_factor": 1.0, "high_freq_factor": 4.0,
          "original_max_position_embeddings": 8192, "rope_type": "llama3"}
LLAMA3_THETA = 500000.0
LLAMA3_HEAD_DIM = 64


def reference_inv_freq(head_dim: int, theta: float, scaling: dict | None) -> np.ndarray:
    """HuggingFace's `_compute_llama3_parameters`, in float64: frequencies
    whose wavelength exceeds the original context / low_freq_factor are
    divided by `factor`, those under the original context / high_freq_factor
    kept, and those between blended linearly in original context / wavelength."""
    inv_freq = theta ** (-np.arange(head_dim // 2, dtype=np.float64) * 2.0 / head_dim)
    if scaling is None:
        return inv_freq
    factor, low, high = scaling["factor"], scaling["low_freq_factor"], scaling["high_freq_factor"]
    old = float(scaling["original_max_position_embeddings"])
    wavelen = 2 * np.pi / inv_freq
    scaled = np.where(wavelen > old / low, inv_freq / factor, inv_freq)
    smooth = (old / wavelen - low) / (high - low)
    blended = (1 - smooth) * scaled / factor + smooth * scaled
    medium = ~(wavelen < old / high) & ~(wavelen > old / low)
    return np.where(medium, blended, scaled)


def llama3_scaling() -> "_microinfer.RopeScaling":
    return _microinfer.RopeScaling(
        factor=LLAMA3["factor"], low_freq_factor=LLAMA3["low_freq_factor"],
        high_freq_factor=LLAMA3["high_freq_factor"],
        original_max_position_embeddings=LLAMA3["original_max_position_embeddings"])


def test_the_reference_scales_low_frequencies_keeps_high_ones_and_blends_between():
    """The reference's own shape, so that a reference wrong in the same way
    as the kernel cannot pass them both: at Llama-3.2's head_dim the fastest
    frequency is kept, the slowest divided by 32, and some lie between, each
    strictly between its two bounds."""
    plain = reference_inv_freq(LLAMA3_HEAD_DIM, LLAMA3_THETA, None)
    scaled = reference_inv_freq(LLAMA3_HEAD_DIM, LLAMA3_THETA, LLAMA3)
    assert scaled[0] == plain[0]
    assert scaled[-1] == pytest.approx(plain[-1] / 32, rel=1e-15)
    between = (scaled < plain) & (scaled > plain / 32)
    assert between.any() and (scaled <= plain).all() and (scaled >= plain / 32).all()


def test_llama3_scaling_matches_the_reference():
    """The rope op with Llama-3.2's scaling against the float64 reference, at
    its head_dim, its query and key head counts, and positions to the end of
    its window; and the table attention reads, which is the same angles.
    Without the scaling, the same op gives the plain angles still."""
    inv_freq = reference_inv_freq(LLAMA3_HEAD_DIM, LLAMA3_THETA, LLAMA3)
    positions = np.array([0, 1, 31, 2047, 2049, 8191, 8192, 16384, 32767, 131071],
                         dtype=np.int32)

    def heads(n):
        x = make_x(len(positions), n, LLAMA3_HEAD_DIM, seed=n)
        got = _microinfer.rope(x, positions, LLAMA3_THETA, llama3_scaling())
        ref, terms = reference_rope(x, positions, LLAMA3_THETA, inv_freq)
        assert_within_gate(got, ref, accumulation_floor(terms, 2))

    each([32, 8], heads)

    x = make_x(len(positions), 8, LLAMA3_HEAD_DIM)
    plain = _microinfer.rope(x, positions, LLAMA3_THETA)
    ref, terms = reference_rope(x, positions, LLAMA3_THETA)
    assert_within_gate(plain, ref, accumulation_floor(terms, 2))
    assert not np.array_equal(plain, _microinfer.rope(x, positions, LLAMA3_THETA,
                                                      llama3_scaling()))

    table = _microinfer.device.RopeTable(LLAMA3_HEAD_DIM, LLAMA3_THETA, llama3_scaling())
    table.cover(int(positions.max()) + 1)
    got = table.to_numpy()[positions]                     # (positions, half, {cos, sin})
    angle = positions.astype(np.float64)[:, None] * inv_freq[None, :]
    np.testing.assert_allclose(got[..., 0], np.cos(angle), atol=2e-7)
    np.testing.assert_allclose(got[..., 1], np.sin(angle), atol=2e-7)


def test_malformed_scaling_is_refused():
    """A factor below one, high_freq_factor not above low_freq_factor (the
    blend divides by their difference), and no original window."""
    for bad in ({"factor": 0.5}, {"high_freq_factor": 1.0}, {"low_freq_factor": 5.0},
                {"original_max_position_embeddings": 0}):
        kwargs = {k: v for k, v in LLAMA3.items() if k != "rope_type"} | bad
        with pytest.raises(ValueError):
            _microinfer.RopeScaling(**kwargs)


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


def test_a_config_s_rope_scaling_reaches_the_kernel():
    """What the forward pass hands the kernels: none for Qwen2.5, llama3's
    four numbers for Llama-3.2."""
    from test_config import LLAMA

    from microinfer.model import rope_scaling

    assert not rope_scaling(ModelConfig.from_card(MODELS[0])).scaled
    s = rope_scaling(ModelConfig.from_dict("llama-3.2-1b-instruct", LLAMA))
    assert s.scaled and (s.factor, s.low_freq_factor, s.high_freq_factor,
                         s.original_max_position_embeddings) == (32.0, 1.0, 4.0, 8192.0)
