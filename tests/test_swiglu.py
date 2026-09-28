"""SwiGLU against a float64 NumPy reference.

The MLP's activation: `silu(gate) * up`, where silu(g) = g * sigmoid(g). The two
projections feeding it are cuBLAS calls (test_linear.py); this kernel is only
the elementwise combination between them and `down_proj`.

The product is multiplicative, so there is no cancellation for a relative error
measure to trip over: the gate from ADR-0006 applies unmodified.
"""

import numpy as np
import pytest
from conftest import each
from ulp_gate import assert_within_gate, fp16_exact

from microinfer import _microinfer
from microinfer.config import ModelConfig
from microinfer.models import VERIFIED

MODELS = sorted(VERIFIED)


def reference_swiglu(gate: np.ndarray, up: np.ndarray) -> np.ndarray:
    g = gate.astype(np.float64)
    return g / (1.0 + np.exp(-g)) * up.astype(np.float64)


def make_case(rows: int, width: int, seed: int = 0, scale: float = 1.0):
    rng = np.random.default_rng(seed)
    gate = fp16_exact(scale * rng.standard_normal((rows, width)))
    up = fp16_exact(scale * rng.standard_normal((rows, width)))
    return gate, up


def test_matches_float64_reference_across_widths_and_gate_values():
    """The kernel is SwiGLU, not a generic GLU: every card names silu, or this
    is the wrong kernel and should fail here rather than downstream. At both
    models' widths; across silu's regimes, nearly linear near zero, identity
    for large positive gates and zero for large negative ones, each a
    different way to get the sigmoid wrong; at widths that are not a
    multiple of anything; and on fp32 input."""
    assert all(ModelConfig.from_card(name).hidden_act == "silu" for name in MODELS)

    def matches(gate, up):
        assert_within_gate(_microinfer.swiglu(gate, up), reference_swiglu(gate, up))

    width = ModelConfig.from_card(MODELS[0]).intermediate_size
    cases = ([make_case(16, ModelConfig.from_card(n).intermediate_size,
                        seed=ModelConfig.from_card(n).intermediate_size) for n in MODELS]
             + [make_case(8, width, seed=int(s * 100), scale=s) for s in (0.01, 1.0, 8.0)]
             + [make_case(3, w, seed=w) for w in (1, 31, 33, 1000, 4865)])
    rng = np.random.default_rng(3)
    cases.append(tuple(rng.standard_normal((16, width)).astype(np.float32) for _ in range(2)))
    each(cases, matches, name=lambda c: f"gate {c[0].shape}")


def test_the_arguments_are_gate_then_up_and_large_gates_stay_finite():
    """An argument swap is the likeliest bug here and a symmetric test will
    not see it. exp(-g) overflows fp32 for g < -88, and the result must still
    be a finite value near zero, not NaN from inf/inf."""
    gate, up = make_case(4, 256)
    assert_within_gate(_microinfer.swiglu(gate, up), reference_swiglu(gate, up))
    assert not np.allclose(_microinfer.swiglu(up, gate), reference_swiglu(gate, up))

    gate = np.array([[-100.0, -60000.0, 0.0, 60000.0]], dtype=np.float32)
    got = _microinfer.swiglu(gate, np.ones_like(gate))
    assert np.all(np.isfinite(got))
    np.testing.assert_array_equal(got[0, :3], [0.0, 0.0, 0.0])
    assert got[0, 3] == np.float16(60000.0)


def test_malformed_operands_are_refused():
    gate, up = make_case(4, 256)
    with pytest.raises(ValueError, match="256"):
        _microinfer.swiglu(gate, up[:, :128].copy())
    with pytest.raises(ValueError):
        _microinfer.swiglu(np.zeros(8, dtype=np.float32), np.zeros(8, dtype=np.float32))
