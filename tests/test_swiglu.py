"""SwiGLU against a float64 NumPy reference.

The MLP's activation: `silu(gate) * up`, where silu(g) = g * sigmoid(g). The two
projections feeding it are cuBLAS calls (test_linear.py); this kernel is only
the elementwise combination between them and `down_proj`.

The product is multiplicative, so there is no cancellation for a relative error
measure to trip over: the gate from ADR-0006 applies unmodified.
"""

import numpy as np
import pytest
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


def test_every_model_uses_silu():
    """The kernel is SwiGLU, not a generic GLU. If a card ever names another
    activation this kernel is the wrong one, and should fail loudly here rather
    than numerically somewhere downstream."""
    for name in MODELS:
        assert ModelConfig.from_card(name).hidden_act == "silu"


@pytest.mark.parametrize("name", MODELS)
def test_matches_float64_reference_at_model_width(name):
    cfg = ModelConfig.from_card(name)
    gate, up = make_case(16, cfg.intermediate_size, seed=cfg.intermediate_size)
    assert_within_gate(_microinfer.swiglu(gate, up), reference_swiglu(gate, up))


@pytest.mark.parametrize("scale", [0.01, 1.0, 8.0])
def test_holds_across_the_range_of_gate_values(scale):
    """silu is nearly linear near zero, saturates to identity for large positive
    gates, and decays to zero for large negative ones. Each regime is a
    different way to get the sigmoid wrong."""
    width = ModelConfig.from_card(MODELS[0]).intermediate_size
    gate, up = make_case(8, width, seed=int(scale * 100), scale=scale)
    assert_within_gate(_microinfer.swiglu(gate, up), reference_swiglu(gate, up))


def test_large_negative_gates_do_not_overflow():
    """exp(-g) overflows fp32 for g < -88. The result must still be a finite
    value near zero, not NaN from inf/inf."""
    gate = np.array([[-100.0, -60000.0, 0.0, 60000.0]], dtype=np.float32)
    up = np.ones_like(gate)
    got = _microinfer.swiglu(gate, up)
    assert np.all(np.isfinite(got))
    np.testing.assert_array_equal(got[0, :3], [0.0, 0.0, 0.0])
    assert got[0, 3] == np.float16(60000.0)


def test_arbitrary_fp32_input_meets_the_gate():
    width = ModelConfig.from_card(MODELS[0]).intermediate_size
    rng = np.random.default_rng(3)
    gate = rng.standard_normal((16, width)).astype(np.float32)
    up = rng.standard_normal((16, width)).astype(np.float32)
    assert_within_gate(_microinfer.swiglu(gate, up), reference_swiglu(gate, up))


@pytest.mark.parametrize("width", [1, 31, 33, 1000, 4865])
def test_width_need_not_be_a_multiple_of_anything(width):
    gate, up = make_case(3, width, seed=width)
    assert_within_gate(_microinfer.swiglu(gate, up), reference_swiglu(gate, up))


def test_gate_and_up_are_not_interchangeable():
    """An argument swap is the likeliest bug here and a symmetric test will not
    see it."""
    gate, up = make_case(4, 256)
    assert_within_gate(_microinfer.swiglu(gate, up), reference_swiglu(gate, up))
    assert not np.allclose(_microinfer.swiglu(up, gate), reference_swiglu(gate, up))


def test_rejects_mismatched_shapes():
    gate, up = make_case(4, 256)
    with pytest.raises(ValueError, match="256"):
        _microinfer.swiglu(gate, up[:, :128].copy())


def test_rejects_non_2d_input():
    with pytest.raises(ValueError):
        _microinfer.swiglu(np.zeros(8, dtype=np.float32), np.zeros(8, dtype=np.float32))
