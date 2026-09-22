"""RMSNorm against a float64 NumPy reference.

Tolerances come from ADR-0006: max relative error < 2e-3, mean < 2e-4.

On measuring relative error: the kernel accumulates in fp32 and stores its
output in fp16, so the dominant error is the final rounding to fp16, which is
*relative* for normal values and bounded by 2^-11 ~ 4.9e-4. Below fp16's
smallest normal (6.104e-5) values are subnormal and their error is bounded in
absolute terms instead, where a relative measure is meaningless. The denominator
is therefore floored at that value. This is a property of the format, not a
concession to the implementation.
"""

import numpy as np
import pytest

from microinfer import _microinfer

FP16_MIN_NORMAL = 6.103515625e-05
MAX_REL = 2e-3
MEAN_REL = 2e-4


def reference_rmsnorm(x: np.ndarray, w: np.ndarray, eps: float) -> np.ndarray:
    """Independent implementation in float64. Deliberately not the kernel's shape."""
    x64 = x.astype(np.float64)
    ms = np.mean(x64 * x64, axis=-1, keepdims=True)
    return (x64 / np.sqrt(ms + eps)) * w.astype(np.float64)


def relative_error(got: np.ndarray, ref: np.ndarray) -> np.ndarray:
    return np.abs(got.astype(np.float64) - ref) / np.maximum(np.abs(ref), FP16_MIN_NORMAL)


def make_case(rows: int, hidden: int, seed: int = 0):
    """Inputs are fp16-exact, so the reference measures only the kernel's error."""
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((rows, hidden)).astype(np.float16).astype(np.float32)
    w = rng.uniform(0.5, 1.5, hidden).astype(np.float16).astype(np.float32)
    return x, w


def test_matches_float64_reference():
    x, w = make_case(64, 896)
    got = _microinfer.rmsnorm(x, w, 1e-6)
    err = relative_error(got, reference_rmsnorm(x, w, 1e-6))
    assert err.max() < MAX_REL, f"max relative error {err.max():.3e}"
    assert err.mean() < MEAN_REL, f"mean relative error {err.mean():.3e}"


@pytest.mark.parametrize("hidden", [64, 128, 896, 1536, 2048, 4096])
def test_hidden_size_is_a_parameter(hidden):
    """0.5B uses head_dim 64 and hidden 896; 1.5B uses 128 and 1536 (ADR-0003).
    Nothing may be hardcoded to either."""
    x, w = make_case(8, hidden, seed=hidden)
    err = relative_error(_microinfer.rmsnorm(x, w, 1e-6), reference_rmsnorm(x, w, 1e-6))
    assert err.max() < MAX_REL


@pytest.mark.parametrize("hidden", [1, 3, 31, 33, 100, 897])
def test_hidden_size_need_not_be_a_multiple_of_the_warp(hidden):
    x, w = make_case(4, hidden, seed=hidden)
    err = relative_error(_microinfer.rmsnorm(x, w, 1e-6), reference_rmsnorm(x, w, 1e-6))
    assert err.max() < MAX_REL


@pytest.mark.parametrize("rows", [1, 2, 7, 512])
def test_row_count_is_a_parameter(rows):
    x, w = make_case(rows, 128, seed=rows)
    err = relative_error(_microinfer.rmsnorm(x, w, 1e-6), reference_rmsnorm(x, w, 1e-6))
    assert err.max() < MAX_REL


def test_rows_are_independent():
    """A reduction bug that leaks across rows passes a single-row test."""
    x, w = make_case(16, 256)
    all_rows = _microinfer.rmsnorm(x, w, 1e-6)
    for i in (0, 7, 15):
        one = _microinfer.rmsnorm(x[i : i + 1].copy(), w, 1e-6)
        np.testing.assert_array_equal(one[0], all_rows[i])


def test_weight_is_applied_elementwise():
    x, w = make_case(4, 128)
    doubled = _microinfer.rmsnorm(x, (w * 2).astype(np.float32), 1e-6)
    base = _microinfer.rmsnorm(x, w, 1e-6)
    err = relative_error(doubled, base.astype(np.float64) * 2)
    assert err.max() < MAX_REL


def test_eps_is_a_parameter():
    x, w = make_case(4, 128)
    assert not np.array_equal(
        _microinfer.rmsnorm(x, w, 1e-6), _microinfer.rmsnorm(x, w, 1.0)
    )


def test_returns_a_numpy_array_of_the_input_shape():
    x, w = make_case(5, 64)
    out = _microinfer.rmsnorm(x, w, 1e-6)
    assert isinstance(out, np.ndarray)
    assert out.shape == x.shape


def test_rejects_mismatched_weight_length():
    x, w = make_case(4, 128)
    with pytest.raises(ValueError, match="128"):
        _microinfer.rmsnorm(x, w[:64].copy(), 1e-6)


def test_rejects_non_2d_input():
    _, w = make_case(4, 128)
    with pytest.raises(ValueError):
        _microinfer.rmsnorm(np.zeros(128, dtype=np.float32), w, 1e-6)
