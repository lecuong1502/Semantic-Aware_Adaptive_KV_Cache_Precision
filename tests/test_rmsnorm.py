"""RMSNorm against a float64 NumPy reference.

Tolerances come from ADR-0006, stated in ulps of the kernel's output format:
max relative error < 4 ulp, mean < 1 ulp. This kernel writes fp16, whose largest
relative ulp is 2^-11.

**There are two roundings, not one.** The kernel's domain is fp16, which is what
the engine stores, but Seam B's surface is fp32 NumPy. So arbitrary fp32 input
is rounded on the way in, and the result is rounded again on store. The kernel
itself accumulates in fp32 and contributes no error of its own beyond that.

Both cases are measured. `make_case` uses fp16-exact inputs, which isolates the
kernel by removing the rounding the *caller* caused; `make_case_fp32` uses the
contract Seam B actually advertises. Both are held to the same gate, which lives
in `ulp_gate`.
"""

import numpy as np
import pytest
from ulp_gate import MAX_REL, fp16_exact, relative_error
from ulp_gate import assert_within_gate as assert_gate

from microinfer import _microinfer

EPS = 1e-6


def reference_rmsnorm(x: np.ndarray, w: np.ndarray, eps: float) -> np.ndarray:
    """Independent implementation in float64. Deliberately not the kernel's shape."""
    x64 = x.astype(np.float64)
    ms = np.mean(x64 * x64, axis=-1, keepdims=True)
    return (x64 / np.sqrt(ms + eps)) * w.astype(np.float64)


def make_case(rows: int, hidden: int, seed: int = 0):
    """fp16-exact inputs: isolates the kernel from the caller's input rounding."""
    rng = np.random.default_rng(seed)
    x = fp16_exact(rng.standard_normal((rows, hidden)))
    w = fp16_exact(rng.uniform(0.5, 1.5, hidden))
    return x, w


def make_case_fp32(rows: int, hidden: int, seed: int = 0):
    """Arbitrary fp32 inputs: the contract Seam B actually advertises."""
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((rows, hidden)).astype(np.float32)
    w = rng.uniform(0.5, 1.5, hidden).astype(np.float32)
    return x, w


def assert_within_gate(x, w, eps=EPS):
    """Both bounds from ADR-0006. The mean is the one that catches systematic
    drift, so no test may check only the max."""
    return assert_gate(_microinfer.rmsnorm(x, w, eps), reference_rmsnorm(x, w, eps))


def test_matches_float64_reference():
    assert_within_gate(*make_case(64, 896))


# 896 is Qwen2.5-0.5B's hidden size and 1536 is Qwen2.5-1.5B's (ADR-0003). The
# others are there to catch anything hardcoded. RMSNorm reduces over the hidden
# axis and has no head_dim: Qwen2.5 has no QK-norm, so no RMSNorm in this model
# ever sees a head dimension. Issue #3's criterion naming `head_dim` is not
# satisfiable for this kernel and is raised on the ticket rather than papered
# over here.
@pytest.mark.parametrize("hidden", [64, 128, 896, 1536, 2048, 4096])
def test_hidden_size_is_a_parameter(hidden):
    assert_within_gate(*make_case(8, hidden, seed=hidden))


@pytest.mark.parametrize("hidden", [1, 3, 31, 33, 100, 897])
def test_hidden_size_need_not_be_a_multiple_of_the_warp(hidden):
    assert_within_gate(*make_case(4, hidden, seed=hidden))


@pytest.mark.parametrize("rows", [1, 2, 7, 512])
def test_row_count_is_a_parameter(rows):
    assert_within_gate(*make_case(rows, 128, seed=rows))


@pytest.mark.parametrize("hidden", [128, 896, 1536])
def test_arbitrary_fp32_input_meets_the_gate(hidden):
    """The contract Seam B advertises, with both roundings in play.

    This was a strict xfail while ADR-0006's mean bound stood at 2e-4, which is
    0.41 ulp — only 14% above the floor a single fp16 rounding produces, and so
    unreachable for a value rounded twice. The ADR is now stated in ulps and the
    case passes on its merits.
    """
    assert_within_gate(*make_case_fp32(128, hidden, seed=hidden))


def test_rows_are_independent():
    """A reduction bug that leaks across rows passes a single-row test."""
    x, w = make_case(16, 256)
    all_rows = _microinfer.rmsnorm(x, w, EPS)
    for i in (0, 7, 15):
        one = _microinfer.rmsnorm(x[i : i + 1].copy(), w, EPS)
        np.testing.assert_array_equal(one[0], all_rows[i])


def test_weight_is_applied_elementwise():
    x, w = make_case(4, 128)
    doubled = _microinfer.rmsnorm(x, (w * 2).astype(np.float32), EPS)
    base = _microinfer.rmsnorm(x, w, EPS)
    err = relative_error(doubled, base.astype(np.float64) * 2)
    assert err.max() < MAX_REL


def test_eps_is_a_parameter():
    x, w = make_case(4, 128)
    assert not np.array_equal(
        _microinfer.rmsnorm(x, w, EPS), _microinfer.rmsnorm(x, w, 1.0)
    )


def test_returns_a_numpy_array_of_the_input_shape():
    x, w = make_case(5, 64)
    out = _microinfer.rmsnorm(x, w, EPS)
    assert isinstance(out, np.ndarray)
    assert out.shape == x.shape


def test_rejects_mismatched_weight_length():
    x, w = make_case(4, 128)
    with pytest.raises(ValueError, match="128"):
        _microinfer.rmsnorm(x, w[:64].copy(), EPS)


def test_rejects_non_2d_input():
    _, w = make_case(4, 128)
    with pytest.raises(ValueError):
        _microinfer.rmsnorm(np.zeros(128, dtype=np.float32), w, EPS)
