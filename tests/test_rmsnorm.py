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
from conftest import each
from ulp_gate import MAX_REL, assert_within_gate, fp16_exact, relative_error

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


def check(x, w, eps=EPS):
    """Both bounds from ADR-0006. The mean is the one that catches systematic
    drift, so no test may check only the max."""
    return assert_within_gate(_microinfer.rmsnorm(x, w, eps), reference_rmsnorm(x, w, eps))


# 896 is Qwen2.5-0.5B's hidden size and 1536 is Qwen2.5-1.5B's (ADR-0003). The
# others are there to catch anything hardcoded. RMSNorm reduces over the hidden
# axis and has no head_dim: Qwen2.5 has no QK-norm, so no RMSNorm in this model
# ever sees a head dimension. Issue #3's criterion naming `head_dim` is not
# satisfiable for this kernel and is raised on the ticket rather than papered
# over here.
def test_matches_float64_reference_at_every_shape():
    """The models' hidden sizes and others to catch anything hardcoded; sizes
    that are not a multiple of the warp; row counts from 1 to 512. And fp32
    input, the contract Seam B advertises with both roundings in play: a
    strict xfail while ADR-0006's mean bound stood at 2e-4 (0.41 ulp, 14%
    above one fp16 rounding's floor), now passing on its merits in ulps."""
    cases = ([(64, 896)] + [(8, h) for h in (64, 128, 896, 1536, 2048, 4096)]
             + [(4, h) for h in (1, 3, 31, 33, 100, 897)]
             + [(r, 128) for r in (1, 2, 7, 512)])
    each(cases, lambda rows, hidden: check(*make_case(rows, hidden, seed=rows * hidden)))
    each([128, 896, 1536], lambda hidden: check(*make_case_fp32(128, hidden, seed=hidden)))


def test_rows_are_independent_and_the_weight_and_eps_apply():
    """A reduction bug that leaks across rows passes a single-row test. The
    weight is applied elementwise, and eps is a parameter. The output is a
    NumPy array of the input's shape."""
    x, w = make_case(16, 256)
    all_rows = _microinfer.rmsnorm(x, w, EPS)
    for i in (0, 7, 15):
        np.testing.assert_array_equal(_microinfer.rmsnorm(x[i : i + 1].copy(), w, EPS)[0],
                                      all_rows[i])
    x, w = make_case(4, 128)
    doubled = _microinfer.rmsnorm(x, (w * 2).astype(np.float32), EPS)
    base = _microinfer.rmsnorm(x, w, EPS)
    assert relative_error(doubled, base.astype(np.float64) * 2).max() < MAX_REL
    assert not np.array_equal(base, _microinfer.rmsnorm(x, w, 1.0))
    assert isinstance(base, np.ndarray) and base.shape == x.shape


def test_malformed_operands_are_refused():
    x, w = make_case(4, 128)
    with pytest.raises(ValueError, match="128"):
        _microinfer.rmsnorm(x, w[:64].copy(), EPS)
    with pytest.raises(ValueError):
        _microinfer.rmsnorm(np.zeros(128, dtype=np.float32), w, EPS)
