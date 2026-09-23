"""ADR-0006's per-kernel gate, shared by every kernel test.

The tolerances are stated in ulps of the kernel's *output* format: max relative
error < 4 ulp, mean < 1 ulp. Every kernel so far writes fp16, whose largest
relative ulp is 2^-11. Kept in one place so that four kernel tests cannot drift
into four slightly different readings of one ADR.

On the error measure: rounding to fp16 is relative for normal values and bounded
by 2^-11 ~ 4.9e-4. Below fp16's smallest normal (6.104e-5) values are subnormal
and the error is bounded in absolute terms instead, where a relative measure is
meaningless. The denominator is floored there. That is a property of the format,
not a concession to the implementation.

**Kernels whose output is a sum** — a projection's dot product, RoPE's
`x1*cos - x2*sin` — need a second floor (ADR-0006, amendment on cancellation).
A sum's error is bounded by the magnitude of its *terms*, not of its result, so
where the terms cancel a result near zero carries an error no correct kernel
can avoid, and dividing by the result measures the cancellation. Such a test
passes `floor`, one value per output element, from one of the two helpers
below; each names the error source it admits and nothing else. A product-shaped
kernel (RMSNorm, SwiGLU) has no sum and passes no floor.
"""

import numpy as np

FP16_MIN_NORMAL = 6.103515625e-05

#: Largest relative ulp of fp16. Derived, never hardcoded: a future kernel
#: writing fp32 inherits a proportionally tighter bound automatically.
FP16_REL_ULP = 2.0**-11  # 4.8828125e-04

MAX_REL = 4 * FP16_REL_ULP  # 1.953e-03
MEAN_REL = 1 * FP16_REL_ULP  # 4.883e-04

#: fp32's counterpart of FP16_REL_ULP, for kernels that accumulate in fp32.
FP32_REL_ULP = 2.0**-24


def accumulation_floor(terms: np.ndarray, n: int) -> np.ndarray:
    """For fp16-exact inputs: the error an fp32 sum of `n` terms incurs.

    `terms` is, per output, the sum of the absolute values of what was added.
    fp32 accumulation over `n` of them errs by about sqrt(n) * u32 * terms (the
    probabilistic bound of Higham & Mary, 2019; the worst case n * u32 is loose
    by orders of magnitude). The floor is the magnitude at which that error
    uses the whole max budget, MAX_REL, and no more: accumulation is allowed to
    be the worst thing about an output, but not to exceed the gate on its own.

    Calibrated rather than assumed. On the model's projections the observed
    error is at most 0.19 of the bound, and the floor replaces |y| for 1.4% of
    outputs at n = 896 and 13.7% at n = 8960. Stating the floor against one ulp
    instead of four would have covered 51% at n = 8960, the review of #7
    measured, and at that point it is the floor, not the result, being judged.

    Deliberately not `terms` itself. A dot product's terms outweigh its result
    by ~sqrt(n), so measuring against them would forgive an fp16 accumulator;
    test_linear.py shows this floor does not."""
    return np.sqrt(n) * FP32_REL_ULP * terms / MAX_REL


def input_rounding_floor(terms: np.ndarray) -> np.ndarray:
    """For arbitrary fp32 inputs: the error the caller's own downcast causes.

    Each operand is rounded to fp16 on the way in, moving it by up to half an
    fp16 ulp; the result then moves by up to half an fp16 ulp of `terms`,
    however the kernel is written. So the error is measured against the terms.
    This is the looser of the two floors by design. The kernel is judged by its
    fp16-exact test; the fp32 test checks only that Seam B's downcast costs what
    it should."""
    return terms


def relative_error(got: np.ndarray, ref: np.ndarray, floor=None) -> np.ndarray:
    floor = FP16_MIN_NORMAL if floor is None else np.maximum(FP16_MIN_NORMAL, floor)
    return np.abs(got.astype(np.float64) - ref) / np.maximum(np.abs(ref), floor)


def assert_within_gate(got: np.ndarray, ref: np.ndarray, floor=None) -> np.ndarray:
    """Both bounds. The mean is the one that catches systematic drift, so no
    test may check only the max."""
    assert got.shape == ref.shape, f"shape {got.shape} != reference {ref.shape}"
    err = relative_error(got, ref, floor)
    assert err.max() < MAX_REL, f"max relative error {err.max():.3e} ({err.max() / FP16_REL_ULP:.2f} ulp)"
    assert err.mean() < MEAN_REL, f"mean relative error {err.mean():.3e} ({err.mean() / FP16_REL_ULP:.2f} ulp)"
    return err


def fp16_exact(a: np.ndarray) -> np.ndarray:
    """Round through fp16 and back, so the kernel's input downcast is lossless.

    Isolates the kernel from the rounding the *caller* would otherwise cause by
    handing fp32 to a kernel whose domain is fp16."""
    return a.astype(np.float16).astype(np.float32)
