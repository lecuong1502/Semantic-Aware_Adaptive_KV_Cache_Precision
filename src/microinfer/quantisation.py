"""A quantised page read on the host, and what its round trip may lose (#16, #17).

The page's layout is quant.h's; this reads it back through the offsets the
extension reports, so it cannot drift from the kernels. Shared by
tests/test_quant.py and tools/quant_roundtrip.py, so that the bound the test
asserts on two prompts is the one the tool counts over every page.

This is measurement, not inference, so it computes in NumPy on the host. The
rule that Python does no arithmetic is about the forward pass (ADR-0002,
amendment on its scope).
"""

from __future__ import annotations

import numpy as np

from . import _microinfer

P = _microinfer.device.page_tokens


def read_page(page: np.ndarray, tier, heads: int, head_dim: int) -> dict:
    """A quantised page's regions: the code width, the codes unpacked to
    (P, heads, head_dim), and the fp16 scales and zero-points in float64,
    key ones (heads, head_dim) and value ones (P, heads)."""
    layout = _microinfer.quantised_page_layout(tier, heads, head_dim)
    bits = layout["bits"]
    per_byte = 8 // bits

    def codes(offset):
        raw = page[offset:offset + P * heads * head_dim // per_byte]
        shifts = np.arange(per_byte, dtype=np.uint8) * bits
        unpacked = (raw[:, None] >> shifts) & ((1 << bits) - 1)
        return unpacked.reshape(P, heads, head_dim).astype(np.int64)

    def halves(offset, shape):
        n = int(np.prod(shape))
        return page[offset:offset + 2 * n].view(np.float16).astype(np.float64).reshape(shape)

    return {"bits": bits,
            "key_codes": codes(layout["key_codes"]),
            "value_codes": codes(layout["value_codes"]),
            "key_scales": halves(layout["key_scales"], (heads, head_dim)),
            "key_zeros": halves(layout["key_zeros"], (heads, head_dim)),
            "value_scales": halves(layout["value_scales"], (P, heads)),
            "value_zeros": halves(layout["value_zeros"], (P, heads))}


def round_trip_bound(x: np.ndarray, got: np.ndarray, scales: np.ndarray) -> np.ndarray:
    """What quantisation may lose, per element, derived rather than chosen:
    - half a step of the stored scale, which is rounded up, so that no
      element lies past the last step and none is clamped;
    - half an fp16 ulp for the dequantised value's own rounding, at the larger
      of what went in and what came out;
    - an fp32 ulp of the element, for the kernel's division and its fused
      multiply-add.

    `scales` broadcasts against `x`: the key scales as (1, heads, head_dim),
    the value scales as (P, heads, 1)."""
    larger = np.maximum(np.abs(x), np.abs(got)).astype(np.float64)
    out_ulp = np.spacing(larger.astype(np.float16)).astype(np.float64)
    return scales / 2 + out_ulp / 2 + 2.0**-23 * larger
