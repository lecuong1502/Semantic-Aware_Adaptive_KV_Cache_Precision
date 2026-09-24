"""One device allocation for all the weights (#23, Seam B).

Weights used to be uploaded one cudaMalloc per tensor, 290 of them for
Qwen2.5-0.5B, and the driver's overhead tracks the number of allocations,
not their size (ADR-0007, notes from #5 and #23). A DeviceArena is one
allocation; each weight is a DeviceTensor that refers into it at an offset,
and owns nothing. The arena goes back to the driver when the last tensor
referring into it goes.

Every tensor starts on a multiple of weight_alignment bytes, which is what
cudaMalloc guaranteed each tensor before. So every kernel that reads a
weight sees the alignment it always saw, cuBLAS included, which chooses its
algorithm partly from its operands' alignment (kernels.h).

Memory is read from the driver's account of this process alone
(nvml.own_used_bytes), which no other process moves, starting from
nvml.settled_own_used_bytes, so the CUDA context is never charged to it.
"""

import gc

import numpy as np
import pytest

from microinfer import _microinfer, nvml

MIB = 2**20
ALIGN = _microinfer.weight_alignment


def test_weights_align_as_cudamalloc_aligned_them():
    assert ALIGN == 256


def test_a_tensor_put_into_the_arena_reads_back_what_went_in():
    rng = np.random.default_rng(0)
    arena = _microinfer.DeviceArena(4 * MIB)
    a = rng.standard_normal(1000).astype(np.float32)
    b = rng.standard_normal(333).astype(np.float32)
    ta = arena.put(0, a)
    tb = arena.put(ALIGN * 8, b)
    np.testing.assert_array_equal(ta.to_numpy(), a.astype(np.float16).astype(np.float32))
    np.testing.assert_array_equal(tb.to_numpy(), b.astype(np.float16).astype(np.float32))
    assert ta.numel == 1000 and tb.nbytes == 333 * 2
    assert tb.address - ta.address == ALIGN * 8
    assert arena.address == ta.address


def test_a_tensor_must_start_aligned_and_end_inside():
    arena = _microinfer.DeviceArena(ALIGN * 4)
    with pytest.raises(ValueError, match="multiple of 256"):
        arena.put(2, np.zeros(4, np.float32))
    with pytest.raises(ValueError, match="past the arena"):
        arena.put(ALIGN * 3, np.zeros(ALIGN, np.float32))
    arena.put(ALIGN * 3, np.zeros(ALIGN // 2, np.float32))  # exactly fits


def test_the_arena_is_one_allocation_and_putting_takes_nothing_more():
    """What the driver takes for this process is the arena, rounded to its
    granularity, at creation; putting tensors into it takes nothing more."""
    before = nvml.settled_own_used_bytes()
    arena = _microinfer.DeviceArena(64 * MIB)
    created = nvml.own_used_bytes() - before
    for i in range(64):
        arena.put(i * MIB, np.zeros(MIB // 2, np.float32))
    assert nvml.own_used_bytes() - before == created
    assert 64 * MIB <= created <= 66 * MIB


def test_the_arena_lives_until_its_last_tensor_goes():
    """A tensor keeps its arena alive after the arena's own handle is gone, so
    a weight can never be read from memory already returned; the last one to
    go returns all of it to the driver."""
    before = nvml.settled_own_used_bytes()
    arena = _microinfer.DeviceArena(32 * MIB)
    values = np.arange(10, dtype=np.float32)
    kept = arena.put(ALIGN, values)
    del arena
    gc.collect()
    assert nvml.own_used_bytes() - before >= 32 * MIB
    np.testing.assert_array_equal(kept.to_numpy(), values)
    del kept
    gc.collect()
    assert nvml.own_used_bytes() == before


def test_a_tensor_in_the_arena_is_an_operand_like_any_other():
    """A tensor in the arena goes wherever a weight goes, and is read as one
    that owns its storage is: here as RMSNorm's weight, to the bit."""
    device = _microinfer.device
    arena = _microinfer.DeviceArena(MIB)
    hidden = 64
    weight = arena.put(ALIGN * 2, np.linspace(0.5, 1.5, hidden).astype(np.float32))
    owned = _microinfer.upload_fp16(np.linspace(0.5, 1.5, hidden).astype(np.float32))
    x = _microinfer.upload_fp16(np.random.default_rng(1).standard_normal(3 * hidden)
                                .astype(np.float32))
    via_arena, via_owned = device.empty(3 * hidden), device.empty(3 * hidden)
    device.rmsnorm(x, weight, via_arena, 3, hidden, 1e-6)
    device.rmsnorm(x, owned, via_owned, 3, hidden, 1e-6)
    np.testing.assert_array_equal(via_arena.to_numpy(), via_owned.to_numpy())
