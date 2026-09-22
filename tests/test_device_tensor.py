"""Seam B: the extension boundary for device storage.

CONTRIBUTING makes `_microinfer` a first-class seam. `DeviceTensor`,
`upload_fp16` and `device_memory_info` are what the footprint in #5 is built
from, so if `nbytes` is wrong the whole report is wrong quietly.
"""

import numpy as np
import pytest
from conftest import stable_free_bytes

from microinfer import _microinfer

MIB = 1024 * 1024


def test_upload_round_trips_values_that_fp16_holds_exactly():
    values = np.array([0.0, 1.0, -2.5, 0.5, 1024.0, -65504.0], dtype=np.float32)
    tensor = _microinfer.upload_fp16(values)
    np.testing.assert_array_equal(tensor.to_numpy(), values)


def test_upload_rounds_to_fp16_and_says_so_in_its_size():
    """The conversion is lossy and the size reflects fp16, not the fp32 input."""
    values = np.array([1.0 + 2**-20], dtype=np.float32)  # unrepresentable in fp16
    tensor = _microinfer.upload_fp16(values)
    assert tensor.to_numpy()[0] == pytest.approx(1.0)
    assert tensor.nbytes == 2


@pytest.mark.parametrize("count", [0, 1, 7, 1024, 100_000])
def test_nbytes_is_two_per_element(count):
    """`footprint.weights` is a sum of these. A wrong factor here misreports
    every figure downstream without failing anything."""
    tensor = _microinfer.upload_fp16(np.zeros(count, dtype=np.float32))
    assert tensor.numel == count
    assert tensor.nbytes == 2 * count


def test_upload_actually_takes_device_memory():
    """Not merely bookkeeping — the driver's free memory moves.

    Allocates 256 MiB and allows a 5% shortfall rather than demanding the exact
    figure, and the reason is worth knowing before writing another test like
    this one.

    In isolation the delta is exact: -32.00 MiB for a 32 MiB tensor, every run.
    Inside the full suite the same assertion failed intermittently, short by
    about 5 MiB — a deferred free landing between the two readings in a
    long-lived process. It is not contention from other processes: sampling
    free memory sixty times over three idle seconds moves it by 0.19 MiB total.

    So the noise is bounded and absolute, which means a larger allocation
    drowns it. 5% of 256 MiB is 12.8 MiB, comfortably above what was observed,
    while still failing outright if an upload stopped taking memory at all.
    """
    before = stable_free_bytes()
    tensor = _microinfer.upload_fp16(np.zeros(256 * MIB // 2, dtype=np.float32))
    after = _microinfer.device_memory_info()["free"]
    taken = before - after
    assert taken >= tensor.nbytes * 0.95, (
        f"driver free memory moved {taken / MIB:.1f} MiB for a "
        f"{tensor.nbytes / MIB:.1f} MiB upload"
    )
    del tensor


def test_device_memory_info_is_plausible():
    info = _microinfer.device_memory_info()
    assert 0 < info["free"] <= info["total"]
    assert info["total"] > 1024 * MIB  # any GPU this project targets


def test_freeing_a_tensor_returns_memory_to_the_driver():
    """A weaker cousin of the contract ADR-0007 will have to prove for pages.
    cudaMalloc returns to the driver on free; the VMM allocator will have to
    show the same for granules."""
    baseline = stable_free_bytes()
    tensor = _microinfer.upload_fp16(np.zeros(64 * MIB // 2, dtype=np.float32))
    assert _microinfer.device_memory_info()["free"] < baseline
    del tensor
    recovered = stable_free_bytes()
    assert baseline - recovered < 8 * MIB, "memory did not come back"


# The margin above is measured, not guessed. If a later change makes these
# assertions flaky again, measure the shortfall before widening the tolerance —
# on this project a loose memory assertion can hide the leak ADR-0007 exists to
# prevent.
