"""NVML's own reading of device memory, through ctypes.

`nvmlDeviceGetMemoryInfo` is the measurement RQ2 stands on and the one
ADR-0007's allocator is judged by, so the allocator tests read NVML itself
rather than `cudaMemGetInfo` and trust the two to agree. The library ships with
the NVIDIA driver; ctypes reaches it without adding a Python dependency, which
matters for a project whose environment is pinned deliberately.
"""

import ctypes
import gc
from functools import cache


class _Memory(ctypes.Structure):
    _fields_ = [("total", ctypes.c_ulonglong), ("free", ctypes.c_ulonglong),
                ("used", ctypes.c_ulonglong)]


@cache
def _device():
    lib = ctypes.CDLL("libnvidia-ml.so.1")
    _check(lib.nvmlInit_v2(), "nvmlInit_v2")
    handle = ctypes.c_void_p()
    _check(lib.nvmlDeviceGetHandleByIndex_v2(0, ctypes.byref(handle)),
           "nvmlDeviceGetHandleByIndex_v2")
    return lib, handle


def _check(status: int, what: str) -> None:
    if status != 0:
        raise RuntimeError(f"{what} failed with NVML status {status}")


def memory() -> _Memory:
    lib, handle = _device()
    info = _Memory()
    _check(lib.nvmlDeviceGetMemoryInfo(handle, ctypes.byref(info)), "nvmlDeviceGetMemoryInfo")
    return info


def free_bytes() -> int:
    """NVML-reported free memory, with this process's own pending frees
    settled first (ADR-0007, note from #6)."""
    gc.collect()
    return memory().free
