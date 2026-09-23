"""NVML's own view of the GPU, through ctypes.

`nvmlDeviceGetMemoryInfo` is the reading RQ2 stands on and the one ADR-0007's
allocator is judged by. So what the project reports about the device comes from
NVML itself, not from `cudaMemGetInfo` trusted to agree with it: on this machine
the two differ by about 90 MiB (ADR-0007, note from #10).

The library ships with the NVIDIA driver. ctypes reaches it without a Python
dependency, which matters in an environment pinned deliberately. It needs no
CUDA context and takes no device memory.
"""

from __future__ import annotations

import ctypes
import os
from dataclasses import dataclass, replace
from functools import cache

_SUCCESS = 0
_INSUFFICIENT_SIZE = 7

# nvml.h's buffer sizes for the strings read here.
_DEVICE_NAME_BUFFER_SIZE = 96
_SYSTEM_DRIVER_VERSION_BUFFER_SIZE = 80
_PROCESS_NAME_BUFFER_SIZE = 256


class _Memory(ctypes.Structure):
    _fields_ = [("total", ctypes.c_ulonglong), ("free", ctypes.c_ulonglong),
                ("used", ctypes.c_ulonglong)]


class _ProcessInfo(ctypes.Structure):
    """nvmlProcessInfo_t, as the _v3 process queries fill it."""

    _fields_ = [("pid", ctypes.c_uint), ("usedGpuMemory", ctypes.c_ulonglong),
                ("gpuInstanceId", ctypes.c_uint), ("computeInstanceId", ctypes.c_uint)]


@dataclass(frozen=True)
class GpuProcess:
    pid: int
    name: str | None  # None where NVML may not name another user's process
    kind: str  # "compute", "graphics", or "compute+graphics" for a process holding both
    used_bytes: int | None  # None where the driver does not report it


def _check(status: int, what: str) -> None:
    if status != _SUCCESS:
        raise RuntimeError(f"{what} failed with NVML status {status}")


class NvmlUnavailable(RuntimeError):
    """No NVIDIA driver library to read from."""


def _declare(lib) -> None:
    """Argument types for every call made, so ctypes cannot guess them wrong."""
    handle, uint_p = ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint)
    lib.nvmlInit_v2.argtypes = []
    lib.nvmlDeviceGetHandleByIndex_v2.argtypes = [ctypes.c_uint, ctypes.POINTER(handle)]
    lib.nvmlDeviceGetMemoryInfo.argtypes = [handle, ctypes.POINTER(_Memory)]
    lib.nvmlDeviceGetName.argtypes = [handle, ctypes.c_char_p, ctypes.c_uint]
    lib.nvmlSystemGetDriverVersion.argtypes = [ctypes.c_char_p, ctypes.c_uint]
    lib.nvmlSystemGetProcessName.argtypes = [ctypes.c_uint, ctypes.c_char_p, ctypes.c_uint]
    for query in (lib.nvmlDeviceGetComputeRunningProcesses_v3,
                  lib.nvmlDeviceGetGraphicsRunningProcesses_v3):
        query.argtypes = [handle, uint_p, ctypes.POINTER(_ProcessInfo)]


@cache
def _device(index: int = 0):
    """NVML's device `index`. Index 0 is the only GPU on this project's
    machines; on a multi-GPU host NVML's order need not match CUDA's, and this
    would have to be chosen by PCI bus id instead."""
    try:
        lib = ctypes.CDLL("libnvidia-ml.so.1")
    except OSError as exc:
        raise NvmlUnavailable("libnvidia-ml.so.1 was not found: NVML ships with the NVIDIA "
                              "driver, and nothing here can be measured without it") from exc
    _declare(lib)
    _check(lib.nvmlInit_v2(), "nvmlInit_v2")
    handle = ctypes.c_void_p()
    _check(lib.nvmlDeviceGetHandleByIndex_v2(index, ctypes.byref(handle)),
           "nvmlDeviceGetHandleByIndex_v2")
    return lib, handle


def memory() -> _Memory:
    lib, handle = _device()
    info = _Memory()
    _check(lib.nvmlDeviceGetMemoryInfo(handle, ctypes.byref(info)), "nvmlDeviceGetMemoryInfo")
    return info


def device_name() -> str:
    lib, handle = _device()
    buf = ctypes.create_string_buffer(_DEVICE_NAME_BUFFER_SIZE)
    _check(lib.nvmlDeviceGetName(handle, buf, len(buf)), "nvmlDeviceGetName")
    return buf.value.decode()


def driver_version() -> str:
    lib, _ = _device()
    buf = ctypes.create_string_buffer(_SYSTEM_DRIVER_VERSION_BUFFER_SIZE)
    _check(lib.nvmlSystemGetDriverVersion(buf, len(buf)), "nvmlSystemGetDriverVersion")
    return buf.value.decode()


def _process_name(lib, pid: int) -> str | None:
    buf = ctypes.create_string_buffer(_PROCESS_NAME_BUFFER_SIZE)
    if lib.nvmlSystemGetProcessName(pid, buf, len(buf)) != _SUCCESS:
        return None
    return buf.value.decode(errors="replace") or None


def processes() -> list[GpuProcess]:
    """Every process holding the GPU, compute and graphics, this one included.

    On a desktop the display server, the compositor, a browser and an editor
    are always here. That is the contention the project studies, which is why
    a measurement records them rather than assuming them away."""
    lib, handle = _device()
    found: dict[int, GpuProcess] = {}
    for kind, query in (("compute", lib.nvmlDeviceGetComputeRunningProcesses_v3),
                        ("graphics", lib.nvmlDeviceGetGraphicsRunningProcesses_v3)):
        count = ctypes.c_uint(0)
        status = query(handle, ctypes.byref(count), None)
        if status not in (_SUCCESS, _INSUFFICIENT_SIZE):
            _check(status, query.__name__)
        # A process can start between the two calls; room for a few more.
        infos = (_ProcessInfo * (count.value + 8))()
        count = ctypes.c_uint(len(infos))
        _check(query(handle, ctypes.byref(count), infos), query.__name__)
        for info in infos[: count.value]:
            pid, used = int(info.pid), info.usedGpuMemory
            if pid in found:
                # In both lists: one process, with both kinds of context. Its
                # memory is one figure, reported by each query.
                found[pid] = replace(found[pid], kind=f"{found[pid].kind}+{kind}")
                continue
            # NVML_VALUE_NOT_AVAILABLE is the all-ones pattern.
            found[pid] = GpuProcess(pid, _process_name(lib, pid), kind,
                                    None if used == ctypes.c_ulonglong(-1).value else int(used))
    return list(found.values())


def other_processes() -> list[GpuProcess]:
    """processes(), without this one."""
    own = os.getpid()
    return [p for p in processes() if p.pid != own]
