"""The build path itself, proven rather than assumed.

These tests exist because ADR-0007 requires the CUDA *driver* API, not only the
runtime API, and a missing `libcuda` link surfaces at link time rather than at
compile time. Discovering that when the allocator is written would be late.
"""

import subprocess

import microinfer
from microinfer import _microinfer


def test_extension_imports():
    assert _microinfer is not None


def test_driver_api_call_succeeded_at_import():
    """`microinfer` calls cuDriverGetVersion while importing; this is its result."""
    assert microinfer.CUDA_DRIVER_VERSION > 0


def test_extension_is_linked_against_libcuda():
    """Assert the linkage directly, not just that a call happened to work."""
    out = subprocess.run(
        ["ldd", _microinfer.__file__], capture_output=True, text=True, check=True
    ).stdout
    assert "libcuda.so" in out, out


def test_device_is_visible():
    assert _microinfer.device_name()


def test_pytorch_is_not_in_the_engine_process():
    """ADR-0002: PyTorch's caching allocator would corrupt the NVML reading the
    whole project depends on. It is never imported in a process running the engine."""
    import sys

    assert "torch" not in sys.modules
