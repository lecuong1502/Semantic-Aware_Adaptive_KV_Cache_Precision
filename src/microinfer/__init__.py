"""MicroInfer: an inference engine that reallocates KV cache precision at runtime.

PyTorch must never be imported in a process that runs this package. Its caching
allocator retains freed device memory, so NVML would report that reservation as
used memory, indistinguishable from the external contention this project exists
to detect (ADR-0002).
"""

from . import _microinfer

try:
    #: Driver version reported by cuDriverGetVersion at import.
    #:
    #: Calling into the driver API here is deliberate. ADR-0007's allocator needs
    #: libcuda, and a missing link surfaces at link time rather than compile
    #: time. Failing at import is far cheaper than failing when the allocator is
    #: first exercised.
    CUDA_DRIVER_VERSION: int = _microinfer.cuda_driver_version()
except Exception as exc:  # pragma: no cover - environment failure, not logic
    raise RuntimeError(
        "MicroInfer could not reach the CUDA driver. The extension links "
        "libcuda (the driver API), which needs an NVIDIA driver and a visible "
        "GPU. Check `nvidia-smi`."
    ) from exc

from .config import ConfigMismatch, ModelConfig  # noqa: E402
from .engine import (Engine, expected_weight_bytes, expected_weight_shapes,  # noqa: E402
                     kv_cache_bytes, paged_cache_bytes, paged_cache_ranges, weight_layout)
from .footprint import Footprint  # noqa: E402

__all__ = [
    "CUDA_DRIVER_VERSION",
    "ConfigMismatch",
    "Engine",
    "Footprint",
    "ModelConfig",
    "_microinfer",
    "expected_weight_bytes",
    "expected_weight_shapes",
    "kv_cache_bytes",
    "paged_cache_bytes",
    "paged_cache_ranges",
    "weight_layout",
]
