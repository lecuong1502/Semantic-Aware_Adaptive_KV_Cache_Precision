"""NVML readings for the tests: the engine's own reader (microinfer.nvml),
plus the one thing only a test needs.

Kept as a module of its own so the allocator tests read `nvml.free_bytes()`
as they always have.
"""

import gc

from microinfer.nvml import memory  # noqa: F401 - re-exported for the tests


def free_bytes() -> int:
    """NVML-reported free memory, with this process's own pending frees
    settled first (ADR-0007, note from #6)."""
    gc.collect()
    return memory().free
