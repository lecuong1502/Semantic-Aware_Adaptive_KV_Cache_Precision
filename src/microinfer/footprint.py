"""What the engine is using, and what the driver thinks is left.

Story 25 of the Milestone 0 spec asks for this specifically, and the reason is
narrow: on a 6 GiB card an out-of-memory failure is ambiguous. It might be the
research — contention, a cache that grew — or it might be an inefficiency in
code that has not been optimised yet. A breakdown settles that in seconds
instead of an afternoon.
"""

from __future__ import annotations

from dataclasses import dataclass

MIB = 1024 * 1024


@dataclass(frozen=True)
class Footprint:
    """Bytes, split by what is holding them."""

    weights: int
    kv_cache: int
    workspace: int
    device_free: int
    device_total: int

    @property
    def engine_total(self) -> int:
        return self.weights + self.kv_cache + self.workspace

    @property
    def unaccounted(self) -> int:
        """Device memory in use that the engine did not allocate.

        The CUDA context, the driver's own structures, the desktop compositor,
        and anything else sharing the card. On this project that last term is
        not noise — it is the phenomenon under study.
        """
        return (self.device_total - self.device_free) - self.engine_total

    def render(self) -> str:
        rows = [
            ("weights", self.weights),
            ("KV cache", self.kv_cache),
            ("workspace", self.workspace),
            ("engine total", self.engine_total),
            ("unaccounted (context, desktop, other processes)", self.unaccounted),
            ("device free", self.device_free),
            ("device total (as CUDA sees it)", self.device_total),
        ]
        width = max(len(label) for label, _ in rows)
        return "\n".join(
            f"  {label:<{width}}  {value / MIB:9.1f} MiB" for label, value in rows
        )
