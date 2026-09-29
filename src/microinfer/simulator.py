"""The contention simulator: device memory held on a schedule (#58).

RQ2 and RQ3 need contention on demand, not only when a browser happens to
spike. The simulator is that contention, made the way #45 decided:

- **A process of its own, with its own CUDA context,** so that what it holds
  is another process's to NVML and to the pressure monitor, as real
  contention is. tools/simulate_contention.py runs it.
- **Memory only.** It holds and releases bytes and launches nothing, which
  isolates the variable RQ2 and RQ3 study.
- **The engine's VMM allocator,** in one-granule pages (ADR-0007): a release
  reaches the driver at once, and what it holds is exact to one granule.
- **A schedule of (time, bytes held).** #59 makes schedules from synthetic
  patterns, and #60 from a recorded trace; this executes either.

It allocates and frees at the tail of its range only, so a release never
moves a page, and it can hold at most what the device has: an allocation the
driver refuses (OutOfMemory) leaves it holding what it could, and says so.
"""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path

from . import _microinfer, nvml

GRANULE = _microinfer.granule_bytes()
_TIER = _microinfer.Tier.FP16
#: How long a change waits for NVML to show it before giving up on seeing it.
NVML_WAIT_S = 1.0


class Schedule:
    """Bytes held over time: from each point's time until the next point's,
    the point's bytes. Times are seconds from the start, from 0, increasing."""

    def __init__(self, points):
        self.points = [(float(t), int(b)) for t, b in points]
        if not self.points or self.points[0][0] != 0:
            raise ValueError("a schedule starts at time 0")
        times = [t for t, _ in self.points]
        if any(b <= a for a, b in zip(times, times[1:])):
            raise ValueError("a schedule's times increase")
        if any(b < 0 for _, b in self.points):
            raise ValueError("a schedule holds no negative bytes")

    def __eq__(self, other):
        return isinstance(other, Schedule) and self.points == other.points

    def __len__(self):
        return len(self.points)

    def at(self, t: float) -> int:
        """The bytes held at `t` seconds."""
        held = self.points[0][1]
        for time_s, bytes_ in self.points:
            if time_s > t:
                break
            held = bytes_
        return held

    @property
    def duration_s(self) -> float:
        return self.points[-1][0]

    @property
    def peak_bytes(self) -> int:
        return max(b for _, b in self.points)

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps({"points": self.points}) + "\n")

    @classmethod
    def load(cls, path: str | Path) -> Schedule:
        return cls(json.loads(Path(path).read_text())["points"])


class Holder:
    """Device memory in whole granules, taken and given back at the tail of
    one address range. `capacity_bytes` reserves address space only; the
    device's total by default."""

    def __init__(self, capacity_bytes: int | None = None):
        capacity = capacity_bytes or _microinfer.device_memory_info()["total"]
        pages = -(-capacity // GRANULE)
        self._cache = _microinfer.PagedKVCache([GRANULE] * 4, [pages, 0, 0, 0])
        self._pages = 0

    @property
    def held_bytes(self) -> int:
        return self._pages * GRANULE

    def hold(self, target_bytes: int) -> int:
        """Hold the whole number of granules nearest `target_bytes`, allocating
        or freeing at the tail. Returns what is held: less than asked if the
        device ran out."""
        want = round(target_bytes / GRANULE)
        while self._pages > want:
            self._pages -= 1
            self._cache.free(0, self._pages)
        while self._pages < want:
            try:
                self._cache.allocate(0, self._pages, _TIER)
            except _microinfer.OutOfMemory:
                break  # contention of its own: hold what the device gave
            self._pages += 1
        return self.held_bytes

    def release(self) -> None:
        """Give everything back to the driver."""
        self.hold(0)


def run(schedule: Schedule, holder: Holder, stop: threading.Event,
        on_change=None) -> list[dict]:
    """Apply each point of `schedule` at its time from the start, until the
    schedule ends or `stop` is set. The times are absolute: a slow change
    delays only itself. Each change is reported, to `on_change` as it happens
    and in the returned list: when it was due and applied, the bytes asked
    and held, and how long it took, until the allocator's calls returned
    (api_ms) and until NVML showed this process holding it (nvml_ms), both
    from when it was due."""
    base = nvml.own_used_bytes() - holder.held_bytes  # the context and the rest
    start = time.monotonic_ns()
    events = []
    for offset_s, target in schedule.points:
        due = start + int(offset_s * 1e9)
        if stop.wait(max(due - time.monotonic_ns(), 0) / 1e9):
            break
        began = time.monotonic_ns()
        held = holder.hold(target)
        applied = time.monotonic_ns()
        seen = applied
        nvml_bytes = nvml.own_used_bytes()
        while abs(nvml_bytes - base - held) > GRANULE and seen - applied < NVML_WAIT_S * 1e9:
            nvml_bytes = nvml.own_used_bytes()
            seen = time.monotonic_ns()
        event = {"scheduled_ns": due, "applied_ns": applied, "t_wall": time.time(),
                 "late_s": (began - due) / 1e9, "target_bytes": target, "held_bytes": held,
                 "nvml_bytes": nvml_bytes, "api_ms": (applied - due) / 1e6,
                 "nvml_ms": (max(seen, applied) - due) / 1e6,
                 "seen": abs(nvml_bytes - base - held) <= GRANULE}
        events.append(event)
        if on_change is not None:
            on_change(event)
    return events


#: The steps a calibration holds, each then released: one granule to 1 GiB.
CALIBRATION_BYTES = (GRANULE, 64 * 2**20, 256 * 2**20, 1024 * 2**20)


def calibrate(repeats: int, sizes=CALIBRATION_BYTES, step_s: float = 0.5,
              stop: threading.Event | None = None) -> dict:
    """How long holding and releasing take on this machine: each size held
    for step_s and released for step_s, `repeats` times. Per direction, the
    changes made, the latency until the call returned (api_ms) and until
    NVML showed it (nvml_ms), median, P90 and max, and how many NVML never
    showed within NVML_WAIT_S."""
    points, t = [(0.0, 0)], step_s
    for _ in range(repeats):
        for size in sizes:
            points += [(t, size), (t + step_s, 0)]
            t += 2 * step_s
    holder = Holder()
    try:
        events = run(Schedule(points), holder, stop or threading.Event())
    finally:
        holder.release()

    def spread(values):
        ordered = sorted(values)
        return {"median": ordered[len(ordered) // 2], "p90": ordered[int(0.9 * (len(ordered) - 1))],
                "max": ordered[-1]}

    results = {}
    for direction, changes in (("hold", [e for e in events if e["target_bytes"]]),
                               ("release", [e for e in events[1:] if not e["target_bytes"]])):
        results[direction] = {"changes": len(changes),
                              "api_ms": spread([e["api_ms"] for e in changes]),
                              "nvml_ms": spread([e["nvml_ms"] for e in changes]),
                              "unseen": sum(not e["seen"] for e in changes)}
    return results
