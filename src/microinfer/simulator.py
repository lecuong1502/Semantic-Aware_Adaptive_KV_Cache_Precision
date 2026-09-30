"""The contention simulator: device memory taken on a schedule (#58).

RQ2 and RQ3 need contention on demand, not only when a browser happens to
spike. The simulator is that contention, made the way #45 decided:

- **A process of its own, with its own CUDA context,** so that what it takes
  is another process's to NVML and to the pressure monitor, as real
  contention is. tools/simulate_contention.py runs it.
- **Memory only.** It takes and gives back bytes and launches nothing, which
  isolates the variable RQ2 and RQ3 study.
- **The engine's VMM allocator,** a granule at a time (ADR-0007): memory
  given back reaches the driver at once, and what it takes is exact to one
  granule.
- **A schedule of (time, bytes taken).** #59 makes schedules from synthetic
  patterns, and #60 from a recorded trace; this executes either. Each level
  lasts until the next point, and the last until the simulator is stopped: a
  schedule that means to give everything back ends at 0.

It takes and gives back granules at the tail of its range only, so giving
back never moves memory. It can take at most what the device has: an
allocation the driver refuses (OutOfMemory) leaves it with what it could
get, and each change records the shortfall.

"Take" rather than "hold": a hold is the engine's (CONTEXT.md).
"""

from __future__ import annotations

import threading
import time

from . import _microinfer, nvml
from .contention import spread
from .schedule import Schedule  # noqa: F401 - the simulator runs these

GRANULE = _microinfer.granule_bytes()
_TIER = _microinfer.Tier.FP16
#: How long a change waits for NVML to show it, and how often it looks.
NVML_WAIT_S = 1.0
NVML_POLL_S = 0.001


class Reservation:
    """Device memory in whole granules, taken and given back at the tail of
    one address range. `capacity_bytes` reserves address space only, and is
    the most it can take; the device's total by default."""

    def __init__(self, capacity_bytes: int | None = None):
        capacity = (_microinfer.device_memory_info()["total"] if capacity_bytes is None
                    else capacity_bytes)
        if capacity <= 0:
            raise ValueError(f"a reservation needs room for a granule; got {capacity} bytes")
        self._capacity = -(-capacity // GRANULE)
        # The allocator's pages are granules here: one tier, one "layer".
        self._range = _microinfer.PagedKVCache([GRANULE] * 4, [self._capacity, 0, 0, 0])
        self._granules = 0

    @property
    def taken_bytes(self) -> int:
        return self._granules * GRANULE

    def take(self, target_bytes: int) -> int:
        """Take the whole number of granules nearest `target_bytes`, up to
        the capacity, allocating or giving back at the tail. Returns what is
        taken: less than asked if the device ran out."""
        want = min(round(target_bytes / GRANULE), self._capacity)
        while self._granules > want:
            self._granules -= 1
            self._range.free(0, self._granules)
        while self._granules < want:
            try:
                self._range.allocate(0, self._granules, _TIER)
            except _microinfer.OutOfMemory:
                break  # contention of its own: keep what the device gave
            self._granules += 1
        return self.taken_bytes

    def release(self) -> None:
        """Give everything back to the driver."""
        self.take(0)


def _seen_by_nvml(base: int, taken: int, stop: threading.Event) -> tuple[int, int, bool]:
    """Read this process's memory from NVML until it shows `taken` above
    `base`, to within a granule, or NVML_WAIT_S passes, or `stop` is set.
    Returns the last reading, when that reading returned, and whether it
    showed it."""
    deadline = time.monotonic_ns() + NVML_WAIT_S * 1e9
    while True:
        reading = nvml.own_used_bytes()
        read_at = time.monotonic_ns()
        seen = abs(reading - base - taken) <= GRANULE
        if seen or read_at > deadline or stop.is_set():
            return reading, read_at, seen
        time.sleep(NVML_POLL_S)


def run(schedule: Schedule, reservation: Reservation, stop: threading.Event,
        on_change=None) -> list[dict]:
    """Apply each point of `schedule` at its time from the start; after the
    last, keep its level until `stop` is set. The times are absolute: a slow
    change delays only itself. Each change is reported, to `on_change` as it
    happens and in the returned list:

    - when it was due, and how late it began (late_s);
    - the bytes asked, taken, and the shortfall, if the device ran out;
    - how long after it was due the allocator's calls had returned (api_ms),
      and the reading of NVML that showed this process with the memory had
      returned (nvml_ms, from the first reading on), and whether one did."""
    base = nvml.own_used_bytes() - reservation.taken_bytes  # the context and the rest
    start = time.monotonic_ns()
    events = []
    for offset_s, target in schedule.points:
        due = start + int(offset_s * 1e9)
        if stop.wait(max(due - time.monotonic_ns(), 0) / 1e9):
            break
        began = time.monotonic_ns()
        taken = reservation.take(target)
        applied = time.monotonic_ns()
        reading, read_at, seen = _seen_by_nvml(base, taken, stop)
        event = {"scheduled_ns": due, "applied_ns": applied, "t_wall": time.time(),
                 "late_s": (began - due) / 1e9, "target_bytes": target, "taken_bytes": taken,
                 "shortfall_bytes": max(round(target / GRANULE) * GRANULE - taken, 0),
                 "nvml_bytes": reading, "api_ms": (applied - due) / 1e6,
                 "nvml_ms": (read_at - due) / 1e6, "seen": seen}
        events.append(event)
        if on_change is not None:
            on_change(event)
    else:
        stop.wait()
    return events


def summarise(events: list[dict]) -> dict:
    """What a run's changes (run's events) came to: how many, how late
    they began and how long until NVML showed them, as median, P90 and max;
    how many NVML never showed; and the largest shortfall."""
    return {"changes": len(events), "late_s": spread([e["late_s"] for e in events]),
            "nvml_ms": spread([e["nvml_ms"] for e in events]),
            "unseen": sum(not e["seen"] for e in events),
            "shortfall_bytes_max": max((e["shortfall_bytes"] for e in events), default=0)}


#: The steps a calibration takes, each then given back: one granule to 1 GiB.
CALIBRATION_BYTES = (GRANULE, 64 * 2**20, 256 * 2**20, 1024 * 2**20)
CALIBRATION_STEP_S = 0.5


def calibrate(repeats: int, sizes=CALIBRATION_BYTES, step_s: float = CALIBRATION_STEP_S,
              stop: threading.Event | None = None) -> dict:
    """How long taking and giving back take on this machine: each size taken
    for step_s and given back for step_s, `repeats` times. Per size, and per
    direction: the changes made; how late they began; the latency until the
    calls returned (api_ms) and until NVML showed it (nvml_ms), as median,
    P90 and max; and how many NVML never showed within NVML_WAIT_S."""
    points, t = [(0.0, 0)], step_s
    for _ in range(repeats):
        for size in sizes:
            points += [(t, size), (t + step_s, 0)]
            t += 2 * step_s
    reservation = Reservation()
    done = stop or threading.Event()
    # The schedule's last level, 0, is kept until stopped: stop once it is applied.
    last = len(points)
    events = []

    def counted(event):
        events.append(event)
        if len(events) == last:
            done.set()

    try:
        run(Schedule(points), reservation, done, on_change=counted)
    finally:
        reservation.release()

    results = {}
    for size in sizes:
        changes = {"take": [], "give_back": []}
        for before, after in zip(events, events[1:]):
            if after["target_bytes"] == size:
                changes["take"].append(after)
            elif before["target_bytes"] == size and after["target_bytes"] == 0:
                changes["give_back"].append(after)
        results[str(size)] = {
            direction: {"changes": len(made), "late_s": spread([e["late_s"] for e in made]),
                        "api_ms": spread([e["api_ms"] for e in made]),
                        "nvml_ms": spread([e["nvml_ms"] for e in made]),
                        "unseen": sum(not e["seen"] for e in made)}
            for direction, made in changes.items()}
    return results
