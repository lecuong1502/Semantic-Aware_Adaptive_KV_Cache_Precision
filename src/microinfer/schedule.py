"""A schedule of contention: bytes taken over time (#58, #59).

What the contention simulator (simulator.py) executes and the pattern
generators (patterns.py) make. Kept apart from both, with no device code,
so that a schedule can be made, read and checked without a GPU.
"""

from __future__ import annotations

import json
from pathlib import Path


class Schedule:
    """Bytes taken over time: from each point's time until the next point's,
    the point's bytes, and the last point's for as long as the simulator
    runs. Times are seconds from the start, from 0, increasing."""

    def __init__(self, points):
        self.points = [(float(t), int(b)) for t, b in points]
        if not self.points or self.points[0][0] != 0:
            raise ValueError("a schedule starts at time 0")
        times = [t for t, _ in self.points]
        if any(b <= a for a, b in zip(times, times[1:])):
            raise ValueError("a schedule's times increase")
        if any(b < 0 for _, b in self.points):
            raise ValueError("a schedule takes no negative bytes")

    def __eq__(self, other):
        return isinstance(other, Schedule) and self.points == other.points

    def at(self, t: float) -> int:
        """The bytes taken at `t` seconds."""
        taken = self.points[0][1]
        for time_s, bytes_ in self.points:
            if time_s > t:
                break
            taken = bytes_
        return taken

    @property
    def duration_s(self) -> float:
        """When the last level begins."""
        return self.points[-1][0]

    @property
    def peak_bytes(self) -> int:
        return max(b for _, b in self.points)

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps({"points": self.points}) + "\n")

    @classmethod
    def load(cls, path: str | Path) -> Schedule:
        return cls(json.loads(Path(path).read_text())["points"])
