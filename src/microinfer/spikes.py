"""Spikes in a contention trace, and what each one measures (#49).

The analysis RQ1's statistics rest on, as the Milestone 1 spec (#45) defines
it. The **baseline** is the rolling median of free memory over the preceding
5 s. A **spike** is free memory at least 64 MiB below the baseline for at
least 100 ms. 64 MiB is well above the noise measured on this machine, a few
2 MiB granules (ADR-0007, note from #10). All three are parameters, so that
the raw traces can be read again at any threshold.

**The baseline holds while a spike lasts, for one window at most.** Left to
roll, the median would sink into a spike longer than half the window and end
it there, however long the memory stayed taken. So from a spike's first
sample the baseline is held at its value then. A spike still below it once a
whole window has passed is a lasting step: it ends there, as not recovered,
and the rolling median, by then made of the new level, takes over as the
baseline again.

For each spike, with `deficit` the held baseline minus free memory:

- **amplitude**, the largest deficit;
- **rise time**, from the deficit's last crossing of 10% of the amplitude
  before it first reaches 90%, to that crossing of 90%;
- **duration**, from the deficit's crossing of the threshold on the way down
  to its crossing back;
- **recovery**, from the deficit's last crossing of 90% of the amplitude, the
  end of the peak, to its crossing of 10% on the way back: the rise time's
  mirror. None for a spike that did not recover.

Crossing times are interpolated linearly between samples, so a measure is
not rounded to the 20 ms period; a ramp shorter than a period is still read
as a fraction of one.

This is measurement, not inference, so it computes in NumPy on the host
(ADR-0002, amendment on its scope).
"""

from __future__ import annotations

import bisect
from collections import deque
from dataclasses import dataclass

import numpy as np

DEFAULT_WINDOW_S = 5.0
DEFAULT_THRESHOLD_BYTES = 64 * 2**20
DEFAULT_MIN_DURATION_S = 0.1

#: The levels, as shares of the amplitude, that rise time and recovery run between.
_LOW, _HIGH = 0.1, 0.9


@dataclass(frozen=True)
class Spike:
    """One spike. Times are on the trace's monotonic clock, in ns."""

    start_ns: int  # the deficit crosses the threshold
    peak_ns: int  # the largest deficit, first reached
    end_ns: int | None  # it crosses back; None if it did not recover
    baseline_bytes: int  # the baseline, held from the start
    amplitude_bytes: int
    rise_s: float | None  # None if the trace does not show the rise begin
    duration_s: float  # to the end of the window for one that did not recover
    recovery_s: float | None

    @property
    def recovered(self) -> bool:
        return self.end_ns is not None


def find_spikes(t_mono_ns: np.ndarray, free_bytes: np.ndarray, *,
                window_s: float = DEFAULT_WINDOW_S,
                threshold_bytes: int = DEFAULT_THRESHOLD_BYTES,
                min_duration_s: float = DEFAULT_MIN_DURATION_S) -> list[Spike]:
    """The spikes in a trace of free memory, in order. No spike is sought
    before a whole window of the trace has passed to give a baseline."""
    t = np.asarray(t_mono_ns, dtype=np.int64)
    free = np.asarray(free_bytes, dtype=np.float64)
    if len(t) != len(free):
        raise ValueError("t_mono_ns and free_bytes differ in length")
    if np.any(np.diff(t) <= 0):
        raise ValueError("the trace's times must be strictly increasing")
    window_ns = int(window_s * 1e9)

    found: list[Spike] = []
    history: deque[tuple[int, float]] = deque()  # the preceding window, in time order
    ordered: list[float] = []  # the same values, sorted
    onset = None  # the first sample of the spike in progress
    held = 0.0  # its baseline
    for i in range(len(t)):
        while history and history[0][0] < t[i] - window_ns:
            del ordered[bisect.bisect_left(ordered, history.popleft()[1])]
        if onset is None:
            if t[i] - t[0] >= window_ns:
                baseline = _median(ordered)
                if baseline - free[i] >= threshold_bytes:
                    onset, held = i, baseline
        elif held - free[i] < threshold_bytes:
            found.append(_measure(t, free, onset, i, held, True, threshold_bytes, window_ns))
            onset = None
        elif t[i] - t[onset] >= window_ns:
            found.append(_measure(t, free, onset, i, held, False, threshold_bytes, window_ns))
            onset = None
        history.append((int(t[i]), float(free[i])))
        bisect.insort(ordered, float(free[i]))
    if onset is not None:
        found.append(_measure(t, free, onset, len(t) - 1, held, False, threshold_bytes,
                              window_ns))
    return [s for s in found if s is not None and s.duration_s >= min_duration_s]


def _median(ordered: list[float]) -> float:
    n = len(ordered)
    mid = n // 2
    return ordered[mid] if n % 2 else (ordered[mid - 1] + ordered[mid]) / 2


def _crossing(t: np.ndarray, deficit: np.ndarray, a: int, b: int, level: float) -> float:
    """When the deficit passed `level` between samples a and b, in ns,
    interpolated linearly."""
    da, db = deficit[a], deficit[b]
    if db == da:
        return float(t[b])
    return float(t[a] + (level - da) / (db - da) * (t[b] - t[a]))


def _measure(t: np.ndarray, free: np.ndarray, onset: int, stop: int, held: float,
             recovered: bool, threshold: float, window_ns: int) -> Spike | None:
    """The measures of the spike from sample `onset`. `stop` is the first
    sample back above the threshold if it recovered, else the sample it was
    cut at."""
    deficit = held - free
    start = _crossing(t, deficit, onset - 1, onset, threshold)
    inside = slice(onset, stop if recovered else stop + 1)
    peak = onset + int(np.argmax(deficit[inside]))
    amplitude = float(deficit[peak])
    low, high = _LOW * amplitude, _HIGH * amplitude

    # The rise: back from the first sample at 90% to the last below 10%,
    # no further back than a window before the spike.
    top = onset + int(np.argmax(deficit[inside] >= high))
    earliest = int(np.searchsorted(t, t[onset] - window_ns))
    rise = None
    for q in range(top - 1, earliest - 1, -1):
        if deficit[q] < low:
            rise = (_crossing(t, deficit, top - 1, top, high)
                    - _crossing(t, deficit, q, q + 1, low)) / 1e9
            break

    end = recovery = None
    if recovered:
        end = _crossing(t, deficit, stop - 1, stop, threshold)
        # The recovery: from the last sample at 90% to the first after it at
        # 10% or less, no further than a window after the spike ends.
        last_top = onset + int(np.flatnonzero(deficit[inside] >= high)[-1])
        latest = int(np.searchsorted(t, t[stop] + window_ns, side="right"))
        for s in range(last_top + 1, min(latest, len(t))):
            if deficit[s] <= low:
                recovery = (_crossing(t, deficit, s - 1, s, low)
                            - _crossing(t, deficit, last_top, last_top + 1, high)) / 1e9
                break
    duration = ((end if end is not None else float(t[stop])) - start) / 1e9
    return Spike(start_ns=int(start), peak_ns=int(t[peak]),
                 end_ns=None if end is None else int(end), baseline_bytes=int(held),
                 amplitude_bytes=int(amplitude), rise_s=rise, duration_s=duration,
                 recovery_s=recovery)
