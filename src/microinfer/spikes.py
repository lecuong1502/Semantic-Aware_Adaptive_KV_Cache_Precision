"""Spikes in a contention trace, and what each one measures (#49).

The analysis RQ1's statistics rest on, as the Milestone 1 spec (#45) defines
it. The **baseline** is the rolling median of free memory over the preceding
5 s. A **spike** is free memory at least 64 MiB below the baseline for at
least 100 ms. 64 MiB is well above the noise measured on this machine, a few
2 MiB granules (ADR-0007, note from #10). All of these are parameters, so
that the raw traces can be read again at any threshold.

Four rules make the definition hold on a real trace:

- **The baseline holds while a spike lasts, for one window at most.** Left
  to roll, the median would sink into a spike longer than half the window
  and end it there, however long the memory stayed taken. So from a spike's
  first sample the baseline is held at its value then. A spike still below
  it once a whole window has passed is a **lasting drop**: it ends there,
  not recovered, and the rolling median, by then made of the new level,
  takes over as the baseline again.
- **A spike ends only when the deficit falls below half the threshold.**
  Without that hysteresis, noise on a spike whose deficit sits near the
  threshold would break it into several. Its duration is still the time the
  deficit spends at or above the threshold, from its first crossing up to
  its last crossing down.
- **A gap in the trace ends a spike.** The recorder skips a deadline it
  misses rather than making it up (recorder.py), and what free memory did
  in a gap longer than `max_gap_s` is unknown. A spike the gap cuts is
  **censored**, as is one the end of the trace cuts: what is known of its
  duration is a lower bound. Nothing is interpolated across a gap.
- **A measure stays within its own spike.** The searches for the rise and
  the recovery stop at the neighbouring spikes and at gaps, and return None
  rather than run on into another spike.

For each spike, with `deficit` the held baseline minus free memory and the
**peak** the run of samples at or above 90% of the amplitude around its
largest deficit:

- **amplitude**, the largest deficit;
- **rise time**, from the deficit's last crossing of 10% of the amplitude
  before the peak to the peak's start;
- **duration**, as above;
- **recovery**, from the peak's end to the deficit's first crossing of 10%
  after it: the rise time's mirror, 90% back to 10%. None for a spike that
  did not recover, or did not come back to 10% before the next one began.

Crossing times are interpolated linearly between samples, so a measure is
not rounded to the 20 ms period.

This is measurement, not inference, so it computes in NumPy on the host
(ADR-0002, amendment on its scope).
"""

from __future__ import annotations

import bisect
from collections import deque
from dataclasses import dataclass
from typing import Literal

import numpy as np

DEFAULT_WINDOW_S = 5.0
DEFAULT_THRESHOLD_BYTES = 64 * 2**20
DEFAULT_MIN_DURATION_S = 0.1
#: Five periods at 50 Hz: longer than any jitter the recorder showed (#46).
DEFAULT_MAX_GAP_S = 0.1

#: The levels, as shares of the amplitude, that rise time and recovery run between.
_LOW, _HIGH = 0.1, 0.9

Ending = Literal["recovered", "lasting", "censored"]


@dataclass(frozen=True)
class Spike:
    """One spike. Times are on the trace's monotonic clock, in ns."""

    start_ns: int  # the deficit crosses the threshold
    end_ns: int | None  # its last crossing back; None unless it recovered
    ending: Ending  # recovered, a lasting drop, or cut by a gap or the trace's end
    baseline_bytes: int  # the baseline, held from the start
    amplitude_bytes: int
    peak_start_ns: int  # the deficit reaches 90% of the amplitude: the rise ends
    peak_end_ns: int  # it leaves 90% for the last time: the recovery begins
    rise_s: float | None  # None if the trace does not show the rise begin
    duration_s: float  # a lower bound unless it recovered
    recovery_s: float | None

    @property
    def recovered(self) -> bool:
        return self.ending == "recovered"


@dataclass(frozen=True)
class _Found:
    """A spike as detection leaves it, before it is measured: its first
    sample, the sample after its last, its held baseline and how it ended."""

    onset: int
    stop: int
    baseline: float
    ending: Ending


def find_spikes(t_mono_ns: np.ndarray, free_bytes: np.ndarray, *,
                window_s: float = DEFAULT_WINDOW_S,
                threshold_bytes: int = DEFAULT_THRESHOLD_BYTES,
                min_duration_s: float = DEFAULT_MIN_DURATION_S,
                release_bytes: int | None = None,
                max_gap_s: float = DEFAULT_MAX_GAP_S) -> list[Spike]:
    """The spikes in a trace of free memory, in order: the t_mono_ns and
    free_bytes columns of a recording (recorder.read). A spike ends when the
    deficit falls below `release_bytes`, half the threshold by default. No
    spike is sought before a whole window of the trace has passed to give a
    baseline."""
    t = np.asarray(t_mono_ns, dtype=np.int64)
    free = np.asarray(free_bytes, dtype=np.float64)
    release = threshold_bytes / 2 if release_bytes is None else release_bytes
    if len(t) != len(free):
        raise ValueError("t_mono_ns and free_bytes differ in length")
    if np.any(np.diff(t) <= 0):
        raise ValueError("the trace's times must be strictly increasing")
    if not (window_s > 0 and threshold_bytes > 0 and 0 < release <= threshold_bytes
            and min_duration_s >= 0 and max_gap_s > 0):
        raise ValueError("the window, threshold and largest gap must be positive, the "
                         "release within the threshold, and the minimum duration not "
                         "negative")
    window_ns, max_gap_ns = int(window_s * 1e9), int(max_gap_s * 1e9)

    found = _detect(t, free, window_ns, threshold_bytes, release, max_gap_ns)
    spikes = []
    for k, f in enumerate(found):
        lo = found[k - 1].stop if k else 0
        hi = found[k + 1].onset if k + 1 < len(found) else len(t)
        spike = _Measure(t, f.baseline - free, threshold_bytes, window_ns, max_gap_ns,
                         lo, hi).spike(f)
        if spike.duration_s >= min_duration_s:
            spikes.append(spike)
    return spikes


def _detect(t: np.ndarray, free: np.ndarray, window_ns: int, threshold: float,
            release: float, max_gap_ns: int) -> list[_Found]:
    found: list[_Found] = []
    history: deque[tuple[int, float]] = deque()  # the preceding window, in time order
    ordered: list[float] = []  # the same values, sorted
    onset = None  # the first sample of the spike in progress
    held = 0.0  # its baseline
    for i in range(len(t)):
        while history and history[0][0] < t[i] - window_ns:
            del ordered[bisect.bisect_left(ordered, history.popleft()[1])]
        if onset is not None:
            ending: Ending | None = None
            if t[i] - t[i - 1] > max_gap_ns:
                ending = "censored"
            elif held - free[i] < release:
                ending = "recovered"
            elif t[i] - t[onset] >= window_ns:
                ending = "lasting"
            if ending is not None:
                found.append(_Found(onset, i, held, ending))
                onset = None
                if ending == "lasting":
                    # The drop is the baseline now; the next spike is sought
                    # from the next sample, against it.
                    _remember(history, ordered, t[i], free[i])
                    continue
        if onset is None and t[i] - t[0] >= window_ns and ordered:
            baseline = _median(ordered)
            if baseline - free[i] >= threshold:
                onset, held = i, baseline
        _remember(history, ordered, t[i], free[i])
    if onset is not None:
        found.append(_Found(onset, len(t), held, "censored"))
    return found


def _remember(history: deque, ordered: list[float], t: int, value: float) -> None:
    history.append((int(t), float(value)))
    bisect.insort(ordered, float(value))


def _median(ordered: list[float]) -> float:
    n = len(ordered)
    mid = n // 2
    return ordered[mid] if n % 2 else (ordered[mid - 1] + ordered[mid]) / 2


class _Measure:
    """The measures of one spike, from its deficit, searching no further
    back than sample `lo` or forward than sample `hi` (the neighbouring
    spikes), and never across a gap."""

    def __init__(self, t: np.ndarray, deficit: np.ndarray, threshold: float,
                 window_ns: int, max_gap_ns: int, lo: int, hi: int):
        self.t, self.deficit = t, deficit
        self.threshold, self.window_ns, self.max_gap_ns = threshold, window_ns, max_gap_ns
        self.lo, self.hi = lo, hi

    def _joined(self, a: int) -> bool:
        """Whether samples a and a + 1 are one period apart, not a gap."""
        return self.t[a + 1] - self.t[a] <= self.max_gap_ns

    def _crossing(self, a: int, level: float) -> float:
        """When the deficit passed `level` between samples a and a + 1, in ns."""
        da, db = self.deficit[a], self.deficit[a + 1]
        ta, tb = float(self.t[a]), float(self.t[a + 1])
        return tb if db == da else ta + (level - da) / (db - da) * (tb - ta)

    def spike(self, f: _Found) -> Spike:
        t, deficit = self.t, self.deficit
        body = range(f.onset, f.stop)
        peak = f.onset + int(np.argmax(deficit[f.onset:f.stop]))
        amplitude = float(deficit[peak])
        low, high = _LOW * amplitude, _HIGH * amplitude

        # The start: the threshold crossed between the sample before and the
        # first, unless a gap lies between them.
        before = f.onset - 1
        start = (self._crossing(before, self.threshold)
                 if before >= 0 and self._joined(before) else float(t[f.onset]))

        # The peak: the run at or above 90% around the largest deficit.
        first = peak
        while first - 1 >= self.lo and self._joined(first - 1) and deficit[first - 1] >= high:
            first -= 1
        last = peak
        while last + 1 < self.hi and self._joined(last) and deficit[last + 1] >= high:
            last += 1
        peak_start = (self._crossing(first - 1, high)
                      if first - 1 >= self.lo and self._joined(first - 1) else float(t[first]))
        peak_end = (self._crossing(last, high)
                    if last + 1 < self.hi and self._joined(last) else float(t[last]))

        # The rise: back from the peak to the deficit's last crossing of 10%,
        # no further back than a window before the spike.
        rise = None
        earliest = max(self.lo, int(np.searchsorted(t, t[f.onset] - self.window_ns)))
        q = first - 1
        while q >= earliest and self._joined(q):
            if deficit[q] < low:
                rise = (peak_start - self._crossing(q, low)) / 1e9
                break
            q -= 1

        # The duration: to the last crossing of the threshold, interpolated
        # if the spike recovered, else to the last sample known above it.
        last_above = max(i for i in body if deficit[i] >= self.threshold)
        end = recovery = None
        if f.ending == "recovered":
            end = self._crossing(last_above, self.threshold)
            # The recovery: forward from the peak to the deficit's first
            # crossing of 10%, no further than a window after the spike.
            latest = min(self.hi, int(np.searchsorted(t, t[f.stop] + self.window_ns)))
            s = last + 1
            while s < latest and self._joined(s - 1):
                if deficit[s] <= low:
                    recovery = (self._crossing(s - 1, low) - peak_end) / 1e9
                    break
                s += 1
        duration = ((end if end is not None else float(t[last_above])) - start) / 1e9
        return Spike(start_ns=int(start), end_ns=None if end is None else int(end),
                     ending=f.ending, baseline_bytes=int(f.baseline),
                     amplitude_bytes=int(amplitude), peak_start_ns=int(peak_start),
                     peak_end_ns=int(peak_end), rise_s=rise, duration_s=duration,
                     recovery_s=recovery)
