"""Evaluating the pressure monitor against the recorder's trace (#64, #65).

#45's measures of whether the monitor sees pressure in time, against a truth
the monitor has no part in:

- **Ground truth is the recorder's.** Its 50 Hz trace of the driver's free
  memory, the same headroom the monitor polls, read by another thread on its
  own schedule, with the monitor's thresholds applied: a **true RED
  episode** is a run of samples below RED, from the first below to the
  first back at or above. The monitor's events never enter the truth.
- **Detection latency:** for each true RED episode of at least 100 ms, from
  its start to the first RED event from then until one poll after it ends
  (the recorder's sample and the monitor's poll need not fall together).
- **A false negative** is such an episode with no RED event.
- **A false positive** is a RED event with no true RED over the K polls that
  settled it, one poll either side. An episode shorter than 100 ms is
  neither missed nor, if caught, false: it was RED.

#64 applies them to a synthetic grid of spikes, amplitude x rise time x
duration: grid_schedule makes it, on a base the simulator takes first so
that headroom starts in GREEN, and tools/evaluate_monitor.py runs it beside
the engine decoding with its monitor on.

This is measurement, not inference, so it computes in NumPy on the host
(ADR-0002, amendment on its scope).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import patterns
from .monitor import POLL_S, RED, PressureEvent, Thresholds
from .schedule import Schedule

#: The shortest true RED episode a monitor is expected to catch (#45).
MIN_EPISODE_S = 0.1

Episode = tuple[int, int]  # start and end, ns on the recorder's clock


def true_red(t_mono_ns: np.ndarray, free_bytes: np.ndarray,
             thresholds: Thresholds) -> list[Episode]:
    """The true RED episodes of a trace: each run of samples below RED, from
    its first sample to the first at or above; to the last sample if the
    trace ends in one."""
    t = np.asarray(t_mono_ns)
    below = np.concatenate([[False], np.asarray(free_bytes) < thresholds.red_below_bytes,
                            [False]])
    edges = np.diff(below.astype(np.int8))
    starts, ends = np.nonzero(edges == 1)[0], np.nonzero(edges == -1)[0]
    return [(int(t[s]), int(t[min(e, len(t) - 1)])) for s, e in zip(starts, ends)]


@dataclass(frozen=True)
class Score:
    episodes: int  # true RED episodes of MIN_EPISODE_S or more
    latencies_ms: list[float]
    false_negatives: int
    false_positives: int
    red_events: int


def score(episodes: list[Episode], events: list[PressureEvent], thresholds: Thresholds,
          poll_s: float = POLL_S) -> Score:
    """The monitor's `events` against the true `episodes` (see the module)."""
    poll = int(poll_s * 1e9)
    reds = sorted(e.t_mono_ns for e in events if e.level == RED)
    counted = [(s, e) for s, e in episodes if e - s >= MIN_EPISODE_S * 1e9]
    latencies, missed = [], 0
    for start, end in counted:
        caught = [t for t in reds if start <= t <= end + poll]
        if caught:
            latencies.append((caught[0] - start) / 1e6)
        else:
            missed += 1
    settled = thresholds.persist_polls * poll
    false = sum(not any(s <= t + poll and e >= t - settled - poll for s, e in episodes)
                for t in reds)
    return Score(len(counted), latencies, missed, false, len(reds))


# -- the grid ------------------------------------------------------------------------------


@dataclass(frozen=True)
class Cell:
    """One point of the grid: a spike of `amplitude_bytes` above the base,
    rising and falling over `rise_s` each, and kept for `plateau_s`."""

    amplitude_bytes: int
    rise_s: float
    plateau_s: float

    @property
    def shape(self) -> patterns.Trapezoid:
        return patterns.Trapezoid(self.amplitude_bytes, self.rise_s, self.plateau_s,
                                  self.rise_s)


@dataclass(frozen=True)
class Grid:
    schedule: Schedule
    spikes: list[tuple[int, float]]  # (cell index, start in s) of every spike


def grid_schedule(cells: list[Cell], base_bytes: int, repeats: int, gap_s: float,
                  lead_s: float, resolution_s: float = 0.02) -> Grid:
    """`base_bytes` taken at once and kept; after `lead_s`, each cell's spike
    `repeats` times on top of it, cell after cell, each `gap_s` after the one
    before has fallen; and, `gap_s` after the last, everything given back."""
    if repeats < 1 or gap_s <= 0 or lead_s < 0 or base_bytes < 0:
        raise ValueError("a grid repeats each cell at least once, with a gap > 0, and a "
                         "lead and a base >= 0")
    arrivals, spikes, t = [], [], 0.0
    for index, cell in enumerate(cells):
        for _ in range(repeats):
            arrivals.append((t, cell.shape))
            spikes.append((index, lead_s + t))
            t += cell.shape.length_s + gap_s
    rendered = patterns.render(arrivals, 0.0, resolution_s)
    points = [(0.0, base_bytes)] + [(lead_s + s, base_bytes + b) for s, b in rendered.points
                                    if lead_s + s > 0]
    end = lead_s + rendered.duration_s + gap_s
    return Grid(Schedule(patterns.sampled([p for p, _ in points] + [end],
                                          [b for _, b in points] + [0]).points), spikes)


def score_cells(grid: Grid, started_ns: int, episodes: list[Episode],
                events: list[PressureEvent], thresholds: Thresholds,
                poll_s: float = POLL_S) -> list[Score]:
    """A Score per cell of `grid`, run from `started_ns`: the episodes that
    began, and the events at, from each of its spikes' start to the next
    spike's, over all its repeats."""
    bounds = [started_ns + int(s * 1e9) for _, s in grid.spikes]
    bounds.append(started_ns + int(grid.schedule.duration_s * 1e9))
    cells = max(c for c, _ in grid.spikes) + 1
    mine: list[tuple[list[Episode], list[PressureEvent]]] = [([], []) for _ in range(cells)]
    for (cell, _), lo, hi in zip(grid.spikes, bounds, bounds[1:]):
        mine[cell][0].extend(e for e in episodes if lo <= e[0] < hi)
        mine[cell][1].extend(e for e in events if lo <= e.t_mono_ns < hi)
    return [score(eps, evs, thresholds, poll_s) for eps, evs in mine]
