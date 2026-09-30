"""Evaluating the pressure monitor against the recorder's trace (#64, #65).

#45's measures of whether the monitor sees pressure in time, against a truth
the monitor has no part in (CONTEXT.md names each):

- **Ground truth is the recorder's.** Its 50 Hz trace of the driver's free
  memory, the same headroom the monitor polls, read by another thread on its
  own schedule, with the monitor's thresholds applied: a **true RED
  episode** is a run of samples below RED, from the first below to the
  first back at or above. The monitor's events never enter the truth.
- **Detection latency:** for each true RED episode of at least 100 ms, from
  its start to the first RED event from then until one poll after it ends
  (the recorder's sample and the monitor's poll need not fall together), in
  ms and in polls. Each event detects one episode at most. An episode that
  begins while the monitor is still RED from the one before needs no event:
  it is counted apart, as already RED, with no latency.
- **A false negative** is such an episode with no RED event. Those shorter
  than K + 1 polls are counted apart: the monitor reports a level only once
  it has held for K polls (ADR-0013), so an episode of 100 ms that ends
  before then is missed by design, not by fault.
- **A false positive** is a RED event with no true RED over the K polls that
  settled it, one poll either side. An episode shorter than 100 ms is
  neither missed nor, if caught, false: it was RED.

#64 applies them to a synthetic grid of pulses, amplitude x ramp x plateau:
grid_schedule makes it, on a base the simulator takes first so that
headroom starts in GREEN, and tools/evaluate_monitor.py runs it beside the
engine decoding with its monitor on. A **pulse** is a trapezoid the
simulator plays, not a spike: a spike is what spikes.py finds in a trace.

This is measurement, not inference, so it computes in NumPy on the host
(ADR-0002, amendment on its scope).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import patterns
from .monitor import POLL_S, RED, Level, PressureEvent, Thresholds
from .schedule import Schedule

#: The shortest true RED episode a monitor is expected to catch (#45).
MIN_EPISODE_S = 0.1
_NS = 1_000_000_000


@dataclass(frozen=True)
class Episode:
    """A true RED episode, on the recorder's clock."""

    start_ns: int
    end_ns: int

    @property
    def duration_s(self) -> float:
        return (self.end_ns - self.start_ns) / _NS


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
    return [Episode(int(t[s]), int(t[min(e, len(t) - 1)])) for s, e in zip(starts, ends)]


@dataclass(frozen=True)
class Score:
    episodes: int  # true RED episodes of MIN_EPISODE_S or more
    detectable: int  # of those, the ones lasting K + 1 polls or more
    latencies_ms: list[float]
    latencies_polls: list[float]
    missed_detectable: int  # false negatives the monitor could have caught
    missed_below_k: int  # false negatives shorter than K + 1 polls
    already_red: int  # episodes that began with the monitor still RED
    false_positives: int
    red_events: int

    @property
    def false_negatives(self) -> int:
        return self.missed_detectable + self.missed_below_k


def _level_at(events: list[PressureEvent], t_ns: int) -> Level | None:
    """The monitor's level at `t_ns`, by the last event at or before it."""
    before = [e for e in events if e.t_mono_ns <= t_ns]
    return before[-1].level if before else None


def score(episodes: list[Episode], events: list[PressureEvent], thresholds: Thresholds,
          poll_s: float = POLL_S, within: tuple[int, int] | None = None) -> Score:
    """The monitor's `events` against the true `episodes` (see the module):
    those that begin, and the RED events, `within` [from, to) ns if given.
    The monitor's level is read from all its events."""
    def inside(t: int) -> bool:
        return within is None or within[0] <= t < within[1]

    poll = int(poll_s * _NS)
    events = sorted(events, key=lambda e: e.t_mono_ns)
    reds = [e.t_mono_ns for e in events if e.level == RED and inside(e.t_mono_ns)]
    counted = [e for e in episodes if inside(e.start_ns) and e.duration_s >= MIN_EPISODE_S]
    detectable_s = (thresholds.persist_polls + 1) * poll_s
    used: set[int] = set()
    latencies, missed_detectable, missed_below_k, already = [], 0, 0, 0
    for episode in counted:
        if _level_at(events, episode.start_ns) == RED:
            already += 1
            continue
        caught = [t for t in reds if t not in used
                  and episode.start_ns <= t <= episode.end_ns + poll]
        if caught:
            used.add(caught[0])
            latencies.append(caught[0] - episode.start_ns)
        elif episode.duration_s >= detectable_s:
            missed_detectable += 1
        else:
            missed_below_k += 1
    settled = thresholds.persist_polls * poll
    false_positives = sum(
        t not in used and not any(e.start_ns <= t + poll and e.end_ns >= t - settled - poll
                                  for e in episodes)
        for t in reds)
    return Score(len(counted), sum(e.duration_s >= detectable_s for e in counted),
                 [ns / 1e6 for ns in latencies], [ns / poll for ns in latencies],
                 missed_detectable, missed_below_k, already, false_positives, len(reds))


# -- the grid ------------------------------------------------------------------------------


@dataclass(frozen=True)
class Cell:
    """One point of the grid: a pulse of `amplitude_bytes` above the base,
    ramping up and down over `ramp_s` each, linearly, and kept for
    `plateau_s`: the grid's duration. The pulse lasts ramp, plateau and
    ramp."""

    amplitude_bytes: int
    ramp_s: float
    plateau_s: float

    @property
    def pulse(self) -> patterns.Trapezoid:
        return patterns.Trapezoid(self.amplitude_bytes, self.ramp_s, self.plateau_s,
                                  self.ramp_s)


@dataclass(frozen=True)
class Pulse:
    """One pulse of the grid: its cell, and when it is due, in seconds from
    the schedule's start."""

    cell: int
    start_s: float


@dataclass(frozen=True)
class Grid:
    cells: list[Cell]
    schedule: Schedule
    pulses: list[Pulse]


def grid_schedule(cells: list[Cell], base_bytes: int, repeats: int, gap_s: float,
                  lead_s: float, resolution_s: float = 0.02, seed: int | None = None,
                  poll_s: float = POLL_S) -> Grid:
    """`base_bytes` taken at once and kept; after `lead_s`, each cell's pulse
    `repeats` times on top of it, cell after cell, each `gap_s` after the one
    before has fallen; and, `gap_s` after the last, everything given back.

    With a `seed`, each gap is longer by a draw from [0, poll_s), so that
    the pulses fall at every phase of the monitor's polls: without it, a
    grid whose times are whole polls would meet them at one phase, and
    every repeat would measure the same latency."""
    if repeats < 1 or gap_s <= 0 or lead_s < 0 or base_bytes < 0:
        raise ValueError("a grid repeats each cell at least once, with a gap > 0, and a "
                         "lead and a base >= 0")
    rng = None if seed is None else np.random.default_rng(seed)
    arrivals, pulses, t = [], [], 0.0
    for index, cell in enumerate(cells):
        for _ in range(repeats):
            arrivals.append((t, cell.pulse))
            pulses.append(Pulse(index, lead_s + t))
            t += cell.pulse.length_s + gap_s + (0.0 if rng is None else rng.uniform(0, poll_s))
    rendered = patterns.render(arrivals, 0.0, resolution_s)
    points = [(0.0, base_bytes)] + [(lead_s + s, base_bytes + b) for s, b in rendered.points
                                    if lead_s + s > 0]
    end = lead_s + rendered.duration_s + gap_s
    schedule = patterns.sampled([p for p, _ in points] + [end], [b for _, b in points] + [0])
    return Grid(list(cells), schedule, pulses)


def applied_starts(grid: Grid, started_ns: int, simulator_events: list[dict]) -> list[int]:
    """When each pulse of `grid`, run from `started_ns`, began to be applied:
    the simulator's first change due at or after its start (its events,
    simulator.run's), when that change was applied. A pulse whose change the
    simulator never made keeps its due time."""
    due = sorted((e["scheduled_ns"], e["applied_ns"]) for e in simulator_events)
    starts = []
    for pulse in grid.pulses:
        at = started_ns + int(pulse.start_s * _NS) - 1_000_000  # a ms for rounding
        starts.append(next((applied for scheduled, applied in due if scheduled >= at),
                           at + 1_000_000))
    return starts


def score_cells(grid: Grid, starts_ns: list[int], end_ns: int, episodes: list[Episode],
                events: list[PressureEvent], thresholds: Thresholds,
                poll_s: float = POLL_S) -> list[Score]:
    """A Score per cell of `grid`, whose pulses began at `starts_ns`: the
    episodes that began, and the RED events, from each of its pulses' start
    to the next pulse's, the last to `end_ns`, over all its repeats, summed."""
    windows: list[list[tuple[int, int]]] = [[] for _ in grid.cells]
    for pulse, lo, hi in zip(grid.pulses, starts_ns, starts_ns[1:] + [end_ns]):
        windows[pulse.cell].append((lo, hi))
    return [_sum([score(episodes, events, thresholds, poll_s, within=w) for w in cell])
            for cell in windows]


def _sum(scores: list[Score]) -> Score:
    return Score(*(sum((getattr(s, f) for s in scores), [] if f.startswith("latencies")
                       else 0) for f in Score.__dataclass_fields__))
