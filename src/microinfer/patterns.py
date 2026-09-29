"""Synthetic contention: schedules for the simulator (#59).

The contention simulator (#58) takes device memory on a schedule of (time,
bytes taken); this makes schedules from patterns:

- **a step**: X bytes for T seconds, after a lead, then given back;
- **a sawtooth**: a climb from 0 to a peak over each period, then a drop;
- **Poisson spikes**: arrivals at a rate, each spike a trapezoid, rising,
  holding a plateau, and falling, whose amplitude and times are drawn from
  distributions. Spikes that overlap add, as two applications' memory does.

The distributions are fixed values, uniform ranges, or RQ1's own spikes
(from_rq1): the measures contention.py logged for every spike of the
recordings named, drawn from as they were measured. A spike's rise time and
recovery were measured from 10% to 90% of its amplitude and back, the
middle 0.8 of a linear ramp, so a ramp is drawn as that measure over 0.8;
its duration, time at or above the threshold, is drawn as the plateau.

Every schedule is deterministic given its seed, its levels are whole
granules, a ramp is steps of `resolution_s`, and it ends at 0: the simulator
keeps a schedule's last level until stopped, and these give everything back.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import benchlog
from .simulator import GRANULE, Schedule

MiB = 2**20
#: The share of a linear ramp that 10% to 90% of it takes.
_MEASURED_SHARE = 0.8


def _granules(level: float) -> int:
    return int(round(max(level, 0.0) / GRANULE)) * GRANULE


def _schedule(times, levels, end_s: float | None = None) -> Schedule:
    """A schedule from sampled levels, keeping only the changes, and a last
    point at `end_s` if given, so that the schedule says how long it is."""
    points = []
    for t, level in zip(times, levels):
        level = _granules(level)
        if not points or level != points[-1][1]:
            points.append((round(float(t), 6), level))
    if end_s is not None and end_s > points[-1][0]:
        points.append((round(float(end_s), 6), points[-1][1]))
    return Schedule(points)


def step(bytes_: int, hold_s: float, lead_s: float = 0.0) -> Schedule:
    """`bytes_` taken for `hold_s` seconds, after `lead_s` with nothing."""
    if bytes_ < 0 or hold_s <= 0 or lead_s < 0:
        raise ValueError("a step takes bytes >= 0 for a hold > 0 after a lead >= 0")
    points = ([(0.0, 0)] if lead_s else []) + [(lead_s, _granules(bytes_)),
                                               (lead_s + hold_s, 0)]
    return Schedule(points)


def sawtooth(peak_bytes: int, period_s: float, cycles: int,
             resolution_s: float = 0.1) -> Schedule:
    """`cycles` periods, each climbing from 0 to `peak_bytes` in steps of
    `resolution_s` and dropping back at once; then 0."""
    if peak_bytes < 0 or period_s <= 0 or cycles < 1 or resolution_s <= 0:
        raise ValueError("a sawtooth has a peak >= 0, a period > 0, cycles >= 1 and a "
                         "resolution > 0")
    steps = max(int(round(period_s / resolution_s)), 1)
    times, levels = [], []
    for cycle in range(cycles):
        for k in range(steps):
            times.append(cycle * period_s + k * period_s / steps)
            levels.append(peak_bytes * (k + 1) / steps)
    times.append(cycles * period_s)
    levels.append(0)
    return _schedule(times, levels)


# -- distributions -------------------------------------------------------------------


@dataclass(frozen=True)
class Fixed:
    value: float

    def draw(self, rng: np.random.Generator) -> float:
        return self.value


@dataclass(frozen=True)
class Uniform:
    low: float
    high: float

    def draw(self, rng: np.random.Generator) -> float:
        return float(rng.uniform(self.low, self.high))


@dataclass(frozen=True)
class Empirical:
    """Values measured, drawn from with replacement."""

    values: tuple[float, ...]

    def __post_init__(self):
        if not self.values:
            raise ValueError("an empirical distribution needs a value to draw")

    def draw(self, rng: np.random.Generator) -> float:
        return float(self.values[rng.integers(len(self.values))])


@dataclass(frozen=True)
class SpikeParameters:
    """What Poisson spikes are drawn from: arrivals per second, and each
    spike's amplitude, rise, plateau and fall."""

    rate_per_s: float
    amplitude_bytes: Fixed | Uniform | Empirical
    rise_s: Fixed | Uniform | Empirical
    duration_s: Fixed | Uniform | Empirical
    fall_s: Fixed | Uniform | Empirical


def poisson_spikes(params: SpikeParameters, duration_s: float, seed: int,
                   resolution_s: float = 0.02) -> Schedule:
    """Spikes arriving as a Poisson process over `duration_s`, sampled every
    `resolution_s`; the schedule ends at 0 once the last spike has fallen."""
    if params.rate_per_s <= 0 or duration_s <= 0 or resolution_s <= 0:
        raise ValueError("Poisson spikes need a rate, a duration and a resolution > 0")
    rng = np.random.default_rng(seed)
    spikes, t = [], float(rng.exponential(1 / params.rate_per_s))
    while t < duration_s:
        spikes.append((t, max(params.amplitude_bytes.draw(rng), 0.0),
                       max(params.rise_s.draw(rng), 0.0), max(params.duration_s.draw(rng), 0.0),
                       max(params.fall_s.draw(rng), 0.0)))
        t += float(rng.exponential(1 / params.rate_per_s))
    end = max([duration_s] + [s + r + d + f + resolution_s for s, _, r, d, f in spikes])
    times = np.arange(0.0, end + resolution_s, resolution_s)
    level = np.zeros_like(times)
    for start, amplitude, rise, plateau, fall in spikes:
        s = times - start
        up = np.clip(s / rise, 0, 1) if rise else (s >= 0).astype(float)
        after = s - rise - plateau
        down = 1 - (np.clip(after / fall, 0, 1) if fall else (after >= 0).astype(float))
        level += amplitude * np.where(s < 0, 0.0, np.minimum(up, down))
    level[-1] = 0.0
    return _schedule(times, level, end_s=end)


def from_rq1(log: str | Path = benchlog.DEFAULT_LOG, issues=(55, 56),
             action: str | None = None, span_s: float = 60.0) -> SpikeParameters:
    """Poisson-spike parameters from RQ1's recordings: every spike that the
    contention-trace entries of `issues` list, of `action` if one is named,
    as empirical distributions of amplitude, rise, plateau and fall. The
    rate is those spikes per second of the time they could happen in: the
    recordings' recorded time, or, for one action, the recordings times the
    `span_s` each spent on it (60 s in the protocol). A measure a spike
    lacks (a rise the trace did not show, a recovery of a spike that never
    recovered) adds nothing to its distribution; a measure no spike has, as
    no spike of an application that stays open recovers, is drawn as 0, an
    instant ramp."""
    spikes, seconds, recordings = [], 0.0, 0
    for entry in benchlog.read(log):
        if entry["kind"] != "contention-trace" or entry["config"].get("issue") not in issues:
            continue
        recordings += 1
        seconds += entry["results"]["recorded_hours"] * 3600
        spikes += [s for s in entry["results"]["spikes"]
                   if action is None or s.get("action") == action]
    if not spikes:
        raise ValueError(f"no spikes of {action or 'any action'} in the recordings of {issues}")

    def measured(key, scale):
        values = tuple(s[key] * scale for s in spikes if s.get(key) is not None)
        return Empirical(values) if values else Fixed(0.0)

    return SpikeParameters(
        rate_per_s=len(spikes) / (seconds if action is None else recordings * span_s),
        amplitude_bytes=measured("amplitude_mib", MiB),
        rise_s=measured("rise_ms", 1e-3 / _MEASURED_SHARE),
        duration_s=measured("duration_ms", 1e-3),
        fall_s=measured("recovery_ms", 1e-3 / _MEASURED_SHARE))
