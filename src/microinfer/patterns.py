"""Synthetic contention: schedules for the simulator (#59).

The contention simulator (#58) takes device memory on a schedule of (time,
bytes taken); this makes schedules from patterns:

- **a step**: X bytes for T seconds, after a lead, then given back;
- **a sawtooth**: a climb from 0 to a peak over each period, then a drop;
- **trapezoids**: one spike of a given shape, or a train of them, rising,
  keeping a plateau and falling: the shapes #64's grid evaluates the
  pressure monitor on;
- **Poisson arrivals** of trapezoids whose amplitude, rise, plateau and fall
  are drawn from distributions. Shapes that overlap add, as two
  applications' memory does.

**What RQ1 measured.** RQ1's recordings (#55, #56) hold two kinds of
contention, and from_rq1 gives each its own parameters:

- **spikes** that recovered: memory taken and given back. Their rise and
  recovery were measured from 10% to 90% of the amplitude and back, the
  middle 0.8 of a linear ramp, so a ramp is that measure over 0.8. Their
  duration was time at or above the threshold, which includes the part of
  each ramp above it, so the plateau is the duration less that part.
- **lasting drops**: an application that opened and kept its memory, 45 of
  RQ1's 48. How long it was kept, RQ1 cannot say: the baseline holds for a
  window and no longer (spikes.py), and a scenario closes everything at
  action 9. So a drop's amplitude and rise are measured, and the time it is
  kept is the caller's to give.

Censored spikes, cut by a gap or the trace's end, measure only a lower bound
and are left out of both.

Every schedule is the same for its seed, its levels are whole granules, a
ramp is steps of `resolution_s`, and it ends at 0 and says how long it is:
the simulator keeps a schedule's last level until stopped, and these give
everything back.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import benchlog, recorder
from .contention import RQ1_ISSUES, logged_recordings
from .schedule import Schedule
from .spikes import MEASURED_SHARE

MiB = 2**20
#: Where RQ1's recordings are kept (docs/rq1-protocol.md).
TRACES = Path(__file__).resolve().parents[2] / "data" / "rq1"


@functools.cache
def granule_bytes() -> int:
    """The allocator's granule, read from the driver the first time a level
    is rounded: the schedule itself needs no device."""
    from . import _microinfer

    return _microinfer.granule_bytes()


def _granules(level: float) -> int:
    """The nearest whole number of granules, halves up."""
    g = granule_bytes()
    return int(np.floor(max(level, 0.0) / g + 0.5)) * g


def sampled(times, levels, end_s: float | None = None) -> Schedule:
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


def step(bytes_: int, for_s: float, lead_s: float = 0.0) -> Schedule:
    """`bytes_` taken for `for_s` seconds, after `lead_s` with nothing."""
    if bytes_ < 0 or for_s <= 0 or lead_s < 0:
        raise ValueError("a step takes bytes >= 0 for a time > 0 after a lead >= 0")
    points = ([(0.0, 0)] if lead_s else []) + [(lead_s, _granules(bytes_)),
                                               (lead_s + for_s, 0)]
    return Schedule(points)


def sawtooth(peak_bytes: int, period_s: float, cycles: int,
             resolution_s: float = 0.1) -> Schedule:
    """`cycles` periods, each climbing from 0 to `peak_bytes` in steps of
    `resolution_s` and dropping back to 0 at its end."""
    steps = int(round(period_s / resolution_s)) if resolution_s > 0 else 0
    if peak_bytes < 0 or period_s <= 0 or cycles < 1 or steps < 2:
        raise ValueError("a sawtooth has a peak >= 0, cycles >= 1, and a period of at least "
                         "two steps of its resolution")
    times, levels = [], []
    for cycle in range(cycles):
        for k in range(steps):
            times.append(cycle * period_s + k * period_s / steps)
            levels.append(peak_bytes * k / (steps - 1))
    return sampled(times + [cycles * period_s], levels + [0])


# -- shapes --------------------------------------------------------------------------


@dataclass(frozen=True)
class Trapezoid:
    """One spike's shape: up linearly over rise_s, level for plateau_s, down
    over fall_s; a ramp of 0 is a sheer edge."""

    amplitude_bytes: float
    rise_s: float
    plateau_s: float
    fall_s: float

    @property
    def length_s(self) -> float:
        return self.rise_s + self.plateau_s + self.fall_s


def render(arrivals: list[tuple[float, Trapezoid]], length_s: float,
           resolution_s: float) -> Schedule:
    """The sum of `arrivals`, sampled every `resolution_s`, over `length_s`
    or until the last has fallen, then 0."""
    if resolution_s <= 0:
        raise ValueError("a resolution is > 0")
    end = max([length_s] + [t + shape.length_s + resolution_s for t, shape in arrivals])
    times = np.arange(0.0, end + resolution_s, resolution_s)
    level = np.zeros_like(times)
    for start, shape in arrivals:
        s = times - start
        up = np.clip(s / shape.rise_s, 0, 1) if shape.rise_s else (s >= 0).astype(float)
        after = s - shape.rise_s - shape.plateau_s
        down = 1 - (np.clip(after / shape.fall_s, 0, 1) if shape.fall_s
                    else (after >= 0).astype(float))
        level += shape.amplitude_bytes * np.where(s < 0, 0.0, np.minimum(up, down))
    level[-1] = 0.0
    return sampled(times, level, end_s=end)


def trapezoids(shape: Trapezoid, count: int = 1, gap_s: float = 0.0, lead_s: float = 0.0,
               resolution_s: float = 0.02) -> Schedule:
    """`count` spikes of `shape`, the first after `lead_s`, each `gap_s`
    after the one before has fallen: a known spike, or a train of them."""
    if count < 1 or gap_s < 0 or lead_s < 0:
        raise ValueError("a train has count >= 1, and a gap and a lead >= 0")
    arrivals = [(lead_s + k * (shape.length_s + gap_s), shape) for k in range(count)]
    return render(arrivals, 0.0, resolution_s)


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


Distribution = Fixed | Uniform | Empirical


@dataclass(frozen=True)
class ShapeParameters:
    """What Poisson arrivals are drawn from: arrivals per second, and each
    one's amplitude, rise, plateau and fall."""

    rate_per_s: float
    amplitude_bytes: Distribution
    rise_s: Distribution
    plateau_s: Distribution
    fall_s: Distribution

    def draw(self, rng: np.random.Generator) -> Trapezoid:
        def positive(d: Distribution) -> float:
            return max(d.draw(rng), 0.0)

        return Trapezoid(positive(self.amplitude_bytes), positive(self.rise_s),
                         positive(self.plateau_s), positive(self.fall_s))


def poisson(params: ShapeParameters, length_s: float, seed: int,
            resolution_s: float = 0.02) -> Schedule:
    """Arrivals as a Poisson process over `length_s`, each a trapezoid drawn
    from `params`; the schedule ends at 0 once the last has fallen."""
    if params.rate_per_s <= 0 or length_s <= 0:
        raise ValueError("Poisson arrivals need a rate and a length > 0")
    rng = np.random.default_rng(seed)
    arrivals, t = [], float(rng.exponential(1 / params.rate_per_s))
    while t < length_s:
        arrivals.append((t, params.draw(rng)))
        t += float(rng.exponential(1 / params.rate_per_s))
    return render(arrivals, length_s, resolution_s)


# -- RQ1's parameters -------------------------------------------------------------------


def _span_seconds(trace: str, traces: Path) -> dict[str, float] | None:
    """Seconds each action spent in a recording, from its labels file, and
    the time between them as "(between actions)"; None without the file."""
    path = recorder.labels_path(traces / trace)
    if not path.exists():
        return None
    _, labels = recorder.read_labels(path)
    seconds: dict[str, float] = {}
    started: dict[str, int] = {}
    for t, event, action in zip(labels["t_mono_ns"], labels["event"], labels["action"]):
        if event == recorder.START:
            started[action] = int(t)
        elif action in started:
            seconds[action] = seconds.get(action, 0.0) + (int(t) - started.pop(action)) / 1e9
    return seconds


def from_rq1(kind: str = "spikes", action: str | None = None,
             log: str | Path = benchlog.DEFAULT_LOG, issues=RQ1_ISSUES,
             traces: Path = TRACES, kept_s: Distribution | None = None) -> ShapeParameters:
    """Parameters from RQ1's recordings, of `kind` "spikes" (those that
    recovered) or "drops" (lasting drops), of `action` if one is named.

    Amplitude and rise are measured for both kinds; plateau and fall only
    for spikes. A drop is kept for `kept_s`, which the caller gives, and
    given back at once. The rate is those spikes per second of the time they
    could happen in: each recording's recorded time, or, for one action, the
    time its spans took, from the labels files beside the traces; a
    recording without the action adds nothing, and one whose labels are
    missing raises."""
    if kind not in ("spikes", "drops"):
        raise ValueError(f"the kinds are 'spikes' and 'drops', not {kind!r}")
    if kind == "drops" and kept_s is None:
        raise ValueError("RQ1 cannot say how long a lasting drop is kept: give kept_s")
    ending = "recovered" if kind == "spikes" else "lasting"
    chosen, seconds = [], 0.0
    for entry in logged_recordings(log, issues):
        results = entry["results"]
        spikes = [s for s in results["spikes"] if s["ending"] == ending
                  and (action is None or s.get("action") == action)]
        if action is None:
            seconds += results["recorded_hours"] * 3600
        else:
            spans = _span_seconds(entry["config"]["trace"], Path(traces))
            if spans is None:
                raise FileNotFoundError(
                    f"no labels beside {entry['config']['trace']} in {traces}: the time "
                    f"spent on {action} is read from them")
            if action == "(between actions)":
                seconds += results["recorded_hours"] * 3600 - sum(spans.values())
            else:
                seconds += spans.get(action, 0.0)
        threshold = entry["config"].get("threshold_mib", 64) * MiB
        chosen += [(s, threshold) for s in spikes]
    if not chosen:
        raise ValueError(f"no {kind} of {action or 'any action'} in the recordings of {issues}")

    def ramp(ms):
        return None if ms is None else ms * 1e-3 / MEASURED_SHARE

    def measured(values):
        values = tuple(v for v in values if v is not None)
        return Empirical(values) if values else Fixed(0.0)

    amplitudes = measured(s["amplitude_mib"] * MiB for s, _ in chosen)
    rises = [ramp(s.get("rise_ms")) for s, _ in chosen]
    rate = len(chosen) / seconds
    if kind == "drops":
        return ShapeParameters(rate, amplitudes, measured(rises), kept_s, Fixed(0.0))

    falls = [ramp(s.get("recovery_ms")) for s, _ in chosen]
    plateaus = []
    for (s, threshold), rise, fall in zip(chosen, rises, falls):
        above = 1 - min(threshold / (s["amplitude_mib"] * MiB), 1.0)  # of each ramp
        plateaus.append(max(s["duration_ms"] * 1e-3 - ((rise or 0) + (fall or 0)) * above, 0.0))
    return ShapeParameters(rate, amplitudes, measured(rises), measured(plateaus), measured(falls))
