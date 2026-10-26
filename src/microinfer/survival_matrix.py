"""The survival matrix (#190): every condition against every contention scenario.

The survival experiment (#108) runs one 32K generation against S5, a step
kept to the end. The matrix runs it against S1 to S4 in its place, under
each of the conditions B1 to B6, three runs a cell (ADR-0014, amendment
#190). Each scenario's schedule is a pure function of its seed and RQ1's
measures, so that every condition faces the same contention run for run:

- **S1, the pulse grid**: #64's 36 cells, 3 repeats each, its gaps
  lengthened by a seeded fraction of a poll.
- **S2, a pulse train**: RQ1's P90 amplitude, ramping up and down over
  RQ1's median rise, kept for the upgrade cooldown and repeated twice the
  cooldown apart, so that each gap lets the upgrades begin before the next
  pulse.
- **S3, Poisson arrivals** at RQ1's rate, every spike it found that
  recovered or lasted, each arrival one of them drawn whole: a recovered
  spike as it was measured, a lasting drop kept to the end, as the
  applications of a desktop accumulate.
- **S4, RQ1's replays**: what the other processes held in a with-engine
  recording, from its first span after the original engine's prefill; the
  run's seed chooses the recording.

**The base.** S1 to S3 play on a base the simulator takes first, which
leaves START_HEADROOM_BYTES, in GREEN, as #64's grid did; S4's leaves the
original's headroom, as #65's replays did. The base is set when the cache
reaches the position S5's step comes at, and the cache grows on from
there. It is counted as an FP16 engine there would have it, so that a
static INT4 or INT2 engine, whose cache holds less, faces the same bytes
taken as every other condition.

**How long.** A run ends with its generation, the simulator giving back what
it holds, so a schedule longer than the run is cut there. S2 and S3 are
made LENGTH_S long, longer than any run is expected to take.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from pathlib import Path
from typing import NamedTuple

import numpy as np

from . import benchlog, controller, evaluation, patterns, replay
from .contention import logged_recordings, pooled
from .footprint import MIB
from .schedule import Schedule
from .spikes import MEASURED_SHARE


class Condition(NamedTuple):
    """An engine of the matrix: its cache's tier, whether it adapts, and
    where an adaptive one's plans take their scores."""

    kv_tier: str
    adaptive: bool
    score_source: str | None


CONDITIONS = {
    "B1": Condition("FP16", False, None),
    "B2": Condition("INT4", False, None),
    "B3": Condition("INT2", False, None),
    "B4": Condition("FP16", True, "uniform"),
    "B5": Condition("FP16", True, "random"),
    "B6": Condition("FP16", True, "semantic"),
}
#: S5 is the survival experiment's own step (#108); the matrix adds S1-S4.
SCENARIOS = ("S1", "S2", "S3", "S4", "S5")
SCHEDULED = ("S1", "S2", "S3", "S4")
RUNS = 3

#: What S1-S3's base leaves when it is taken: #64's grid's start (#159).
START_HEADROOM_BYTES = 1536 * MIB
#: Each synthetic schedule's time on its base before its first pulse, #64's.
LEAD_S = 5.0
#: How long S2 and S3 are made: longer than a run is expected to take.
LENGTH_S = 7200.0
RESOLUTION_S = 0.02

#: #64's grid (tools/evaluate_monitor.py's defaults).
GRID_AMPLITUDES_MIB = (768, 1152, 1408)
GRID_RAMPS_S = (0.0, 0.25, 1.0)
GRID_PLATEAUS_S = (0.1, 0.2, 0.5, 2.0)
GRID_REPEATS = 3
GRID_GAP_S = 3.0

#: The with-engine recordings S4 replays, a run's seed choosing among them.
RECORDINGS = tuple(f"scenario-with-r{k}.csv.gz" for k in range(1, 6))


def condition_of(kv_tier: str, adaptive: bool, score_source: str | None) -> str:
    """The name of the condition an engine is, B1 to B6."""
    wanted = Condition(kv_tier, adaptive, score_source if adaptive else None)
    for name, condition in CONDITIONS.items():
        if condition == wanted:
            return name
    raise ValueError(f"no condition is a {'adaptive' if adaptive else 'static'} "
                     f"{kv_tier} engine with {score_source} scores")


def base_bytes(headroom_bytes: int, *, leave_bytes: int, simulator_context_bytes: int) -> int:
    """What the base takes, beyond the simulator's own CUDA context, still
    to come, to leave `leave_bytes` of `headroom_bytes`; the simulator
    takes it to the nearest granule. Raises ValueError for headroom below
    that."""
    room = headroom_bytes - simulator_context_bytes - leave_bytes
    if room < 0:
        raise ValueError(f"headroom is {headroom_bytes / MIB:.0f} MiB with the simulator's "
                         f"context to come, below the {leave_bytes / MIB:.0f} MiB the "
                         f"scenario starts from")
    return room


class Rq1(NamedTuple):
    """What S2 to S4 are made from: RQ1's spikes, each one shape, the
    seconds recorded, the pooled amplitude and rise RQ1 reports, and where
    its recordings are kept."""

    spikes: list[patterns.Measured]
    recorded_s: float
    p90_amplitude_bytes: float
    median_rise_s: float
    traces: Path

    @classmethod
    def read(cls, log: str | Path = benchlog.DEFAULT_LOG,
             traces: Path = patterns.TRACES) -> Rq1:
        spikes, seconds = patterns.rq1_spikes(log)
        found = pooled(logged_recordings(log))
        return cls(spikes, seconds, found["amplitude_mib"]["p90"] * MIB,
                   found["rise_ms"]["median"] * 1e-3 / MEASURED_SHARE, Path(traces))

    @property
    def rate_per_s(self) -> float:
        return len(self.spikes) / self.recorded_s


def raised(schedule: Schedule, base: int) -> Schedule:
    """`schedule` played on `base`, taken from its start: every level `base`
    higher but the last, 0, which gives the base back with the rest."""
    *levels, (end, last) = schedule.points
    if last != 0:
        raise ValueError("a scenario's schedule ends by giving everything back")
    return Schedule([(t, b + base) for t, b in levels] + [(end, 0)])


def grid(seed: int) -> evaluation.Grid:
    """S1: #64's grid, its gaps lengthened by `seed`'s draws."""
    cells = [evaluation.Cell(a * MIB, r, p) for a in GRID_AMPLITUDES_MIB for r in GRID_RAMPS_S
             for p in GRID_PLATEAUS_S]
    return evaluation.grid_schedule(cells, 0, GRID_REPEATS, GRID_GAP_S, LEAD_S, RESOLUTION_S,
                                    seed=seed)


def pulse_train(seed: int, rq1: Rq1, *,
                cooldown_s: float = controller.DEFAULT_UPGRADE_COOLDOWN_SECONDS,
                length_s: float = LENGTH_S) -> evaluation.Grid:
    """S2: pulses of RQ1's P90 amplitude, ramping over its median rise and
    kept for `cooldown_s`, each 2 * `cooldown_s` after the one before has
    fallen, for `length_s`: one cell of a grid, repeated."""
    cell = evaluation.Cell(rq1.p90_amplitude_bytes, rq1.median_rise_s, cooldown_s)
    gap = 2 * cooldown_s
    count = max(math.ceil((length_s - LEAD_S) / (cell.pulse.length_s + gap)), 1)
    return evaluation.grid_schedule([cell], 0, count, gap, LEAD_S, RESOLUTION_S, seed=seed)


def poisson_spikes(seed: int, rq1: Rq1, *, length_s: float = LENGTH_S) -> Schedule:
    """S3: arrivals as a Poisson process at RQ1's rate over `length_s`, after
    the lead, each one of RQ1's spikes drawn whole, a lasting drop kept to
    the end; shapes that overlap add."""
    rng = np.random.default_rng(seed)
    arrivals, t = [], float(rng.exponential(1 / rq1.rate_per_s))
    while t < length_s:
        drawn = rq1.spikes[int(rng.integers(len(rq1.spikes)))]
        shape = drawn.shape
        if drawn.lasting:
            shape = patterns.Trapezoid(shape.amplitude_bytes, shape.rise_s,
                                       max(length_s - t - shape.rise_s, 0.0), 0.0)
        arrivals.append((LEAD_S + t, shape))
        t += float(rng.exponential(1 / rq1.rate_per_s))
    return patterns.render(arrivals, LEAD_S + length_s, RESOLUTION_S)


def recording(seed: int, rq1: Rq1) -> Path:
    """S4's recording for a run's `seed`: run k replays the (k mod 5)-th."""
    return rq1.traces / RECORDINGS[seed % len(RECORDINGS)]


def others_held(seed: int, rq1: Rq1) -> replay.OthersHeld:
    """What the other processes held in S4's recording for `seed`, from its
    first span after the original engine's prefill (#65)."""
    path = recording(seed, rq1)
    return replay.held(path, start_s=replay.scenario_start_s(path))


def leave_bytes(scenario: str, seed: int, rq1: Rq1) -> int:
    """What the scenario's base leaves of the headroom."""
    if scenario == "S4":
        return others_held(seed, rq1).headroom_bytes
    return START_HEADROOM_BYTES


def schedule(scenario: str, seed: int, rq1: Rq1) -> Schedule:
    """The scenario's schedule for a run's `seed`, above its base (raised)."""
    if scenario == "S1":
        return grid(seed).schedule
    if scenario == "S2":
        return pulse_train(seed, rq1).schedule
    if scenario == "S3":
        return poisson_spikes(seed, rq1)
    if scenario == "S4":
        return replay.to_schedule(others_held(seed, rq1))
    raise ValueError(f"the scheduled scenarios are {SCHEDULED}, not {scenario!r}")


def describe(scenario: str, seed: int, rq1: Rq1) -> dict:
    """What a log entry's config names of a scenario's rule and its inputs."""
    if scenario == "S1":
        rule = {"amplitudes_mib": list(GRID_AMPLITUDES_MIB), "ramps_s": list(GRID_RAMPS_S),
                "plateaus_s": list(GRID_PLATEAUS_S), "repeats": GRID_REPEATS,
                "gap_s": GRID_GAP_S, "phase_seed": seed}
    elif scenario == "S2":
        cooldown = controller.DEFAULT_UPGRADE_COOLDOWN_SECONDS
        rule = {"amplitude_mib": rq1.p90_amplitude_bytes / MIB, "ramp_s": rq1.median_rise_s,
                "plateau_s": cooldown, "gap_s": 2 * cooldown, "phase_seed": seed,
                "from": "RQ1's P90 amplitude and median rise; the upgrade cooldown"}
    elif scenario == "S3":
        rule = {"rate_per_min": rq1.rate_per_s * 60, "spikes": len(rq1.spikes),
                "lasting": sum(s.lasting for s in rq1.spikes), "arrival_seed": seed,
                "from": "every RQ1 spike that recovered or lasted, drawn whole; a lasting "
                        "drop kept to the end"}
    elif scenario == "S4":
        path = recording(seed, rq1)
        rule = {"recording": path.name, "recording_sha256": benchlog.file_sha256(path),
                "from_s": replay.scenario_start_s(path)}
    else:
        raise ValueError(f"the scheduled scenarios are {SCHEDULED}, not {scenario!r}")
    return {"scenario": scenario, "lead_s": LEAD_S if scenario != "S4" else 0.0,
            "length_s": LENGTH_S if scenario in ("S2", "S3") else None,
            "start_headroom_mib": None if scenario == "S4" else START_HEADROOM_BYTES / MIB,
            **rule}


class Cell(NamedTuple):
    """One run of the matrix: its condition, scenario and run."""

    condition: str
    scenario: str
    run: int

    @property
    def name(self) -> str:
        return f"{self.scenario}-{self.condition}-r{self.run}"


def cells(conditions: Sequence[str] = tuple(CONDITIONS), scenarios: Sequence[str] = SCHEDULED,
          runs: int = RUNS) -> list[Cell]:
    """The matrix's runs, run by run, so that a matrix stopped part way has
    every cell's first runs before any cell's last."""
    return [Cell(c, s, r) for r in range(runs) for s in scenarios for c in conditions]


def cell_of(entry: dict) -> Cell | None:
    """The run of the matrix a "survival" entry is, None for one of no
    scenario's run: the survival experiment's own, before the matrix."""
    config = entry["config"]
    if entry["kind"] != "survival" or "condition" not in config:
        return None
    return Cell(config["condition"], config["scenario"], config["run"])


def logged(commit: str, model: str, log: str | Path = benchlog.DEFAULT_LOG) -> set[Cell]:
    """The runs of the matrix on `model` logged at the code of `commit`
    (benchlog.same_code): a matrix stopped part way is taken up from them."""
    if not Path(log).exists():
        return set()
    held = benchlog.same_code_as(commit, log)
    return {cell for e in benchlog.read(log)
            if (cell := cell_of(e)) is not None and e["model"] == model
            and held(e["git_commit"])}
