"""A survival run against time, from the recordings its entry names (#212).

The log holds what a survival run came to and names, by sha256, the
recordings it was drawn from: the device's 50 Hz trace, the processes', the
simulator's events, the monitor's and the timeline. They are too large to
commit. The paper's survival timeline is to be drawn from the log alone, so
what it plots is read from them here, thinned, and logged as one
"survival-timeline" entry by tools/summarise_timeline.py. It measures
nothing anew.

Every time is in seconds from the simulator's take, and every size in MiB:

- **The headroom** is the trace's, the lowest sample of each bin, so that no
  dip is thinned away.
- **The cache** is the engine's process memory as the driver reports it, less
  what the engine held besides its cache at the trace's first sample, when
  the cache held the contention's position at FP16. The engine is the
  process, other than the simulator, that held most there. What the engine
  holds besides its cache changes by some tens of MiB as it goes from
  prefill to decode, so the cache is known to that.
- **The plans** are the timeline's: a run logs its plans' moves and not when
  each applied (ADR-0014, #207), and the steps after which more pages were
  below FP16 are where its downgrades showed. Its upgrades span the first
  batch to the last, as the entry's recovery gives them.

A run that ran out of memory ends where its engine last held its most: what
follows is its cache being freed.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from . import benchlog, recorder, survival
from .config import ModelConfig
from .engine import paged_cache_bytes
from .footprint import MIB

KIND = "survival-timeline"
#: The bin of a headroom's points, and the last second read.
BIN_S, UNTIL_S = 0.25, 620.0


def read(entry: dict, folders: Sequence[Path], *, bin_s: float = BIN_S,
         until_s: float = UNTIL_S) -> dict:
    """What the run of the survival or pulse-release `entry` was against
    time, to `until_s` seconds after its take, from its recordings in
    `folders`. FileNotFoundError if one is in none of them, ValueError if
    one is not the file the entry names, or if an adaptive run has no
    timeline."""
    config, results = entry["config"], entry["results"]
    events = _lines(_file(entry, ".events.jsonl", folders))
    take_ns = events[1]["applied_ns"]  # after the simulator's idle lead

    def since(t_ns) -> float:
        return round((int(t_ns) - take_ns) / 1e9, 3)

    simulator = json.loads(_file(entry, ".replay.json", folders).read_text())["simulator_pid"]
    _, _, procs = recorder.read_processes(_file(entry, ".procs.csv.gz", folders))
    engine = _engine(procs, simulator)
    adaptive = config["engine"] == "adaptive"
    steps = _steps(entry, folders, take_ns) if adaptive else []
    if results["ending"]["state"] == survival.OUT_OF_MEMORY:
        most = engine["used_bytes"].max()
        end = since(engine["t_mono_ns"][engine["used_bytes"] == most][-1])
    elif steps:
        end = steps[-1][0]
    else:
        end = since(engine["t_mono_ns"][-1])
    until = min(end, until_s)
    besides = int(engine["used_bytes"][0]) - paged_cache_bytes(
        ModelConfig.from_card(entry["model"]), results["contention_position"])
    _, trace = recorder.read(_file(entry, ".csv.gz", folders, but=".procs.csv.gz"))
    recovery = results.get("recovery")
    return {
        "entry": entry["sha256"], "ending": results["ending"]["state"], "end_s": end,
        "headroom_mib": _lowest([(since(t), round(b / MIB, 2)) for t, b in
                                 zip(trace["t_mono_ns"], trace["free_bytes"])
                                 if since(t) <= until], bin_s),
        "cache_mib": _changes([(since(t), round((int(b) - besides) / MIB, 2)) for t, b in
                               zip(engine["t_mono_ns"], engine["used_bytes"])
                               if since(t) <= until]),
        "simulator_mib": [[since(e["applied_ns"]), e["taken_bytes"] / MIB] for e in events[1:]
                          if since(e["applied_ns"]) <= until],
        "levels": [[since(e["t_mono_ns"]), e["level"]]
                   for e in _lines(_file(entry, ".pressure.jsonl", folders))
                   if since(e["t_mono_ns"]) <= until],
        "downgrades_s": [t for (_, before), (t, after) in zip(steps, steps[1:])
                         if after > before and t <= until],
        "upgrades_s": [recovery["first_upgrade_s"], recovery["last_upgrade_s"]]
        if recovery and recovery["first_upgrade_s"] is not None else None}


def _file(entry: dict, ending: str, folders: Sequence[Path], *, but: str | None = None) -> Path:
    """The recording with `ending` that `entry` names."""
    for name in entry["config"]["files"]:
        if not name.endswith(ending) or (but is not None and name.endswith(but)):
            continue
        if (path := benchlog.named_file(entry, name, folders)) is not None:
            return path
        raise FileNotFoundError(f"{name}, which entry {entry['sha256'][:8]} names, is in none "
                                f"of {', '.join(map(str, folders))}")
    raise ValueError(f"entry {entry['sha256'][:8]} names no {ending} recording: it has none, "
                     f"and no timeline can be read of it")


def _lines(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def _engine(procs: np.ndarray, simulator: int) -> np.ndarray:
    """The engine's samples: the process, not the simulator, that held most
    at the first."""
    first = procs[(procs["t_mono_ns"] == procs["t_mono_ns"][0]) & (procs["pid"] != simulator)]
    return procs[procs["pid"] == first["pid"][np.argmax(first["used_bytes"])]]


def _steps(entry: dict, folders: Sequence[Path], take_ns: int) -> list[tuple[float, int]]:
    """Each step's seconds from the take and its pages below FP16, None, of a
    cache with no full page, as none."""
    return [(round(s["t"] - take_ns / 1e9, 3), s["pages_below_fp16"] or 0)
            for s in _lines(_file(entry, ".timeline.jsonl", folders))]


def _lowest(points: Sequence[tuple[float, float]], bin_s: float) -> list[list[float]]:
    """The lowest of `points` in each `bin_s` of their times, where it was."""
    lowest: dict[int, tuple[float, float]] = {}
    for t, value in points:
        k = int(np.floor(t / bin_s))
        if k not in lowest or value < lowest[k][1]:
            lowest[k] = (t, float(value))
    return _changes([lowest[k] for k in sorted(lowest)])


def _changes(points: Sequence[tuple[float, float]]) -> list[list[float]]:
    """`points` without those equal to both neighbours, which a line drawn
    through the others passes through."""
    return [[t, v] for i, (t, v) in enumerate(points)
            if not (0 < i < len(points) - 1 and points[i - 1][1] == v == points[i + 1][1])]
