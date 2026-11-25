"""The survival timeline: two survival runs against time, from the recordings
their entries name (#212).

The log holds what a survival run came to and names, by sha256, the
recordings it was drawn from: the device's 50 Hz trace, the processes', the
simulator's events, the monitor's and the run's timeline of its steps (#207).
They are too large to commit. The paper's figure is to be drawn from the log
alone, so what it plots of each run is read from them here, thinned, and
logged as one "survival-timeline" entry of both runs by
tools/summarise_survival_timeline.py. It measures nothing anew.

Every time is in seconds from the simulator's take, and every size in MiB:

- **The headroom** is the trace's, the lowest sample of each bin, so that no
  dip is thinned away.
- **The cache** of an adaptive run is its positions' FP16 bytes at the
  effective bits its steps report. A static run logged no steps before #207,
  and its cache is the engine's process memory as the driver reports it, less
  what the engine held besides an FP16 cache of the contention's position at
  the recording's first sample. The engine is the process, other than the
  simulator, that held most there. That account holds through prefill, which
  the paper's static run ends in: what an engine holds besides its cache falls
  by 70 MiB as decoding begins (B6's S6 run).
- **The downgrades** are the steps that found more pages below FP16 than the
  step before: a run logs its plans' moves and not when each applied
  (ADR-0014, #207). Its upgrades span the first batch to the last, as the
  entry's recovery gives them.

A static run that ran out of memory ends where its engine last held its most:
what follows is its cache being freed.
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


def read(entry: dict, folders: Sequence[Path]) -> dict:
    """What the run of the survival or pulse-release `entry` was against
    time, to UNTIL_S seconds after its take, from its recordings in
    `folders`. FileNotFoundError if one is in none of them, ValueError if
    one is not the file the entry names, or if an adaptive run has no
    timeline of its steps."""
    results = entry["results"]
    events = _lines(_file(entry, ".events.jsonl", folders))
    take_ns = events[1]["applied_ns"]  # after the simulator's idle lead

    def since(t_ns) -> float:
        return round((int(t_ns) - take_ns) / 1e9, 3)

    cfg = ModelConfig.from_card(entry["model"])
    if entry["config"]["engine"] == "adaptive":
        steps = [(since(s["t"] * 1e9), s) for s in
                 _lines(_file(entry, ".timeline.jsonl", folders))]
        cache = [(t, _cache_bytes(cfg, s)) for t, s in steps]
        below = [(t, s["pages_below_fp16"] or 0) for t, s in steps]
        end = steps[-1][0]
    else:
        simulator = json.loads(_file(entry, ".replay.json", folders).read_text())["simulator_pid"]
        _, _, procs = recorder.read_processes(_file(entry, ".procs.csv.gz", folders))
        engine = _engine(procs, simulator)
        besides = int(engine["used_bytes"][0]) - paged_cache_bytes(
            cfg, results["contention_position"])
        cache = [(since(t), int(b) - besides) for t, b in
                 zip(engine["t_mono_ns"], engine["used_bytes"])]
        below = []
        # A run that ran out then frees its cache, which is no part of it.
        ran_out = results["ending"]["state"] == survival.OUT_OF_MEMORY
        held = engine["used_bytes"] == engine["used_bytes"].max()
        end = since(engine["t_mono_ns"][held][-1] if ran_out else engine["t_mono_ns"][-1])
    _, trace = recorder.read(_file(entry, ".csv.gz", folders, but=".procs.csv.gz"))
    start, until = since(trace["t_mono_ns"][0]), min(end, UNTIL_S)

    def shown(points) -> list[tuple[float, float]]:
        return [(t, v) for t, v in points if t <= until]

    recovery = results.get("recovery")
    return {
        "entry": entry["sha256"], "ending": results["ending"]["state"], "end_s": end,
        "headroom_mib": _corners(_lowest(shown(
            (since(t), _mib(b)) for t, b in zip(trace["t_mono_ns"], trace["free_bytes"])))),
        "cache_mib": _corners([(t, _mib(b)) for t, b in shown(_from(start, cache))]),
        "simulator_mib": [list(p) for p in shown(
            (since(e["applied_ns"]), _mib(e["taken_bytes"])) for e in events[1:])],
        "levels": [list(p) for p in shown(
            (since(e["t_mono_ns"]), e["level"])
            for e in _lines(_file(entry, ".pressure.jsonl", folders)))],
        "downgrades_s": [t for (_, before), (t, after) in zip(below, below[1:])
                         if after > before and t <= until],
        "upgrades_s": [recovery["first_upgrade_s"], recovery["last_upgrade_s"]]
        if recovery and recovery["first_upgrade_s"] is not None else None}


def _from(start: float, steps: Sequence[tuple[float, int]]) -> list[tuple[float, int]]:
    """`steps` from `start`, which begins with what the last step before it
    held."""
    held = [value for t, value in steps if t <= start]
    return [(start, v) for v in held[-1:]] + [(t, v) for t, v in steps if t > start]


def _mib(nbytes) -> float:
    return round(int(nbytes) / MIB, 2)


def _cache_bytes(cfg: ModelConfig, step: dict) -> int:
    """What the cache held at a step: its positions' FP16 bytes at its
    effective bits, which are 16 of a cache with no full page."""
    bits = step["effective_bits"]
    return round(paged_cache_bytes(cfg, step["position"]) * (16 if bits is None else bits) / 16)


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


def _lowest(points: Sequence[tuple[float, float]]) -> list[tuple[float, float]]:
    """The lowest of `points` in each BIN_S of their times, where it was."""
    lowest: dict[int, tuple[float, float]] = {}
    for t, value in points:
        k = int(np.floor(t / BIN_S))
        if k not in lowest or value < lowest[k][1]:
            lowest[k] = (t, value)
    return [lowest[k] for k in sorted(lowest)]


def _corners(points: Sequence[tuple[float, float]]) -> list[list[float]]:
    """`points` without those equal to both neighbours, which a line drawn
    through the others passes through."""
    return [[t, v] for i, (t, v) in enumerate(points)
            if not (0 < i < len(points) - 1 and points[i - 1][1] == v == points[i + 1][1])]
