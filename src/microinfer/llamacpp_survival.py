"""The survival experiment's contention played against llama.cpp (#211).

A llama.cpp server (microinfer.llamacpp) fills the same 32K generation as
the survival experiment's engine (#108), the simulator, a process of its
own, taking memory when its cache holds the position contention comes at
(ADR-0014's amendment, #211):

- **The prompt's first part is a completion of no tokens**, so that the
  cache holds exactly the contention's position, as a MicroInfer run's does
  at its chunk boundary, and waits there until the simulator has taken its
  first level.
- **The same bytes taken.** The rule that sets them, S5's shortfall or a
  scenario's base, is given the headroom an FP16 MicroInfer would have had
  there: what llama.cpp leaves, and the difference between what its cache
  holds, the whole context's from the start, and an FP16 cache of those
  positions (survival.counted_as_fp16).
- **The whole schedule.** llama.cpp's generation takes seconds where
  MicroInfer's took minutes, and would meet a few of S1's pulses. It is
  therefore made again, the cache cut back to the contention's position,
  until the schedule has been played to its end. S5's step is kept to the
  generation's end, one generation.

**What a run comes to** is how it ended, COMPLETE, OUT_OF_MEMORY or FAILED,
at what position and how long after the take; a server lost before the
contention came has met none, and is not a run. And **what the contention was
refused**: the most the simulator asked for and the device did not have.
An engine that holds its memory from the load is not made to run out by
what comes after; what comes after is.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import NamedTuple, Protocol

import numpy as np

from . import benchlog, llamacpp, nvml, recorder, rq3_summary, survival
from . import survival_matrix as matrix
from . import survival_summary
from .config import ModelConfig
from .footprint import MIB
from .schedule import Schedule

KIND = "llamacpp-survival"
SUMMARY_KIND = "llamacpp-summary"
#: The scenarios played, and the runs of each cell.
SCENARIOS = ("S1", "S5")
RUNS = 3
#: S5's shortfall, the survival experiment's (#108).
SHORTFALL_BYTES = int(survival_summary.S5_GENERATION["shortfall_mib"] * MIB)
#: A run that ended neither complete nor out of memory.
FAILED = "failed"
#: The MicroInfer conditions whose refused contention is given beside.
BESIDE = ("B1", "B8", "B2", "B4", "B6")

_now = time.monotonic


class BaselineRun(NamedTuple):
    """One run of llama.cpp: its cell, a cache type against a scenario, and
    which of the cell's runs it is, the scenario's seed."""

    cache_type: str
    scenario: str
    run: int

    @property
    def name(self) -> str:
        return f"{self.scenario}-{self.cache_type}-r{self.run}"


class Serving(Protocol):
    """What a run asks of a server (llamacpp.Server)."""

    buffers_mib: dict[str, float]

    def complete(self, ids: Sequence[int] | np.ndarray, new_tokens: int,
                 report: llamacpp.Report) -> dict: ...

    def log(self) -> str: ...


class Contention(NamedTuple):
    """The contention a run faces: its scenario, S1 or S5, and the run of
    its cell, the scenario's seed. S1 is made from RQ1's measures."""

    scenario: str
    seed: int
    rq1: matrix.Rq1

    def schedule(self, cfg: ModelConfig, headroom_bytes: int, *, positions: int, context: int,
                 simulator_context_bytes: int) -> Schedule:
        """What the simulator plays, from the headroom an FP16 MicroInfer
        would have at `positions` of `context`."""
        if self.scenario == "S5":
            return survival.step(survival.contention_bytes(
                cfg, headroom_bytes=headroom_bytes, positions=positions, context=context,
                simulator_context_bytes=simulator_context_bytes,
                shortfall_bytes=SHORTFALL_BYTES))
        scenario = matrix.SCENARIOS[self.scenario]
        base = matrix.base_bytes(headroom_bytes,
                                 leave_bytes=scenario.leave_bytes(self.seed, self.rq1),
                                 simulator_context_bytes=simulator_context_bytes)
        return matrix.raised(scenario.schedule(self.seed, self.rq1), base)


@dataclass
class Run:
    """What a run came to: its ending, the contention it met, and each
    generation's timings as the server gave them."""

    ending: dict
    contention_position: int
    #: The headroom once the server had loaded, and when the simulator was
    #: started: as it was, the whole cache held.
    loaded_headroom_bytes: int
    headroom_bytes: int
    #: The first level, what the simulator asked for and what it was given,
    #: and the most of any level it was refused.
    asked_bytes: int
    taken_bytes: int
    refused_bytes: int
    schedule: Schedule
    generations: list[dict] = field(default_factory=list)

    @property
    def survived(self) -> bool:
        return self.ending["state"] == survival.COMPLETE


def run(server: Serving, cfg: ModelConfig, prompt: np.ndarray, new_tokens: int, *,
        contention_at: int, out: str | Path, schedule: Callable[[int], Schedule]) -> Run:
    """Fill the server's cache to `contention_at` positions of `prompt`, have
    the simulator play what `schedule` makes of the headroom then, as an FP16
    MicroInfer would have it, and complete `prompt` and `new_tokens`, again
    and again from `contention_at` until the schedule has been played,
    while the recorder records the device to `out`. A server lost once the
    contention has come, or a completion of fewer tokens, ends the run where
    it stood; one lost before raises Lost."""
    out = Path(out)
    reached = tokens = 0

    def report(positions: int, decoded: int) -> None:
        nonlocal reached, tokens
        reached, tokens = positions, decoded

    loaded_headroom = nvml.memory().free
    server.complete(prompt[:contention_at], 0, report)
    headroom = nvml.memory().free
    played = schedule(survival.counted_as_fp16(
        cfg, headroom, positions=contention_at,
        held_bytes=round(server.buffers_mib["KV"] * MIB)))
    simulator = survival.Simulator(out)
    generations: list[dict] = []
    error = None
    try:
        simulator.play(played)
        took = _now()
        while not generations or _now() - took < played.duration_s:
            logged = len(server.log())  # an allocation that failed before was recovered
            report(contention_at, 0)
            if generations:  # cut the cache back to where the contention came
                server.complete(prompt[:contention_at], 0, report)
            began = _now()
            done = server.complete(prompt, new_tokens, report)
            report(len(prompt) + new_tokens - 1, int(done["tokens_predicted"]))
            if tokens != new_tokens:
                error = f"the server stopped after {tokens} of {new_tokens} tokens"
                reached = len(prompt) + max(tokens - 1, 0)
                break
            generations.append({"seconds": _now() - began, "timings": done.get("timings")})
    except llamacpp.Lost as exc:
        error = str(exc)
    finally:
        ended = _now()
        simulator.close()
    state = survival.COMPLETE if error is None else (
        survival.OUT_OF_MEMORY if llamacpp.out_of_memory(server.log()[logged:]) else FAILED)
    ending = {"state": state, "positions": reached, "tokens": tokens,
              "seconds": ended - took, "error": error}
    events = _events(recorder.companion(out, ".events.jsonl"))
    first = events[1]  # after the simulator's idle lead
    return Run(ending, contention_at, loaded_headroom, headroom, played.points[0][1],
               first["taken_bytes"], max(e["shortfall_bytes"] for e in events), played,
               generations)


def _events(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()]


# -- the matrix of its runs ----------------------------------------------------------------


def runs(cache_types: Sequence[str] = llamacpp.CACHE_TYPES,
         scenarios: Sequence[str] = SCENARIOS, count: int = RUNS) -> list[BaselineRun]:
    """The runs, every cell's first before any cell's second."""
    return [BaselineRun(c, s, r) for r in range(count) for s in scenarios for c in cache_types]


def logged(commit: str, build: dict, log: str | Path = benchlog.DEFAULT_LOG) -> set[BaselineRun]:
    """The runs logged at the code of `commit` (benchlog.same_code) with the
    binary and the model file of `build`."""
    if not Path(log).exists():
        return set()
    same_code = benchlog.same_code_as(commit, log)
    wanted = {k: build[k] for k in ("binary_sha256", "gguf_sha256")}
    return {_run_of(e) for e in benchlog.read(log)
            if e["kind"] == KIND and same_code(e["git_commit"])
            and all(e["config"]["build"][k] == v for k, v in wanted.items())}


def _run_of(entry: dict) -> BaselineRun:
    config = entry["config"]
    return BaselineRun(config["cache_type"], config["scenario"], config["run"])


def summarise(entries: Iterable[dict]) -> dict[str, dict[str, dict]]:
    """Each scenario's cells by cache type, the latest entry of each run: the
    runs that completed with their Wilson 95% interval, how each ended, where,
    when and why, and what each one's contention was given and refused."""
    latest = {_run_of(e): e for e in entries if e["kind"] == KIND}
    cells: dict[str, dict[str, list[dict]]] = {}
    for run in sorted(latest, key=lambda r: (r.scenario, llamacpp.CACHE_TYPES.index(
            r.cache_type), r.run)):
        cells.setdefault(run.scenario, {}).setdefault(run.cache_type, []).append(latest[run])
    return {scenario: {c: _cell_summary(es) for c, es in by_type.items()}
            for scenario, by_type in cells.items()}


def _cell_summary(entries: Sequence[dict]) -> dict:
    results = [e["results"] for e in entries]
    survived = sum(r["survived"] for r in results)
    return {"survived": survived, "runs": len(results),
            "wilson_95": list(rq3_summary.wilson(survived, len(results))),
            "endings": [r["ending"] for r in results],
            "refused_mib": [r["refused_mib"] for r in results],
            "taken_mib": [r["taken_mib"] for r in results],
            "kv_mib": results[0]["buffers_mib"]["KV"],
            "entries": [e["sha256"] for e in entries]}


def refused_mib_of(entry: dict, folders: Sequence[Path]) -> float | None:
    """The most a MicroInfer run's contention was refused, from the
    simulator's events its entry names, found in one of `folders`; None if
    none holds them. ValueError if what is there is not the file the entry
    names, by its sha256."""
    for name, sha256 in entry["config"].get("files", {}).items():
        if not name.endswith(".events.jsonl"):
            continue
        for folder in folders:
            path = Path(folder) / name
            if path.is_file():
                if benchlog.file_sha256(path) != sha256:
                    raise ValueError(f"{path} is not the file its entry names: its sha256 is "
                                     f"another")
                return max(e["shortfall_bytes"] for e in _events(path)) / MIB
    return None
