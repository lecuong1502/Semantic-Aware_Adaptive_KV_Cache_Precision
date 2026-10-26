#!/usr/bin/env python3
"""Evaluate the pressure monitor on synthetic or replayed contention (#64, #65).

    .venv/bin/python tools/evaluate_monitor.py grid --out data/monitor/grid.csv.gz --issue 64
        [--amplitudes 768 1152 1408] [--ramps 0 0.25 1] [--plateaus 0.1 0.2 0.5 2]
        [--repeats 3] [--gap 3] [--start-headroom 1536] [--persist-polls 3]
    .venv/bin/python tools/evaluate_monitor.py replay --recording data/rq1/scenario-with-r1.csv.gz
        --out data/monitor/replay-with-r1.csv.gz --issue 65 [--from S] [--to S]
    .venv/bin/python tools/evaluate_monitor.py rescore --out data/monitor/grid-k1.csv.gz --issue 159

Either way the engine holds a generation (Engine.hold) with its pressure
monitor on, recording every event between steps. Once its cache has stopped
growing, the contention simulator, a process of its own, takes a base and
plays a schedule on it while the recorder records the device throughout
(microinfer.replay.replay).

- `grid` plays every cell of a grid of pulses, amplitude (MiB above the
  base) x ramp, up and down (s) x plateau (s), --repeats times, --gap
  seconds apart and a seeded fraction of a poll more, so that the pulses
  meet the polls at every phase. The base leaves --start-headroom MiB, in
  GREEN.
- `replay` plays what the other processes held in an RQ1 recording
  (microinfer.replay.held), from its first span after the engine's prefill,
  unless --from says otherwise: through the prefill the original engine
  grew, and the headroom it left with it. The base leaves the headroom the original
  had with the others at their least, so that the machine has the
  original's headroom throughout, whatever this engine holds.

The truth is the recorder's trace with the monitor's thresholds applied, and
the monitor is scored against it (microinfer.evaluation): detection latency,
false negatives and false positives, and for a grid, per cell. False
negatives shorter than K + 1 polls and a recorder sample are counted apart:
ADR-0013's K cannot be sure to catch them. The result is logged as a
"monitor-evaluation" entry with the sha256 of every file, the monitor's
events among them (.pressure.jsonl), so that it can be scored again without
running again.

For the grid, --persist-polls sets the monitor's K, ADR-0013's 3 by
default, its thresholds left as they are. #159 runs the grid at K = 1, 3
and 5, each to an --out of its own, all three at one code, so that K alone
differs between them:

    .venv/bin/python tools/evaluate_monitor.py grid --out data/monitor/grid-k1.csv.gz \
        --issue 159 --persist-polls 1

and the same with 3 and 5. The summary for RQ2 takes ADR-0013's K alone, so
the K = 3 grid supersedes #64's there; tools/summarise_monitor.py
--hysteresis compares the grid at each.

`rescore` scores a grid's evaluation again from its files, the trace and
the monitor's events beside it, without running again: the latest entry
whose files they are, their sha256 checked. It logs a "monitor-evaluation"
entry of the same config, naming the entry it rescores, which supersedes it
as the latest. #159 rescored its grids once each pulse's cell began when
the simulator began its change, not when the change returned
(evaluation.pulse_starts): a sheer edge's true RED began before, in the
cell of the pulse before.

Close every application first: the desktop moves headroom too, and what it
does counts against the monitor as if the simulator had done it. Refuses a
dirty tree, and an --out that exists.
"""

from __future__ import annotations

import argparse
import json
import signal
import sys
import threading
import time
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from microinfer import Engine, benchlog, evaluation, monitor, nvml, recorder, replay  # noqa: E402
from microinfer.contention import spread  # noqa: E402
from microinfer.engine import DECODING  # noqa: E402
from microinfer.footprint import MIB  # noqa: E402
from microinfer.schedule import Schedule  # noqa: E402

SEED = 64
#: How long headroom settles after the engine's cache stops growing.
SETTLE_S = 2.0


class GridWorkload:
    """A grid of pulses on a base that leaves --start-headroom (#64)."""

    def __init__(self, args):
        self.args = args
        self.cells = [evaluation.Cell(int(a * MIB), r, p) for a in args.amplitudes
                      for r in args.ramps for p in args.plateaus]
        self.grid = None

    def leave_bytes(self) -> int:
        return int(self.args.start_headroom * MIB)

    def schedule(self, base: int) -> Schedule:
        a = self.args
        self.grid = evaluation.grid_schedule(self.cells, base, a.repeats, a.gap, a.lead,
                                             seed=SEED)
        print(f"{len(self.cells)} cells x {a.repeats} over "
              f"{self.grid.schedule.duration_s:.0f} s, on a base of {base / MIB:.0f} MiB",
              flush=True)
        return self.grid.schedule

    @classmethod
    def from_config(cls, config: dict, out: Path) -> GridWorkload:
        """The grid an entry's `config` logged, its trace at `out`: the inverse
        of config() (#159)."""
        return cls(argparse.Namespace(
            out=out, amplitudes=config["amplitudes_mib"], ramps=config["ramps_s"],
            plateaus=config["plateaus_s"], repeats=config["repeats"], gap=config["gap_s"],
            lead=config["lead_s"], start_headroom=config["start_headroom_mib"]))

    def config(self) -> dict:
        a = self.args
        return {"workload": "synthetic grid", "repeats": a.repeats, "gap_s": a.gap,
                "lead_s": a.lead, "phase_seed": SEED, "start_headroom_mib": a.start_headroom,
                "amplitudes_mib": a.amplitudes, "ramps_s": a.ramps, "plateaus_s": a.plateaus,
                "duration": "a pulse's plateau; it lasts ramp + plateau + ramp"}

    def results(self, started: int, end: int, episodes, events, thresholds) -> dict:
        changes = [json.loads(line) for line in recorder.companion(
            self.args.out, ".events.jsonl").read_text().splitlines()]
        starts = evaluation.pulse_starts(self.grid, started, changes)
        per_cell = evaluation.score_cells(self.grid, starts, end, episodes, events, thresholds)
        return {"cells": [{**asdict(c), "amplitude_mib": c.amplitude_bytes / MIB,
                           **evaluation.summary(s)}
                          for c, s in zip(self.cells, per_cell)]}


class ReplayWorkload:
    """What the other processes held in an RQ1 recording, on a base that
    leaves the original's headroom (#65)."""

    def __init__(self, args):
        self.args = args
        if args.start_s is None:
            args.start_s = replay.scenario_start_s(args.recording)
        self.original = replay.held(args.recording, start_s=args.start_s, end_s=args.end_s)

    def leave_bytes(self) -> int:
        return self.original.headroom_bytes

    def schedule(self, base: int) -> Schedule:
        schedule = replay.to_schedule(self.original, base_bytes=base)
        print(f"replaying {self.args.recording.name} from {self.args.start_s:.0f} s, "
              f"{schedule.duration_s:.0f} s, on a base of {base / MIB:.0f} MiB", flush=True)
        return schedule

    def config(self) -> dict:
        a = self.args
        return {"workload": "replayed RQ1 recording", "recording": a.recording.name,
                "recording_sha256": benchlog.file_sha256(a.recording),
                "from_s": a.start_s, "to_s": a.end_s, "engine_pid": self.original.engine_pid,
                "original_headroom_mib": self.original.headroom_bytes / MIB}

    def results(self, started: int, end: int, episodes, events, thresholds) -> dict:
        # The original's own true RED, over the samples replayed.
        _, samples = recorder.read(self.args.recording)
        kept = np.isin(samples["t_mono_ns"], self.original.t_mono_ns)
        original = evaluation.true_red(samples["t_mono_ns"][kept],
                                       samples["free_bytes"][kept], thresholds)
        return {"original_red_episodes": sum(e.duration_s >= evaluation.MIN_EPISODE_S
                                             for e in original)}


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    workloads = parser.add_subparsers(dest="workload", required=True)
    grid_args = workloads.add_parser("grid", help="a synthetic grid of pulses (#64)")
    replay_args = workloads.add_parser("replay", help="a replayed RQ1 recording (#65)")
    rescore_args = workloads.add_parser("rescore", help="a grid's evaluation scored again "
                                                        "from its files (#159)")
    for sub in (grid_args, replay_args, rescore_args):
        sub.add_argument("--out", type=Path, required=True, help="the recorder's trace")
        sub.add_argument("--issue", type=int, required=True)
        sub.add_argument("--log", type=Path, default=benchlog.DEFAULT_LOG)
    for sub in (grid_args, replay_args):
        sub.add_argument("--model", default="qwen2.5-1.5b-instruct")
        sub.add_argument("--prompt-positions", type=int, default=1024)
        sub.add_argument("--decode-positions", type=int, default=64,
                         help="the span the hold decodes again and again")
    grid_args.add_argument("--amplitudes", type=float, nargs="+", default=[768, 1152, 1408],
                           help="MiB above the base")
    grid_args.add_argument("--ramps", type=float, nargs="+", default=[0.0, 0.25, 1.0],
                           help="seconds each pulse ramps up, and down")
    grid_args.add_argument("--plateaus", type=float, nargs="+", default=[0.1, 0.2, 0.5, 2.0],
                           help="seconds each pulse is kept at its amplitude: the grid's "
                                "duration")
    grid_args.add_argument("--repeats", type=int, default=3)
    grid_args.add_argument("--gap", type=float, default=3.0, help="seconds between pulses")
    grid_args.add_argument("--lead", type=float, default=5.0,
                           help="seconds on the base before the first pulse")
    grid_args.add_argument("--persist-polls", type=int, default=monitor.DEFAULT.persist_polls,
                           help="K, the polls a new level must hold for (#159); ADR-0013's "
                                "by default")
    grid_args.add_argument("--start-headroom", type=float, default=1536,
                           help="MiB of headroom the base leaves")
    replay_args.add_argument("--recording", type=Path, required=True,
                             help="an RQ1 device trace")
    replay_args.add_argument("--from", dest="start_s", type=float,
                             help="seconds after the recording's first sample; its first "
                                  "span after the engine's prefill by default")
    replay_args.add_argument("--to", dest="end_s", type=float,
                             help="the same; its end if omitted, and only a replay to the "
                                  "end enters the summary")
    args = parser.parse_args(argv)
    if args.workload == "rescore":
        return rescore(parser, args)
    k = getattr(args, "persist_polls", monitor.DEFAULT.persist_polls)  # the grid's alone
    if k < 1:
        parser.error(f"--persist-polls is at least 1; got {k}")
    if args.out.exists():
        parser.error(f"{args.out} exists; an evaluation does not overwrite a recording")
    args.out.parent.mkdir(parents=True, exist_ok=True)  # before minutes of loading
    if benchlog.environment(args.log)["git_dirty"]:
        parser.error("tracked files have uncommitted changes; commit first, so that the "
                     "entry names the code that produced it")
    workload = GridWorkload(args) if args.workload == "grid" else ReplayWorkload(args)

    stop = threading.Event()
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: stop.set())

    engine = Engine(REPO / "models" / args.model, kv_tier="FP16")
    engine.load_weights()
    rng = np.random.default_rng(SEED)
    prompt = rng.integers(1000, engine.config.vocab_size - 1000,
                          args.prompt_positions).astype(np.int32)
    context = args.prompt_positions + args.decode_positions
    full = threading.Event()  # the cache holds the whole context: it grows no more

    def report(state: str, position: int, token: int | None) -> None:
        if state == DECODING and position == context:
            full.set()

    halt = threading.Event()
    holding = threading.Thread(
        target=lambda: engine.hold(prompt, context=context, stop=halt, report=report),
        name="hold")
    watching = engine.start_monitor(monitor.Monitor(
        thresholds=replace(monitor.DEFAULT, persist_polls=k)))
    holding.start()
    try:
        while not full.wait(0.5):
            if stop.is_set() or not holding.is_alive():
                raise SystemExit("stopped before the hold's cache was full")
        context_bytes = replay.simulator_context_bytes(args.out.parent)
        time.sleep(SETTLE_S)
        # The simulator's context is taken as it starts, after this reading.
        headroom = nvml.memory().free - context_bytes
        base = int(headroom - workload.leave_bytes())
        if base < 0:
            raise SystemExit(f"headroom is {headroom / MIB:.0f} MiB, below the "
                             f"{workload.leave_bytes() / MIB:.0f} MiB the evaluation "
                             f"starts from")
        schedule = workload.schedule(base)
        started = replay.replay(schedule, args.out, stop=stop)
    finally:
        halt.set()
        holding.join()
        engine.stop_monitor()

    thresholds = watching.thresholds
    events = [r.event for r in engine.pressure_events]
    end = started + int(schedule.duration_s * 1e9)
    overall, scores = score_against_trace(workload, started, end, events, thresholds)
    results = {"overall": evaluation.summary(overall), "events_recorded": len(events),
               "monitor_error": None if engine.monitor_error is None
               else str(engine.monitor_error),
               "missed_polls": watching.missed, **scores}

    pressure = recorder.companion(args.out, ".pressure.jsonl")
    write_pressure(pressure, engine.pressure_events)
    files = replay.files(args.out) + [pressure]
    config = {"issue": args.issue, "base_bytes": base,
              "simulator_context_bytes": context_bytes, "poll_s": monitor.POLL_S,
              "thresholds": asdict(thresholds), "min_episode_s": evaluation.MIN_EPISODE_S,
              **workload.config(),
              "files": {f.name: benchlog.file_sha256(f) for f in files if f.exists()}}
    benchlog.append("monitor-evaluation", model=args.model, context_length=context,
                    precision_tiers={"FP16": 1.0}, config=config, results=results,
                    log=args.log)
    print(json.dumps({"episodes": overall.episodes, "latency_ms": spread(overall.latencies_ms),
                      "false_negatives_detectable": overall.missed_detectable,
                      "false_negatives_below_k": overall.missed_below_k,
                      "already_red": overall.already_red,
                      "false_positives": overall.false_positives,
                      **{k: v for k, v in results.items() if k == "original_red_episodes"}}))
    return 0


def write_pressure(path: Path, records) -> None:
    """The monitor's events as the engine recorded them, a JSON line each,
    beside the trace: the truth can be scored again without running again."""
    path.write_text("".join(
        json.dumps({**asdict(r.event), "positions_held": r.positions_held,
                    "drained_ns": r.drained_ns}) + "\n" for r in records))


def read_pressure(path: Path) -> list[monitor.PressureEvent]:
    """The events write_pressure wrote, as far as scoring reads them: when,
    at which poll, from and to which level, on what headroom."""
    return [monitor.PressureEvent(r["t_mono_ns"], r["poll"],
                                  r["previous"] and monitor.Level(r["previous"]),
                                  monitor.Level(r["level"]), r["headroom_bytes"])
            for r in map(json.loads, path.read_text().splitlines())]


def score_against_trace(workload, started: int, end: int, events, thresholds):
    """The monitor's `events` scored against the trace at the workload's
    --out, over [started, end): overall, and the workload's own results."""
    _, samples = recorder.read(workload.args.out)
    episodes = evaluation.true_red(samples["t_mono_ns"], samples["free_bytes"], thresholds)
    overall = evaluation.score(episodes, events, thresholds, within=(started, end))
    return overall, workload.results(started, end, episodes, events, thresholds)


def rescore(parser: argparse.ArgumentParser, args: argparse.Namespace) -> int:
    """A grid's evaluation scored again from its files at --out (#159): the
    run's own entry, never an earlier rescore of it."""
    if benchlog.environment(args.log)["git_dirty"]:
        parser.error("tracked files have uncommitted changes; commit first, so that the "
                     "entry names the code that scored it")
    trace = benchlog.file_sha256(args.out)
    made_with = [e for e in benchlog.read(args.log) if e["kind"] == "monitor-evaluation"
                 and "rescores" not in e["config"]
                 and e["config"]["files"].get(args.out.name) == trace]
    if not made_with:
        parser.error(f"no evaluation in the log was made with {args.out}")
    entry = made_with[-1]
    config = entry["config"]
    if config["workload"] != "synthetic grid" or config["phase_seed"] != SEED:
        parser.error("only a grid's evaluation, of this tool's phase seed, is rescored")
    if config["min_episode_s"] != evaluation.MIN_EPISODE_S:
        parser.error(f"the evaluation counted episodes from {config['min_episode_s']} s, "
                     f"this code from {evaluation.MIN_EPISODE_S} s")
    for name, sha in config["files"].items():
        if benchlog.file_sha256(args.out.parent / name) != sha:
            parser.error(f"{name} is not the file the evaluation logged")
    workload = GridWorkload.from_config(config, args.out)
    schedule = workload.schedule(config["base_bytes"])
    started = json.loads(recorder.companion(args.out, ".replay.json").read_text())["started_ns"]
    events = read_pressure(recorder.companion(args.out, ".pressure.jsonl"))
    overall, scores = score_against_trace(workload, started,
                                          started + int(schedule.duration_s * 1e9), events,
                                          monitor.Thresholds(**config["thresholds"]))
    benchlog.append("monitor-evaluation", model=entry["model"],
                    context_length=entry["context_length"],
                    precision_tiers=entry["precision_tiers"],
                    config={**config, "issue": args.issue, "rescores": entry["sha256"]},
                    results={**entry["results"], "overall": evaluation.summary(overall), **scores},
                    log=args.log)
    print(json.dumps({"rescores": entry["sha256"][:8], "episodes": overall.episodes,
                      "false_negatives_detectable": overall.missed_detectable,
                      "per_cell_false_negatives_detectable": sum(
                          c["missed_detectable"] for c in scores["cells"])}))
    return 0

if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
