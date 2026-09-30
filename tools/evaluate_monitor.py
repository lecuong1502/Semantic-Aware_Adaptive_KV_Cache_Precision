#!/usr/bin/env python3
"""Evaluate the pressure monitor on a synthetic grid of pulses (#64).

    .venv/bin/python tools/evaluate_monitor.py --out data/monitor/grid.csv.gz --issue 64
        [--amplitudes 768 1152 1408] [--ramps 0 0.25 1] [--plateaus 0.1 0.2 0.5 2]
        [--repeats 3] [--gap 3] [--start-headroom 1536]

The engine holds a generation (Engine.hold) with its pressure monitor on,
recording every event between steps. Once its cache has stopped growing, the
contention simulator, a process of its own, takes a base, so that headroom
sits at --start-headroom MiB, in GREEN, and then plays every cell of the grid
of amplitude (MiB above the base) x ramp, up and down (s) x plateau (s),
--repeats times, --gap seconds apart and a seeded fraction of a poll more, so
that the pulses meet the polls at every phase. The recorder records the device
throughout (microinfer.replay.replay).

The truth is the recorder's trace with the monitor's thresholds applied, and
the monitor is scored against it (microinfer.evaluation): detection latency,
false negatives and false positives, overall and per cell. False negatives
shorter than K + 1 polls and a recorder sample are counted apart: ADR-0013's
K cannot be sure to catch them. The result is
logged as a "monitor-evaluation" entry with the sha256 of every file, the
monitor's events among them (.pressure.jsonl), so that it can be scored
again without running again.

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
from dataclasses import asdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from microinfer import Engine, benchlog, evaluation, monitor, nvml, recorder, replay  # noqa: E402
from microinfer.contention import spread  # noqa: E402
from microinfer.engine import DECODING  # noqa: E402
from microinfer.footprint import MIB  # noqa: E402

SEED = 64
#: How long headroom settles after the engine's cache stops growing.
SETTLE_S = 2.0


def summary(s: evaluation.Score) -> dict:
    return {**asdict(s), "false_negatives": s.false_negatives,
            "latency_ms": spread(s.latencies_ms), "latency_polls": spread(s.latencies_polls)}


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, required=True, help="the recorder's trace")
    parser.add_argument("--issue", type=int, required=True)
    parser.add_argument("--model", default="qwen2.5-1.5b-instruct")
    parser.add_argument("--prompt-positions", type=int, default=1024)
    parser.add_argument("--decode-positions", type=int, default=64,
                        help="the span the hold decodes again and again")
    parser.add_argument("--amplitudes", type=float, nargs="+", default=[768, 1152, 1408],
                        help="MiB above the base")
    parser.add_argument("--ramps", type=float, nargs="+", default=[0.0, 0.25, 1.0],
                        help="seconds each pulse ramps up, and down")
    parser.add_argument("--plateaus", type=float, nargs="+", default=[0.1, 0.2, 0.5, 2.0],
                        help="seconds each pulse is kept at its amplitude: the grid's "
                             "duration")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--gap", type=float, default=3.0, help="seconds between pulses")
    parser.add_argument("--lead", type=float, default=5.0,
                        help="seconds on the base before the first pulse")
    parser.add_argument("--start-headroom", type=float, default=1536,
                        help="MiB of headroom the base leaves")
    parser.add_argument("--log", type=Path, default=benchlog.DEFAULT_LOG)
    args = parser.parse_args(argv)
    if args.out.exists():
        parser.error(f"{args.out} exists; an evaluation does not overwrite a recording")
    args.out.parent.mkdir(parents=True, exist_ok=True)  # before minutes of loading
    if benchlog.environment(args.log)["git_dirty"]:
        parser.error("tracked files have uncommitted changes; commit first, so that the "
                     "entry names the code that produced it")

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
    watching = engine.start_monitor()
    holding.start()
    try:
        while not full.wait(0.5):
            if stop.is_set() or not holding.is_alive():
                raise SystemExit("stopped before the hold's cache was full")
        time.sleep(SETTLE_S)
        headroom = nvml.memory().free
        base = int(headroom - args.start_headroom * MIB)
        if base < 0:
            raise SystemExit(f"headroom is {headroom / MIB:.0f} MiB, below the "
                             f"{args.start_headroom:g} MiB the grid starts from")
        cells = [evaluation.Cell(int(a * MIB), r, p)
                 for a in args.amplitudes for r in args.ramps for p in args.plateaus]
        grid = evaluation.grid_schedule(cells, base, args.repeats, args.gap, args.lead,
                                        seed=SEED)
        print(f"{len(cells)} cells x {args.repeats} over {grid.schedule.duration_s:.0f} s, "
              f"on a base of {base / MIB:.0f} MiB", flush=True)
        started = replay.replay(grid.schedule, args.out, stop=stop)
    finally:
        halt.set()
        holding.join()
        engine.stop_monitor()

    thresholds = watching.thresholds
    events = [r.event for r in engine.pressure_events]
    _, samples = recorder.read(args.out)
    episodes = evaluation.true_red(samples["t_mono_ns"], samples["free_bytes"], thresholds)
    changes = [json.loads(line) for line
               in recorder.companion(args.out, ".events.jsonl").read_text().splitlines()]
    starts = evaluation.applied_starts(grid, started, changes)
    end = started + int(grid.schedule.duration_s * 1e9)
    overall = evaluation.score(episodes, events, thresholds, within=(started, end))
    per_cell = evaluation.score_cells(grid, starts, end, episodes, events, thresholds)
    results = {"overall": summary(overall),
               "cells": [{**asdict(c), "amplitude_mib": c.amplitude_bytes / MIB, **summary(s)}
                         for c, s in zip(cells, per_cell)],
               "events_recorded": len(events),
               "monitor_error": None if engine.monitor_error is None
               else str(engine.monitor_error),
               "missed_polls": watching.missed}
    # The monitor's events, as the engine recorded them, beside the trace:
    # the truth can be scored again without running again.
    pressure = recorder.companion(args.out, ".pressure.jsonl")
    pressure.write_text("".join(
        json.dumps({**asdict(r.event), "positions_held": r.positions_held,
                    "drained_ns": r.drained_ns}) + "\n" for r in engine.pressure_events))
    files = replay.files(args.out) + [pressure]
    benchlog.append(
        "monitor-evaluation", model=args.model, context_length=context,
        precision_tiers={"FP16": 1.0},
        config={"issue": args.issue, "workload": "synthetic grid", "repeats": args.repeats,
                "gap_s": args.gap, "lead_s": args.lead, "phase_seed": SEED,
                "start_headroom_mib": args.start_headroom, "base_bytes": base,
                "amplitudes_mib": args.amplitudes, "ramps_s": args.ramps,
                "plateaus_s": args.plateaus,
                "duration": "a pulse's plateau; it lasts ramp + plateau + ramp",
                "poll_s": monitor.POLL_S,
                "thresholds": asdict(thresholds),
                "min_episode_s": evaluation.MIN_EPISODE_S,
                "files": {f.name: benchlog.file_sha256(f) for f in files if f.exists()}},
        results=results, log=args.log)
    latency = results["overall"]["latency_ms"]
    print(json.dumps({"episodes": overall.episodes, "latency_ms": latency,
                      "missed_detectable": overall.missed_detectable,
                      "missed_below_k": overall.missed_below_k,
                      "already_red": overall.already_red,
                      "false_positives": overall.false_positives}))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
