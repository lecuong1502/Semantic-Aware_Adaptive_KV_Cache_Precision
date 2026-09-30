#!/usr/bin/env python3
"""Evaluate the pressure monitor on a synthetic grid of spikes (#64).

    .venv/bin/python tools/evaluate_monitor.py --out data/monitor/grid.csv.gz --issue 64
        [--amplitudes 768 1152 1408] [--rises 0 0.25 1] [--plateaus 0.1 0.2 0.5 2]
        [--repeats 3] [--gap 3] [--start-headroom 1536]

The engine holds a generation (Engine.hold) with its pressure monitor on,
recording every event between steps. Once its cache has stopped growing, the
contention simulator, a process of its own, takes a base, so that headroom
sits at --start-headroom MiB, in GREEN, and then plays every cell of the grid
of amplitude (MiB above the base) x rise time (and fall, s) x plateau (s),
--repeats times, --gap seconds apart. The recorder records the device
throughout (microinfer.replay.replay).

The truth is the recorder's trace with the monitor's thresholds applied, and
the monitor is scored against it (microinfer.evaluation): detection latency,
false negatives and false positives, overall and per cell. The result is
logged as a "monitor-evaluation" entry with the sha256 of every file.

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
    return {**asdict(s), "latency_ms": spread(s.latencies_ms)}


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
    parser.add_argument("--rises", type=float, nargs="+", default=[0.0, 0.25, 1.0])
    parser.add_argument("--plateaus", type=float, nargs="+", default=[0.1, 0.2, 0.5, 2.0])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--gap", type=float, default=3.0, help="seconds between spikes")
    parser.add_argument("--lead", type=float, default=5.0,
                        help="seconds on the base before the first spike")
    parser.add_argument("--start-headroom", type=float, default=1536,
                        help="MiB of headroom the base leaves")
    parser.add_argument("--log", type=Path, default=benchlog.DEFAULT_LOG)
    args = parser.parse_args(argv)
    if args.out.exists():
        parser.error(f"{args.out} exists; an evaluation does not overwrite a recording")
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
                 for a in args.amplitudes for r in args.rises for p in args.plateaus]
        grid = evaluation.grid_schedule(cells, base, args.repeats, args.gap, args.lead)
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
    during = [e for e in episodes if e[0] >= started]
    overall = evaluation.score(during, [e for e in events if e.t_mono_ns >= started],
                               thresholds)
    per_cell = evaluation.score_cells(grid, started, episodes, events, thresholds)
    results = {"overall": summary(overall),
               "cells": [{**asdict(c), "amplitude_mib": c.amplitude_bytes / MIB, **summary(s)}
                         for c, s in zip(cells, per_cell)],
               "events_recorded": len(events),
               "monitor_error": None if engine.monitor_error is None
               else str(engine.monitor_error),
               "missed_polls": watching.missed}
    files = [args.out, recorder.processes_path(args.out),
             recorder.companion(args.out, ".schedule.json"),
             recorder.companion(args.out, ".events.jsonl"),
             recorder.companion(args.out, ".replay.json")]
    benchlog.append(
        "monitor-evaluation", model=args.model, context_length=context,
        precision_tiers={"FP16": 1.0},
        config={"issue": args.issue, "workload": "synthetic grid", "repeats": args.repeats,
                "gap_s": args.gap, "lead_s": args.lead,
                "start_headroom_mib": args.start_headroom, "base_bytes": base,
                "amplitudes_mib": args.amplitudes, "rises_s": args.rises,
                "plateaus_s": args.plateaus, "poll_s": monitor.POLL_S,
                "thresholds": asdict(thresholds),
                "min_episode_s": evaluation.MIN_EPISODE_S,
                "files": {f.name: benchlog.file_sha256(f) for f in files if f.exists()}},
        results=results, log=args.log)
    latency = results["overall"]["latency_ms"]
    print(json.dumps({"episodes": overall.episodes, "latency_ms": latency,
                      "false_negatives": overall.false_negatives,
                      "false_positives": overall.false_positives}))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
