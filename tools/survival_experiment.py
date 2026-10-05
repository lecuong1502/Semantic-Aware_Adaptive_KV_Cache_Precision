#!/usr/bin/env python3
"""The survival experiment (#108): Milestone 2's closing measurement.

    .venv/bin/python tools/survival_experiment.py static --out data/survival/static.csv.gz \\
        --issue 108
    .venv/bin/python tools/survival_experiment.py adaptive --out data/survival/adaptive.csv.gz \\
        --issue 108
        [--model qwen2.5-1.5b-instruct] [--prompt-positions 32704] [--new-tokens 64]
        [--contention-at 16384] [--shortfall 256] [--score-source semantic]

Qwen2.5-1.5B prefills a prompt and decodes to 32K positions, its cache at
FP16, with its pressure monitor on (microinfer.survival.run). When the cache
reaches --contention-at positions, the contention simulator, a process of
its own, takes what leaves the engine --shortfall MiB short of its full
cache, and keeps it to the end, as most of RQ1's spikes kept theirs; the
generation waits at that position until the simulator has taken it. The
recorder records the device throughout.

- `static` holds every page at FP16 and does not adapt: it runs out of
  memory before its cache is full, and the OutOfMemory is logged with the
  position it reached.
- `adaptive` downgrades on the monitor's YELLOW and RED, and on an
  allocation that fails (#105, #107), from --score-source's scores: it is
  to finish, every token decoded. Every plan is logged with the memory its
  cache returned, what the driver saw the engine's process give back, and
  whether the two agree to a granule, and how long it took. Every pressure
  event is logged with how long it waited for the step boundary at which
  the engine drained it: under pressure, prefill chunks are sized to keep
  that short (#135).

The rule, not a number, sets what the simulator takes, so that the two runs
face the same contention whatever else the desktop holds. Each run is one
"survival" entry in the benchmark log, with the sha256 of the trace, the
simulator's schedule and events, and the engine's pressure events
(.pressure.jsonl).

Close every application first: what the desktop takes or gives back during
a run moves the shortfall. A 32K run takes tens of minutes. Refuses a dirty
tree, and an --out that exists.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from microinfer import Engine, benchlog, monitor, recorder, replay, survival  # noqa: E402
from microinfer.footprint import MIB  # noqa: E402
from microinfer.score_sources import SOURCES  # noqa: E402

SEED = 108


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("engine", choices=("static", "adaptive"))
    parser.add_argument("--out", type=Path, required=True, help="the recorder's trace")
    parser.add_argument("--issue", type=int, required=True)
    parser.add_argument("--model", default="qwen2.5-1.5b-instruct")
    parser.add_argument("--prompt-positions", type=int, default=32704)
    parser.add_argument("--new-tokens", type=int, default=64)
    parser.add_argument("--contention-at", type=int, default=16384,
                        help="the cache's positions when the simulator takes memory")
    parser.add_argument("--shortfall", type=float, default=256,
                        help="MiB the engine is left short of its full FP16 cache")
    parser.add_argument("--score-source", choices=SOURCES, default="semantic",
                        help="where an adaptive engine's plans take their scores (#104)")
    parser.add_argument("--log", type=Path, default=benchlog.DEFAULT_LOG)
    args = parser.parse_args(argv)
    if args.out.exists():
        parser.error(f"{args.out} exists; a run does not overwrite a recording")
    if benchlog.environment(args.log)["git_dirty"]:
        parser.error("tracked files have uncommitted changes; commit first, so that the "
                     "entry names the code that produced it")
    args.out.parent.mkdir(parents=True, exist_ok=True)  # before minutes of loading

    # The simulator's context is taken as it starts, after the headroom the
    # take is set from has been read: measured once, before the engine loads.
    context_bytes = replay.simulator_context_bytes(args.out.parent)
    adaptive = args.engine == "adaptive"
    engine = Engine(REPO / "models" / args.model, kv_tier="FP16", kv_adaptive=adaptive,
                    kv_score_source=args.score_source)
    engine.load_weights()
    prompt = np.random.default_rng(SEED).integers(
        1000, engine.config.vocab_size - 1000, args.prompt_positions).astype(np.int32)
    result = survival.run(engine, prompt, args.new_tokens, contention_at=args.contention_at,
                          shortfall_bytes=int(args.shortfall * MIB), out=args.out,
                          simulator_context_bytes=context_bytes)

    pressure = recorder.companion(args.out, ".pressure.jsonl")
    pressure.write_text("".join(
        json.dumps({**asdict(r.event), "positions_held": r.positions_held,
                    "drained_ns": r.drained_ns}) + "\n" for r in result.pressure_events))
    files = replay.files(args.out) + [pressure]
    context = args.prompt_positions + args.new_tokens - 1
    config = {"issue": args.issue, "engine": args.engine,
              "score_source": args.score_source if adaptive else None,
              "prompt_positions": args.prompt_positions, "new_tokens": args.new_tokens,
              "prompt_seed": SEED, "prefill_chunk": engine.prefill_chunk,
              "contention_at": args.contention_at, "shortfall_mib": args.shortfall,
              "pressure_chunk_seconds": engine.pressure_chunk_seconds if adaptive else None,
              "contention_rule": "the simulator takes what leaves the engine the shortfall "
                                 "short of its full FP16 cache, from the headroom when the "
                                 "cache reaches contention_at, and keeps it to the end",
              "simulator_context_bytes": context_bytes, "poll_s": monitor.POLL_S,
              "thresholds": asdict(monitor.DEFAULT),
              "files": {f.name: benchlog.file_sha256(f) for f in files if f.exists()}}
    results = {**result.summary, "contention_position": result.contention_position,
               "headroom_at_contention_mib": result.headroom_bytes / MIB,
               "taken_mib": result.taken_bytes / MIB,
               "pressure": survival.waits(result.pressure_events)}
    # An adaptive cache ends mixed, page by page; its plans say how.
    benchlog.append("survival", model=args.model, context_length=context,
                    precision_tiers=None if adaptive else {"FP16": 1.0}, config=config,
                    results=results, log=args.log)
    print(json.dumps({"engine": args.engine, "survived": result.summary["survived"],
                      "ending": result.summary["ending"], "taken_mib": result.taken_bytes / MIB,
                      **result.summary["totals"]}))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
