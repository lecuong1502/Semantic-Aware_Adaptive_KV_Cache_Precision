#!/usr/bin/env python3
"""The survival experiment (#108): Milestone 2's closing measurement.

    .venv/bin/python tools/survival_experiment.py static --out data/survival/static.csv.gz \\
        --issue 108
    .venv/bin/python tools/survival_experiment.py adaptive --out data/survival/adaptive.csv.gz \\
        --issue 108
        [--model qwen2.5-1.5b-instruct] [--prompt-positions 32704] [--new-tokens 64]
        [--contention-at 16384] [--shortfall 256] [--score-source semantic]
        [--pulse SECONDS] [--kv-tier FP16|INT4|INT2] [--scenario S1|S2|S3|S4|S5] [--run N]

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

With `--pulse SECONDS`, an adaptive run's simulator gives its memory back
that long after it took it, and the generation goes on (#155): pressure
passes, and after the cooldown the upgrades restore the pages from their
shadows. The run is one "pulse-release" entry, with what its recovery comes
to (survival.recovery): the time from the end of pressure to every page back
at FP16, the effective bits over time, and any downgrade and upgrade thrash;
its every step's effective bits are in a companion .timeline.jsonl. For
#155, a prompt of 30720 positions and 2048 new tokens leave minutes of
steps after the pulse for the upgrades. The upgrades come a batch a step,
so the restore takes as long as its steps: on the RTX 4050, 1.5B's restore
took about 540 s, its prefill chunks up to 40 s each near 30K positions,
and the run about 73 min, most of it 2048 tokens decoded at 1.8 s each
after every page was back at FP16; 64 new tokens would end it within a
minute of the restore.

The rule, not a number, sets what the simulator takes, so that the two runs
face the same contention whatever else the desktop holds. Each run is one
"survival" entry in the benchmark log, with the sha256 of the trace, the
simulator's schedule and events, and the engine's pressure events
(.pressure.jsonl).

With `--scenario S1` to `S4` (#190), the simulator plays that scenario of
the survival matrix in place of S5's step, the run's seed --run
(microinfer.survival_matrix): when the cache reaches --contention-at
positions, a base that leaves the scenario's headroom, and the scenario's
schedule on it, until the generation ends. `static` runs at --kv-tier,
B1 to B3; `adaptive` at FP16 from --score-source, B4 to B6. Every entry
names its condition, scenario and run, and the monitor's detection scored
against the trace, as RQ2 scored it; a static INT4 or INT2 engine's
headroom is counted as an FP16 engine's, so that every condition faces the
same bytes taken. tools/survival_matrix.py runs the matrix.

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
from microinfer import survival_matrix as matrix  # noqa: E402
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
    parser.add_argument("--shortfall", type=float, default=None,
                        help="MiB the engine is left short of its full FP16 cache, S5's; "
                             "256 by default")
    parser.add_argument("--score-source", choices=SOURCES, default="semantic",
                        help="where an adaptive engine's plans take their scores (#104)")
    parser.add_argument("--pulse", type=float, default=None, metavar="SECONDS",
                        help="give the memory back this long after it was taken (#155); "
                             "adaptive only")
    parser.add_argument("--kv-tier", choices=("FP16", "INT4", "INT2"), default="FP16",
                        help="a static engine's cache tier: B1, B2 or B3 (#190)")
    parser.add_argument("--scenario", choices=matrix.EVERY_SCENARIO, default="S5",
                        help="the contention: S5's step, or a scenario of the matrix (#190)")
    parser.add_argument("--run", type=int, default=0,
                        help="the run of its cell, the scenario's seed")
    parser.add_argument("--log", type=Path, default=benchlog.DEFAULT_LOG)
    args = parser.parse_args(argv)
    if args.out.exists():
        parser.error(f"{args.out} exists; a run does not overwrite a recording")
    adaptive = args.engine == "adaptive"
    if args.pulse is not None and not adaptive:
        parser.error("a pulse is released for the upgrades of an adaptive engine")
    if adaptive and args.kv_tier != "FP16":
        parser.error("an adaptive engine's cache starts at FP16")
    scheduled = args.scenario in matrix.SCHEDULED
    if scheduled and (args.shortfall is not None or args.pulse is not None):
        parser.error(f"{args.scenario} plays its own schedule: --shortfall and --pulse are S5's")
    shortfall = None if scheduled else 256.0 if args.shortfall is None else args.shortfall
    condition = matrix.condition_of(args.kv_tier, adaptive, args.score_source)
    if benchlog.environment(args.log)["git_dirty"]:
        parser.error("tracked files have uncommitted changes; commit first, so that the "
                     "entry names the code that produced it")
    args.out.parent.mkdir(parents=True, exist_ok=True)  # before minutes of loading

    # The simulator's context is taken as it starts, after the headroom the
    # take is set from has been read: measured once, before the engine loads.
    context_bytes = replay.simulator_context_bytes(args.out.parent)
    if scheduled:
        rq1 = matrix.Rq1.read(args.log)
        scenario = matrix.SCENARIOS[args.scenario]
        scenario_schedule = scenario.schedule(args.run, rq1)
        leave = scenario.leave_bytes(args.run, rq1)

        def on_base(headroom: int):
            base = matrix.base_bytes(headroom, leave_bytes=leave,
                                     simulator_context_bytes=context_bytes)
            return matrix.raised(scenario_schedule, base)

        contention = {"schedule": on_base}
    else:
        contention = {"shortfall_bytes": int(shortfall * MIB), "pulse_s": args.pulse}
    engine = Engine(REPO / "models" / args.model, kv_tier=args.kv_tier, kv_adaptive=adaptive,
                    kv_score_source=args.score_source)
    engine.load_weights()
    prompt = np.random.default_rng(SEED).integers(
        1000, engine.config.vocab_size - 1000, args.prompt_positions).astype(np.int32)
    result = survival.run(engine, prompt, args.new_tokens, contention_at=args.contention_at,
                          out=args.out, simulator_context_bytes=context_bytes, **contention)

    pressure = recorder.companion(args.out, ".pressure.jsonl")
    pressure.write_text("".join(
        json.dumps({**asdict(r.event), "positions_held": r.positions_held,
                    "drained_ns": r.drained_ns}) + "\n" for r in result.pressure_events))
    files = replay.files(args.out) + [pressure]
    if args.pulse is not None:
        timeline = recorder.companion(args.out, ".timeline.jsonl")
        timeline.write_text("".join(json.dumps(step._asdict()) + "\n" for step in result.timeline))
        files.append(timeline)
    context = args.prompt_positions + args.new_tokens - 1
    config = {"issue": args.issue, "engine": args.engine, "condition": condition,
              "kv_tier": args.kv_tier, "scenario": args.scenario, "run": args.run,
              "score_source": args.score_source if adaptive else None,
              "prompt_positions": args.prompt_positions, "new_tokens": args.new_tokens,
              "prompt_seed": SEED, "prefill_chunk": engine.prefill_chunk,
              "contention_at": args.contention_at, "shortfall_mib": shortfall,
              "pressure_chunk_seconds": engine.pressure_chunk_seconds if adaptive else None,
              "contention_rule": (
                  "when the cache reaches contention_at, the simulator takes a base that "
                  "leaves the scenario's headroom, as an FP16 engine would have it, and "
                  "plays the scenario's schedule on it until the generation ends"
                  if scheduled else
                  "the simulator takes what leaves the engine the shortfall short of its "
                  "full FP16 cache, from the headroom, as an FP16 engine would have it, when "
                  "the cache reaches contention_at, and keeps it to the end"
                  + ("" if args.pulse is None else ", or for pulse_s, and then gives it back")),
              "scenario_rule": matrix.describe(args.scenario, args.run, rq1) if scheduled
              else None,
              "pulse_s": args.pulse,
              "upgrade_cooldown_seconds": engine.kv_upgrade_cooldown_seconds,
              "simulator_context_bytes": context_bytes, "poll_s": monitor.POLL_S,
              "thresholds": asdict(monitor.DEFAULT),
              "files": {f.name: benchlog.file_sha256(f) for f in files if f.exists()}}
    results = {**result.summary, "contention_position": result.contention_position,
               "headroom_at_contention_mib": result.headroom_bytes / MIB,
               "taken_mib": result.taken_bytes / MIB,
               "pressure_events": len(result.pressure_events),
               "pressure_levels": [r.event.level.value for r in result.pressure_events],
               "pressure": survival.waits(result.pressure_events, engine.plans),
               "detection": survival.detection(args.out,
                                               [r.event for r in result.pressure_events],
                                               monitor.DEFAULT, result.started_ns)}
    if scheduled:
        played_s = (result.timeline[-1].t - result.took_at_seconds if result.timeline
                    else 0.0)
        # What the base left, as an FP16 engine would have it: less than the
        # scenario asks only for headroom already below it (base_bytes).
        headroom = survival.as_fp16(engine.config, result.headroom_bytes,
                                    positions=result.contention_position, kv_tier=args.kv_tier)
        results["schedule"] = {"base_mib": result.taken_bytes / MIB,
                               "leave_mib": leave / MIB,
                               "left_mib": (headroom - context_bytes - result.taken_bytes) / MIB,
                               "peak_mib": result.schedule.peak_bytes / MIB,
                               "points": len(result.schedule.points),
                               "duration_s": result.schedule.duration_s, "played_s": played_s,
                               "cut": played_s < result.schedule.duration_s}
    if args.pulse is not None:
        results["recovery"] = survival.recovery(
            result, engine.plans, cooldown_seconds=engine.kv_upgrade_cooldown_seconds)
    # An adaptive cache ends mixed, page by page; its plans say how.
    benchlog.append("survival" if args.pulse is None else "pulse-release",
                    model=args.model, context_length=context,
                    precision_tiers=None if adaptive else {args.kv_tier: 1.0}, config=config,
                    results=results, log=args.log)
    print(json.dumps({"condition": condition, "scenario": args.scenario, "run": args.run,
                      "survived": result.summary["survived"],
                      "ending": result.summary["ending"], "taken_mib": result.taken_bytes / MIB,
                      **result.summary["totals"], **({} if args.pulse is None else {
                          "recovery": results["recovery"]})}))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
