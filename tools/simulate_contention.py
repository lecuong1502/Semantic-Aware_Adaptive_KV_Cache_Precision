#!/usr/bin/env python3
"""Take device memory on a schedule, as a process of its own (#58).

    .venv/bin/python tools/simulate_contention.py --schedule schedule.json
        [--events events.jsonl]
    .venv/bin/python tools/simulate_contention.py --calibrate --issue 58 [--repeats 20]

The contention simulator (microinfer.simulator): its own CUDA context, memory
only, taken a granule at a time through the engine's VMM allocator,
following a schedule of (time, bytes taken) from a JSON file
({"points": [[t, bytes], ...]}). The last level is kept until the simulator
is stopped; a schedule that means to give everything back ends at 0.

It prints "ready" once its CUDA context exists, before the schedule starts,
for a caller that must know when its memory is another process's. Each change
is appended to --events as a JSON line as it happens: when it was due, how
late it began, the bytes asked and taken and any shortfall, and its latency
until the calls returned and until NVML showed it. It stops on SIGINT or
SIGTERM, giving back everything it took; killed, the driver takes it all back
with the process.

--calibrate takes and gives back steps from one granule to 1 GiB --repeats
times, and logs the latency of each size and direction to the benchmark log.
Refuses a dirty tree, so that the entry names its code.
"""

from __future__ import annotations

import argparse
import json
import signal
import sys
import threading
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from microinfer import benchlog, simulator  # noqa: E402


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--schedule", type=Path, help="the schedule to follow")
    parser.add_argument("--events", type=Path, help="a JSON-lines file of every change")
    parser.add_argument("--calibrate", action="store_true",
                        help="measure the latency of holding and releasing, and log it")
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--issue", type=int, help="the ticket a calibration is logged for")
    parser.add_argument("--log", type=Path, default=benchlog.DEFAULT_LOG)
    args = parser.parse_args(argv)
    if args.calibrate:
        if args.issue is None:
            parser.error("--calibrate logs its result, and needs --issue")
        if benchlog.environment(args.log)["git_dirty"]:
            parser.error("tracked files have uncommitted changes; commit first, so that "
                         "the entry names the code that produced it")
        schedule = None
    elif args.schedule is None:
        parser.error("--schedule is required unless calibrating")
    else:
        schedule = simulator.Schedule.load(args.schedule)

    stop = threading.Event()
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: stop.set())

    if args.calibrate:
        results = simulator.calibrate(args.repeats, stop=stop)
        benchlog.append("simulator-latency", model=None, context_length=None,
                        precision_tiers=None,
                        config={"issue": args.issue, "repeats": args.repeats,
                                "sizes_bytes": list(simulator.CALIBRATION_BYTES),
                                "step_s": simulator.CALIBRATION_STEP_S,
                                "granule_bytes": simulator.GRANULE,
                                "nvml_poll_s": simulator.NVML_POLL_S},
                        results=results, log=args.log)
        for size, directions in results.items():
            print(f"{int(size) / 2**20:6.0f} MiB: " + "; ".join(
                f"{d} NVML {r['nvml_ms']['median']:.2f} ms median, {r['nvml_ms']['max']:.2f} "
                f"max, {r['unseen']} unseen" for d, r in directions.items()))
        return 0

    out = open(args.events, "a") if args.events else None

    def record(event: dict) -> None:
        if out is not None:
            out.write(json.dumps(event) + "\n")
            out.flush()

    reservation = simulator.Reservation()
    print("ready", flush=True)
    try:
        simulator.run(schedule, reservation, stop, on_change=record)
    finally:
        reservation.release()
        if out is not None:
            out.close()
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
