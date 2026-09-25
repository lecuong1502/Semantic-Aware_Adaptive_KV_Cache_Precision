#!/usr/bin/env python3
"""Record device-wide free and used memory at a fixed rate (#46).

    .venv/bin/python tools/record_contention.py --out trace.csv.gz [--rate 50] [--duration S]
    .venv/bin/python tools/record_contention.py --calibrate --issue 46 [--duration 60]

The recorder of RQ1 (microinfer.recorder), as a process of its own: it holds
no device memory and shares nothing with the engine, so it can record the
engine as one more process among the desktop's. It stops cleanly on SIGINT or
SIGTERM, closing its file; killed outright, it leaves a file that reads back
to within a second.

--calibrate records for --duration seconds without a file and logs what the
recorder achieved on this machine: the cost of one NVML query, the rate, and
the jitter of the periods. Refuses a dirty tree when it logs.
"""

from __future__ import annotations

import argparse
import signal
import sys
import threading
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from microinfer import benchlog, recorder  # noqa: E402


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, help="the compressed CSV to write")
    parser.add_argument("--rate", type=float, default=50.0, help="samples per second")
    parser.add_argument("--duration", type=float, default=None,
                        help="seconds to record; until interrupted if omitted")
    parser.add_argument("--calibrate", action="store_true",
                        help="record without a file and log what was achieved")
    parser.add_argument("--issue", type=int, help="the ticket a calibration is logged for")
    parser.add_argument("--log", type=Path, default=benchlog.DEFAULT_LOG)
    args = parser.parse_args(argv)

    if args.calibrate:
        if args.issue is None:
            parser.error("--calibrate logs its result, and needs --issue")
        if benchlog.environment(args.log)["git_dirty"]:
            parser.error("tracked files have uncommitted changes; commit first, so that "
                         "the entry names the code that produced it")
        if args.duration is None:
            args.duration = 60.0
    elif args.out is None:
        parser.error("--out is required unless calibrating")

    stop = threading.Event()
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: stop.set())

    stats = recorder.record(None if args.calibrate else args.out, args.rate,
                            duration=args.duration, stop=stop)
    print(f"{stats['samples']} samples at {stats.get('achieved_hz', 0):.2f} Hz, "
          f"{stats['missed']} deadlines missed; period p99 "
          f"{stats.get('period_ms', {}).get('p99', 0):.2f} ms; NVML query median "
          f"{stats.get('query_us', {}).get('median', 0):.0f} us")
    if args.calibrate:
        benchlog.append(
            "recorder-calibration", model=None, context_length=None, precision_tiers=None,
            config={"rate_hz": args.rate, "seconds": args.duration,
                    "query": "nvmlDeviceGetMemoryInfo", "issue": args.issue},
            results=stats, log=args.log)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
