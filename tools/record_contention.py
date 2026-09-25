#!/usr/bin/env python3
"""Record device-wide free and used memory, and who holds it, at fixed rates (#46, #47).

    .venv/bin/python tools/record_contention.py --out trace.csv.gz [--rate 50]
        [--processes-rate 5] [--duration S]
    .venv/bin/python tools/record_contention.py --calibrate --issue 46 [--duration 60]

The recorder of RQ1 (microinfer.recorder), as a process of its own: it holds
no device memory and shares nothing with the engine, so it can record the
engine as one more process among the desktop's. It stops cleanly on SIGINT or
SIGTERM, closing its files; killed outright, it leaves files that read back
to within a second. Beside --out it writes the processes stream, trace.csv.gz
becoming trace.procs.csv.gz: each GPU process's memory, and the GPU's P-state
and clocks.

--calibrate records for --duration seconds without a file and logs what the
recorder achieved on this machine: the cost of one NVML query, the rate, the
jitter of the periods, and whether NVML saw the recorder as a GPU process,
which it should not. Refuses a dirty tree when it logs.
"""

from __future__ import annotations

import argparse
import os
import signal
import sys
import threading
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from microinfer import benchlog, nvml, recorder  # noqa: E402


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, help="the compressed CSV to write")
    parser.add_argument("--rate", type=float, default=50.0, help="samples per second")
    parser.add_argument("--processes-rate", type=float, default=5.0,
                        help="process samples per second; 0 turns the stream off")
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
        if args.out is not None:
            parser.error("--calibrate records no file; drop --out")
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
                            duration=args.duration, stop=stop,
                            processes_rate_hz=args.processes_rate)
    # The claim that the recorder is not a GPU process, checked before it exits.
    gpu_process = os.getpid() in {p.pid for p in nvml.processes()}
    print(f"{stats['samples']} samples at {stats.get('achieved_hz', 0):.2f} Hz, "
          f"{stats['missed']} deadlines missed; period p99 "
          f"{stats.get('period_ms', {}).get('p99', 0):.2f} ms; NVML query median "
          f"{stats.get('query_us', {}).get('median', 0):.0f} us")
    if args.calibrate:
        benchlog.append(
            "recorder-calibration", model=None, context_length=None, precision_tiers=None,
            # The device stream's figures are measured with the processes
            # stream's thread running beside it, when its rate is not 0.
            config={"rate_hz": args.rate, "duration_seconds": args.duration,
                    "query": "nvmlDeviceGetMemoryInfo",
                    "processes_rate_hz": args.processes_rate,
                    "processes_queries": ["nvmlDeviceGetComputeRunningProcesses_v3",
                                          "nvmlDeviceGetGraphicsRunningProcesses_v3",
                                          "nvmlDeviceGetPerformanceState",
                                          "nvmlDeviceGetClockInfo"],
                    "issue": args.issue},
            results={**stats, "recorder_is_gpu_process": gpu_process}, log=args.log)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
