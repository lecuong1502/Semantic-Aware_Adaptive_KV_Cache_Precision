#!/usr/bin/env python3
"""Turn contention recordings into RQ1's results, and log them (#50).

    .venv/bin/python tools/analyse_contention.py --issue N trace.csv.gz [more.csv.gz ...]

For each recording (microinfer.contention): its spikes, each attributed to
the process most responsible; the distributions per labelled action, or the
rates per hour of a passive session; and the spike counts at 32, 64 and
128 MiB. Each recording is one benchmark-log entry carrying the sha256 of
its files, the processes and labels files beside it included. Refuses a
dirty tree, so that each entry names the code that produced it.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from microinfer import benchlog, contention  # noqa: E402


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("traces", type=Path, nargs="+", help="device traces (*.csv.gz)")
    parser.add_argument("--issue", type=int, required=True,
                        help="the ticket the results are logged for")
    parser.add_argument("--log", type=Path, default=benchlog.DEFAULT_LOG)
    args = parser.parse_args(argv)
    if benchlog.environment(args.log)["git_dirty"]:
        parser.error("tracked files have uncommitted changes; commit first, so that "
                     "each entry names the code that produced it")

    for trace in args.traces:
        results = contention.log_recording(trace, issue=args.issue, log=args.log)["results"]
        counts = ", ".join(f"{m} MiB: {c['count']}"
                           for m, c in results["sensitivity_mib"].items())
        print(f"{trace.name}: {len(results['spikes'])} spikes over "
              f"{results['recorded_hours'] * 60:.1f} min recorded ({counts}); attributed "
              f"{results['all']['attributed']}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
