#!/usr/bin/env python3
"""Summarise the pressure monitor's evaluations for RQ2 (#65).

    .venv/bin/python tools/summarise_monitor.py --issue 65

Reads every "monitor-evaluation" entry of the benchmark log, takes the
latest of each grid and each replayed recording that no correction
supersedes, and sums them per workload, synthetic and replayed
(microinfer.evaluation.summarise): true RED episodes, detectable ones,
misses, false positives, and latency pooled over them. Logs the summary as
a "monitor-summary" entry naming the entries it took, and prints it as a
table for the ADR note. Refuses a dirty tree.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from microinfer import benchlog, evaluation  # noqa: E402


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--issue", type=int, required=True)
    parser.add_argument("--log", type=Path, default=benchlog.DEFAULT_LOG)
    args = parser.parse_args(argv)
    if benchlog.environment(args.log)["git_dirty"]:
        parser.error("tracked files have uncommitted changes; commit first, so that the "
                     "entry names the code that produced it")

    summary = evaluation.summarise(benchlog.read(args.log))
    if not summary:
        parser.error("the log has no monitor-evaluation entry to summarise")
    benchlog.append("monitor-summary", model=None, context_length=None, precision_tiers=None,
                    config={"issue": args.issue}, results=summary, log=args.log)

    print("| Workload | Entries | Episodes | Detectable | Missed | Missed under K "
          "| Already RED | False positives | Latency median / P90 / max (ms) |")
    print("|---|---|---|---|---|---|---|---|---|")
    for workload, s in summary.items():
        latency = s["latency_ms"]
        spread = ("-" if latency is None else
                  f"{latency['median']:.0f} / {latency['p90']:.0f} / {latency['max']:.0f}")
        print(f"| {workload} | {len(s['entries'])} | {s['episodes']} | {s['detectable']} "
              f"| {s['missed_detectable']} | {s['missed_below_k']} | {s['already_red']} "
              f"| {s['false_positives']} | {spread} |")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
