#!/usr/bin/env python3
"""Summarise the pressure monitor's evaluations for RQ2 (#65).

    .venv/bin/python tools/summarise_monitor.py --issue 65
    .venv/bin/python tools/summarise_monitor.py --issue 159 --hysteresis

Reads every "monitor-evaluation" entry of the benchmark log, takes the
latest grid and the latest replay of each recording replayed whole that no
correction supersedes, and sums them per workload, synthetic and replayed
(microinfer.evaluation.summarise): true RED episodes, detectable ones, false
negatives, false positives, and latency pooled over them. Logs the summary as
a "monitor-summary" entry naming the entries it took, and prints it as a
table for the ADR note. Only evaluations at ADR-0013's thresholds and K
enter it.

With --hysteresis, the synthetic grid at each K instead (#159,
microinfer.evaluation.grid_by_persist_polls): the latest grid at each, all at
one code and of one shape, logged as a "monitor-hysteresis-summary" entry
whose results are keyed by K as a string, as JSON keys are.

Refuses a dirty tree.
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
    parser.add_argument("--hysteresis", action="store_true",
                        help="the synthetic grid at each K the monitor was run at (#159)")
    args = parser.parse_args(argv)
    if benchlog.environment(args.log)["git_dirty"]:
        parser.error("tracked files have uncommitted changes; commit first, so that the "
                     "entry names the code that produced it")
    if args.hysteresis:
        return hysteresis(parser, args)

    summary = evaluation.summarise(benchlog.read(args.log))
    if not summary:
        parser.error("the log has no monitor-evaluation entry to summarise")
    benchlog.append("monitor-summary", model=None, context_length=None, precision_tiers=None,
                    config={"issue": args.issue}, results=summary, log=args.log)

    print("| Workload | Entries | Episodes | Detectable | False negatives "
          "| False negatives under K "
          "| Already RED | False positives | Latency median / P90 / max (ms) |")
    print("|---|---|---|---|---|---|---|---|---|")
    for workload, s in summary.items():
        print(f"| {workload} | {len(s['entries'])} | {s['episodes']} | {s['detectable']} "
              f"| {s['missed_detectable']} | {s['missed_below_k']} | {s['already_red']} "
              f"| {s['false_positives']} | {latency_cell(s['latency_ms'])} |")
    return 0


def latency_cell(spread: dict | None) -> str:
    """A spread of latencies as the table's cell: median / P90 / max."""
    return "-" if spread is None else (
        f"{spread['median']:.0f} / {spread['p90']:.0f} / {spread['max']:.0f}")


def hysteresis(parser: argparse.ArgumentParser, args: argparse.Namespace) -> int:
    """The grid at each K (#159): logged, and printed as a table for
    ADR-0014's amendment."""
    try:
        by_k = evaluation.grid_by_persist_polls(benchlog.read(args.log))
    except ValueError as exc:
        parser.error(str(exc))
    if not by_k:
        parser.error("the log has no evaluation of the synthetic grid to summarise")
    benchlog.append("monitor-hysteresis-summary", model=None, context_length=None,
                    precision_tiers=None, config={"issue": args.issue},
                    results={str(k): s for k, s in by_k.items()}, log=args.log)

    print("| K | Entry | Commit | Episodes | Detectable | False negatives "
          "| False negatives under K | Already RED | False positives | RED events "
          "| Latency median / P90 / max (ms) |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    for k, s in by_k.items():
        print(f"| {k} | `{s['entry'][:8]}…` | {s['commit'][:7]} | {s['episodes']} "
              f"| {s['detectable']} | {s['missed_detectable']} | {s['missed_below_k']} "
              f"| {s['already_red']} | {s['false_positives']} | {s['red_events']} "
              f"| {latency_cell(s['latency_ms'])} |")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
