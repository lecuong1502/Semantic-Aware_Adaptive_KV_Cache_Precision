#!/usr/bin/env python3
"""Summarise the survival matrix for RQ2 (#191).

    .venv/bin/python tools/summarise_survival.py --issue 191

Reads every "survival" and "pulse-release" entry of the benchmark log and
pools them by cell, condition and scenario (microinfer.survival_summary):
S1 to S4 the matrix's three runs a cell, S5 and S6 the survival
experiment's one. Logs the summary as a "survival-summary" entry naming
every run it pooled, and prints the survival table, each rate with its Wilson
95% interval, and the reaction table, B6's stages by scenario, for the
thesis.

Refuses a dirty tree.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from microinfer import benchlog, survival_matrix as matrix, survival_summary  # noqa: E402


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--issue", type=int, required=True)
    parser.add_argument("--model", default=survival_summary.MODEL)
    parser.add_argument("--log", type=Path, default=benchlog.DEFAULT_LOG)
    args = parser.parse_args(argv)
    if benchlog.environment(args.log)["git_dirty"]:
        parser.error("tracked files have uncommitted changes; commit first, so that the "
                     "entry names the code that produced it")
    benchlog.verify(args.log)  # the hashes the summary names are of the entries as appended
    summary = survival_summary.summarise(benchlog.read(args.log), args.model)
    if not summary["entries"]:
        parser.error("the log has no survival run to summarise")
    benchlog.append(survival_summary.SUMMARY_KIND, model=args.model, context_length=None,
                    precision_tiers=None, config={"issue": args.issue}, results=summary,
                    log=args.log)

    print("| Condition | " + " | ".join(survival_summary.TABLE_SCENARIOS) + " |")
    print("|---|" + "---|" * len(survival_summary.TABLE_SCENARIOS))
    for condition in matrix.CONDITIONS:
        cells = [summary["survival"].get(s, {}).get(condition)
                 for s in survival_summary.TABLE_SCENARIOS]
        print(f"| {condition} | " + " | ".join(rate_cell(c) for c in cells) + " |")

    print("\n| Scenario | Detection | Wait for a step | Planning | Application "
          "| Target met | Runs |")
    print("|---|---|---|---|---|---|---|")
    for scenario, r in summary["reaction"].items():
        print(f"| {scenario} | {spread_cell(r['detection_ms'])} | {spread_cell(r['wait_ms'])} "
              f"| {spread_cell(r['planning_ms'])} | {spread_cell(r['applying_ms'])} "
              f"| {r['target_met']}/{r['plans']} | {len(r['entries'])} |")
    return 0


def rate_cell(cell: dict | None) -> str:
    """A cell's rate, its runs and its Wilson interval, in %."""
    if cell is None:
        return "-"
    low, high = (round(100 * b) for b in cell["wilson_95"])
    return (f"{100 * cell['rate']:.0f} ({cell['survived']}/{cell['runs']}) "
            f"[{low}, {high}]")


def spread_cell(spread: dict | None) -> str:
    """A stage's median / P90 in ms, and how many."""
    return "-" if spread is None else (
        f"{spread['median']:.1f} / {spread['p90']:.1f} (n={spread['n']})")


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
