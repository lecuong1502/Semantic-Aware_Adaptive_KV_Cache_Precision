#!/usr/bin/env python3
"""Log two survival runs against time, for the paper's timeline (#212).

    .venv/bin/python tools/summarise_timeline.py --issue 212
        [--adaptive 94948638] [--static e2f55721]

The log names a survival run's recordings by sha256 and does not hold them.
This reads what the timeline plots from the recordings of two runs, an
adaptive engine's and a static one's under the same contention
(microinfer.survival_timeline), and logs it as one "survival-timeline"
entry, so that tools/figure_survival.py draws the figure from the log alone.
The runs are named by the start of their entries' hashes. The defaults are
the ones the paper reports: the semantic source's S6 run (#155) and static
FP16's S5 run (#108), whose step is S6's before its release. It runs no
engine. Refuses a dirty tree, and two runs whose contention differs.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from microinfer import benchlog, survival_timeline  # noqa: E402
from microinfer.footprint import MIB  # noqa: E402

#: Where the runs' recordings are kept.
RECORDINGS = (REPO / "data" / "survival", REPO / "data" / "survival" / "matrix")
KINDS = ("survival", "pulse-release")
#: What the contention of two runs is set by: they must agree.
CONTENTION = ("contention_at", "shortfall_mib", "thresholds")


def named(parser: argparse.ArgumentParser, entries: list[dict], prefix: str,
          engine: str) -> dict:
    found = [e for e in entries if e["kind"] in KINDS and e["sha256"].startswith(prefix)]
    if len(found) != 1 or found[0]["config"]["engine"] != engine:
        parser.error(f"{prefix} names {len(found)} survival runs of the log, not one of "
                     f"the {engine} engine")
    return found[0]


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--issue", type=int, required=True)
    parser.add_argument("--adaptive", default="94948638")
    parser.add_argument("--static", default="e2f55721")
    parser.add_argument("--log", type=Path, default=benchlog.DEFAULT_LOG)
    args = parser.parse_args(argv)
    if benchlog.environment(args.log)["git_dirty"]:
        parser.error("tracked files have uncommitted changes; commit first, so that the "
                     "entry names the code that produced it")
    benchlog.verify(args.log)  # the hashes the entry names are of the runs as appended
    entries = benchlog.read(args.log)
    runs = {engine: named(parser, entries, getattr(args, engine), engine)
            for engine in ("static", "adaptive")}
    static, adaptive = runs["static"], runs["adaptive"]
    differ = [k for k in CONTENTION if static["config"][k] != adaptive["config"][k]]
    differ += [k for k in ("model", "context_length") if static[k] != adaptive[k]]
    if differ:
        parser.error(f"the two runs differ in {', '.join(differ)}: not one contention")
    read = {engine: survival_timeline.read(run, RECORDINGS) for engine, run in runs.items()}
    thresholds = adaptive["config"]["thresholds"]
    benchlog.append(survival_timeline.KIND, model=adaptive["model"],
                    context_length=adaptive["context_length"], precision_tiers=None,
                    config={"issue": args.issue, "bin_s": survival_timeline.BIN_S,
                            "until_s": survival_timeline.UNTIL_S,
                            "runs": {engine: run["sha256"] for engine, run in runs.items()}},
                    results={**read, "thresholds_mib": {
                        "RED": thresholds["red_below_bytes"] / MIB,
                        "YELLOW": thresholds["yellow_below_bytes"] / MIB}},
                    log=args.log)
    print(json.dumps({engine: {k: run[k] for k in ("entry", "ending", "end_s", "upgrades_s")}
                      | {"downgrades": len(run["downgrades_s"]),
                         "points": sum(len(run[k]) for k in ("headroom_mib", "cache_mib"))}
                      for engine, run in read.items()}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
