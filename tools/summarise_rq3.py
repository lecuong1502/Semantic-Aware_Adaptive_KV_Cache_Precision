#!/usr/bin/env python3
"""Summarise RQ3's needle runs, and answer its closing question (#152).

    .venv/bin/python tools/summarise_rq3.py --issue 152

Verifies the benchmark log's chain, then pools the screened set's runs
(microinfer.rq3_summary): accuracy by condition, level and depth with
bootstrap 95% intervals, and McNemar's test of the semantic source against
uniform and against random, paired by sample. Logs the summary as an
"rq3-summary" entry naming, by their hashes, the screening and every run it
pooled, and prints it as tables, by condition and by depth, with the
closing question's answer, whatever it is. Refuses a dirty tree.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "tools"))

from rq3 import NEEDLE_SEED  # noqa: E402

from microinfer import benchlog, needles, rq3, rq3_summary  # noqa: E402


def interval(pair: list[float], digits: int = 2) -> str:
    return f"[{pair[0]:.{digits}f}, {pair[1]:.{digits}f}]"


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--issue", type=int, required=True)
    parser.add_argument("--model", default="qwen2.5-1.5b-instruct")
    parser.add_argument("--needle-seed", type=int, default=NEEDLE_SEED)
    parser.add_argument("--haystack-tokens", type=int, default=needles.DEFAULT_HAYSTACK_TOKENS)
    parser.add_argument("--plan-seed", type=int, default=0)
    parser.add_argument("--experiments", nargs="+", default=list(rq3.EXPERIMENTS),
                        choices=rq3.EXPERIMENTS)
    parser.add_argument("--log", type=Path, default=benchlog.DEFAULT_LOG)
    args = parser.parse_args(argv)
    if benchlog.environment(args.log)["git_dirty"]:
        parser.error("tracked files have uncommitted changes; commit first, so that the "
                     "entry names the code that produced it")

    benchlog.verify(args.log)  # the hashes the summary names are of the entries as appended
    try:
        summary = rq3_summary.summarise(benchlog.read(args.log), args.model, args.needle_seed,
                                        args.haystack_tokens, plan_seed=args.plan_seed,
                                        experiments=args.experiments)
    except ValueError as exc:
        parser.error(str(exc))
    benchlog.append("rq3-summary", model=args.model, context_length=None, precision_tiers=None,
                    config={"issue": args.issue, "needle_seed": args.needle_seed,
                            "haystack_tokens": args.haystack_tokens,
                            "plan_seed": args.plan_seed, "experiments": args.experiments},
                    results=summary, log=args.log)

    for x, conditions in summary["accuracy"].items():
        print(f"\n### {x}\n")
        print("| Condition | Correct | Bootstrap 95% | Wilson 95% | Effective bits "
              "| Needle pages at INT2 |")
        print("|---|---|---|---|---|---|")
        for name, a in conditions.items():
            print(f"| {name} | {a['correct']}/{a['n']} | {interval(a['bootstrap_95'])} "
                  f"| {interval(a['wilson_95'])} | {a['effective_bits']:.2f} "
                  f"| {a['needle_page_tiers'].get('INT2', 0):.0%} |")
        depths = list(next(iter(conditions.values()))["by_depth"])
        print(f"\n| Condition | {' | '.join(depths)} |")
        print(f"|---|{'---|' * len(depths)}")
        for name, a in conditions.items():
            cells = " | ".join(f"{d['correct']}/{d['n']}" for d in a["by_depth"].values())
            print(f"| {name} | {cells} |")
        print("\n| Level | Pair | Only semantic | Only other | McNemar p "
              "| Difference, bootstrap 95% |")
        print("|---|---|---|---|---|---|")
        for level, pairs in summary["paired"][x].items():
            for pair, p in pairs.items():
                print(f"| {level} | {pair} | {p['semantic_only']} | {p['other_only']} "
                      f"| {p['mcnemar_p']:.3g} | {p['difference']:+.2f} "
                      f"{interval(p['difference_bootstrap_95'])} |")
    print(f"\n**{summary['closing']['question']}**\n\n{summary['closing']['answer']}.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
