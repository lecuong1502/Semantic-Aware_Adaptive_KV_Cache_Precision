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

With --long, the 32K subset's runs instead (#156, rq3_summary.summarise_long),
logged as an "rq3-long-summary" entry: the direction they point, no test.

    .venv/bin/python tools/summarise_rq3.py --issue 156 --long

With --ablations, ADR-0018's quality ablations instead (#158,
rq3_summary.summarise_ablations), logged as an "rq3-ablation-summary"
entry: each one's change in accuracy from the full policy, its interval,
McNemar's test and its effective bits.

    .venv/bin/python tools/summarise_rq3.py --issue 158 --ablations
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


def table(header: list[str], rows: list[list]) -> None:
    """Print a Markdown table."""
    print(f"| {' | '.join(header)} |")
    print(f"|{'---|' * len(header)}")
    for row in rows:
        print(f"| {' | '.join(str(cell) for cell in row)} |")


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--issue", type=int, required=True)
    parser.add_argument("--model", default="qwen2.5-1.5b-instruct")
    parser.add_argument("--needle-seed", type=int, default=NEEDLE_SEED)
    parser.add_argument("--haystack-tokens", type=int, default=needles.DEFAULT_HAYSTACK_TOKENS)
    parser.add_argument("--plan-seed", type=int, default=0)
    parser.add_argument("--experiments", nargs="+", default=list(rq3.EXPERIMENTS),
                        choices=rq3.EXPERIMENTS)
    summaries_of_their_own = parser.add_mutually_exclusive_group()
    summaries_of_their_own.add_argument("--long", action="store_true",
                                        help="the 32K subset (#156)")
    summaries_of_their_own.add_argument("--ablations", action="store_true",
                                        help="ADR-0018's quality ablations (#158)")
    parser.add_argument("--log", type=Path, default=benchlog.DEFAULT_LOG)
    args = parser.parse_args(argv)
    if benchlog.environment(args.log)["git_dirty"]:
        parser.error("tracked files have uncommitted changes; commit first, so that the "
                     "entry names the code that produced it")

    benchlog.verify(args.log)  # the hashes the summary names are of the entries as appended
    if args.long:
        return long_subset(parser, args)
    if args.ablations:
        return ablations(parser, args)
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
        table(["Condition", "Correct", "Bootstrap 95%", "Wilson 95%", "Effective bits",
               "Needle pages at INT2"],
              [[name, f"{a['correct']}/{a['n']}", interval(a["bootstrap_95"]),
                interval(a["wilson_95"]), f"{a['effective_bits']:.2f}",
                f"{a['needle_page_tiers'].get('INT2', 0):.0%}"] for name, a in conditions.items()])
        depths = list(next(iter(conditions.values()))["by_depth"])
        print()
        table(["Condition", *depths],
              [[name, *(f"{d['correct']}/{d['n']}" for d in a["by_depth"].values())]
               for name, a in conditions.items()])
        print()
        table(["Level", "Pair", "Only semantic", "Only other", "McNemar p",
               "Difference, bootstrap 95%"],
              [[level, pair, p["semantic_only"], p["other_only"], f"{p['mcnemar_p']:.3g}",
                f"{p['difference']:+.2f} {interval(p['difference_bootstrap_95'])}"]
               for level, pairs in summary["paired"][x].items() for pair, p in pairs.items()])
    print(f"\n**{summary['closing']['question']}**\n\n{summary['closing']['answer']}.")
    return 0


def long_subset(parser: argparse.ArgumentParser, args: argparse.Namespace) -> int:
    """Log and print the 32K subset's summary."""
    try:
        summary = rq3_summary.summarise_long(benchlog.read(args.log), args.model,
                                             args.needle_seed, plan_seed=args.plan_seed)
    except ValueError as exc:
        parser.error(str(exc))
    benchlog.append(rq3_summary.LONG_SUMMARY_KIND, model=args.model, context_length=None,
                    precision_tiers=None,
                    config={"issue": args.issue, "needle_seed": args.needle_seed,
                            "haystack_tokens": summary["haystack_tokens"],
                            "plan_seed": args.plan_seed, "experiments": [summary["experiment"]]},
                    results=summary, log=args.log)
    accuracy = summary["accuracy"]
    depths = list(next(iter(accuracy.values()))["by_depth"])
    print(f"\n### 32K subset, {summary['experiment']}\n")
    table(["Condition", "Correct", "Wilson 95%", "Effective bits", *depths],
          [[name, f"{a['correct']}/{a['n']}", interval(a["wilson_95"]),
            f"{a['effective_bits']:.2f}", *("right" if r else "WRONG"
                                             for r in a["by_depth"].values())]
           for name, a in accuracy.items()])
    p = summary["paired"]
    print(f"\nSemantic against uniform at 80%: {p['semantic_only']} only semantic answered, "
          f"{p['uniform_only']} only uniform, {p['both']} both, {p['neither']} neither; "
          f"direction: {summary['direction'] or 'none'} ({summary['claim']}).")
    return 0


def ablations(parser: argparse.ArgumentParser, args: argparse.Namespace) -> int:
    """Log and print the quality ablations' summary."""
    try:
        summary = rq3_summary.summarise_ablations(benchlog.read(args.log), args.model,
                                                  args.needle_seed, args.haystack_tokens,
                                                  plan_seed=args.plan_seed)
    except ValueError as exc:
        parser.error(str(exc))
    benchlog.append(rq3_summary.ABLATION_SUMMARY_KIND, model=args.model, context_length=None,
                    precision_tiers=None,
                    config={"issue": args.issue, "needle_seed": args.needle_seed,
                            "haystack_tokens": args.haystack_tokens,
                            "plan_seed": args.plan_seed, "experiments": [summary["experiment"]]},
                    results=summary, log=args.log)
    against = ", ".join(f"{name} {a['correct']}/{a['n']} at {a['effective_bits']:.2f} bits"
                        for name, a in summary["against"].items())
    print(f"\n### Quality ablations, {summary['experiment']}, against {against}\n")
    table(["Ablation", "Element", "Against", "Correct", "Change, bootstrap 95%",
           "Only ablation", "Only against", "McNemar p", "Effective bits", "Short plans"],
          [[name, a["element"], a["against"], f"{a['correct']}/{a['n']}",
            f"{a['change']:+.2f} {interval(a['change_bootstrap_95'])}", a["ablation_only"],
            a["against_only"], f"{a['mcnemar_p']:.3g}", f"{a['effective_bits']:.2f}", a["short"]]
           for name, a in summary["ablations"].items()])
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
