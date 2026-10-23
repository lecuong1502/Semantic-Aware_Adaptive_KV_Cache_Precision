"""Run the recovery task on the screened needle set, and log each run (#155).

Two steps, each a command (microinfer.recovery):
1. By default: the screened needle set (#148) under each condition chosen,
   FP16 and every score source at each level, or those --conditions names:
   turn 1, the plan, turn 2, the restore of every page to FP16, and turn 3,
   the question again. One rq3-recovery entry a run. A sample's source
   conditions run turn 1 once (rq3.runs).
2. `--summarise`: the recovery-summary entry, each condition's accuracy at
   turn 2 and turn 3 and the samples turn 3 recovered and lost, and the
   semantic source paired against uniform and random at each, and its
   tables.

A run already logged at the same code, for the model, set and plan seed,
is not run again, so that a run of hours stopped part way is taken up by
the same command. Refuses a dirty tree, as the checkpoint recorder does.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "tools"))

from perplexity import held_out_text  # noqa: E402
from rq3 import NEEDLE_SEED  # noqa: E402

from microinfer import Engine, benchlog, needles, recovery, rq3  # noqa: E402


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--issue", type=int, required=True,
                        help="the ticket these runs are made for")
    parser.add_argument("--model", default="qwen2.5-1.5b-instruct")
    parser.add_argument("--summarise", action="store_true",
                        help="log the summary of the screened set's runs")
    parser.add_argument("--conditions", nargs="+", default=list(recovery.CONDITIONS),
                        choices=recovery.CONDITIONS,
                        help="default: fp16 and every score source at each level")
    parser.add_argument("--needle-seed", type=int, default=NEEDLE_SEED)
    parser.add_argument("--plan-seed", type=int, default=0,
                        help="the random source's seed, with each sample's id (rq3.sample_seed)")
    parser.add_argument("--haystack-tokens", type=int, default=needles.DEFAULT_HAYSTACK_TOKENS)
    parser.add_argument("--log", type=Path, default=benchlog.DEFAULT_LOG)
    args = parser.parse_args(argv)
    of = recovery.NeedleSet(args.model, args.needle_seed, args.haystack_tokens)
    names = list(dict.fromkeys(args.conditions))

    if args.summarise:
        benchlog.verify(args.log)
        summary = recovery.summarise(benchlog.read(args.log), of, plan_seed=args.plan_seed,
                                     conditions=names)
        entry = benchlog.append(
            recovery.SUMMARY_KIND, model=args.model, context_length=None,
            precision_tiers=None,
            config={**of.config(), "plan_seed": args.plan_seed, "issue": args.issue},
            results=summary, log=args.log)
        print(tables(summary))
        print(f"\nlogged {entry['sha256'][:12]}")
        return 0

    environment = benchlog.environment(args.log)
    if environment["git_dirty"]:
        parser.error("tracked files have uncommitted changes; commit first, so that "
                     "each entry names the code that produced it")
    kept, draws = rq3.screened(args.model, args.needle_seed, args.haystack_tokens, args.log)
    done = recovery.logged(of, environment["git_commit"], args.log, plan_seed=args.plan_seed)
    todo = {s: [n for n in names if (s, n) not in done] for s in kept}
    count = sum(len(v) for v in todo.values())
    print(f"{len(kept) * len(names) - count} of {len(kept) * len(names)} runs logged at this "
          f"code already; {count} to run", flush=True)
    if not count:
        return 0
    engine = Engine(REPO / "models" / args.model)
    every = needles.needle_set(held_out_text(), engine.tokenizer, seed=args.needle_seed,
                               samples_per_depth=draws, haystack_tokens=args.haystack_tokens)
    by_id = {s.sample_id: s for s in every}
    engine.load_weights()
    start, ran = time.perf_counter(), 0
    for sample_id, left in todo.items():
        if not left:
            continue
        sample = by_id[sample_id]
        for cond, results in recovery.runs(engine, sample, [rq3.condition(n) for n in left],
                                           plan_seed=args.plan_seed):
            benchlog.append(
                recovery.KIND, model=args.model, context_length=len(sample.turn_1),
                precision_tiers=results["tier_shares"],
                config=rq3.config(sample, cond, recovery.EXPERIMENT,
                                  needle_seed=args.needle_seed, plan_seed=args.plan_seed,
                                  haystack_tokens=args.haystack_tokens, issue=args.issue),
                results=results, log=args.log)
            ran += 1
            restore = results["restore"]
            print(f"{ran}/{count} {sample_id} {cond.name}: "
                  f"turn 2 {'right' if results['correct'] else 'WRONG'}, "
                  f"turn 3 {'right' if results['turn_3_correct'] else 'WRONG'} "
                  f"{results['turn_3_answer']!r}"
                  + ("" if restore is None else f"  restore {restore['seconds']:.2f} s")
                  + f"  ({time.perf_counter() - start:.0f} s)", flush=True)
    return 0


def tables(summary: dict) -> str:
    """The summary's accuracy by condition and turn, and its pairs, as markdown."""
    rows = [f"{summary['samples']} needle samples", "",
            "| Condition | Turn 2 | Turn 3 | Wilson 95% (turn 3) | Recovered | Lost | "
            "McNemar p | Repeats | Restore s |",
            "|---|---|---|---|---|---|---|---|---|"]
    for name, a in summary["accuracy"].items():
        two, three = a["turn 2"], a["turn 3"]
        low, high = three["wilson_95"]
        restore = "" if a["restore"] is None else f"{a['restore']['seconds_mean']:.2f}"
        rows.append(f"| {name} | {two['correct']}/{two['n']} | {three['correct']}/{three['n']} "
                    f"| [{low:.2f}, {high:.2f}] | {a['recovered']} | {a['lost']} | "
                    f"{a['mcnemar_p']:.3g} | {a['turn_3_repeats_turn_2']} | {restore} |")
    rows += ["", "| Turn | Level | Pair | Only semantic | Only other | McNemar p | Difference |",
             "|---|---|---|---|---|---|---|"]
    for turn, levels in summary["paired"].items():
        for level, pairs in levels.items():
            for pair, p in pairs.items():
                rows.append(f"| {turn} | {level} | {pair} | {p['semantic_only']} | "
                            f"{p['other_only']} | {p['mcnemar_p']:.3g} | "
                            f"{p['difference']:+.2f} |")
    return "\n".join(rows)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
