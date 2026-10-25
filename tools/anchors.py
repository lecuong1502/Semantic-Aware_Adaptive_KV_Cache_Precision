"""Screen, run and summarise the anchor-fact dialogue, and log each (#154).

Three steps, each a command (microinfer.anchors):
1. `--screen`: the set drawn with up to --draws dialogues a gap, run at FP16
   in their order until five a gap are answered rightly (rq3.screen by
   gap). Each FP16 run is an rq3-anchor entry, screening true, and the
   screened set an anchor-screening entry; a gap short after --draws is
   logged as a screening not complete.
2. By default: the screened set under each condition chosen, every score
   source at each level, or those --conditions names; one rq3-anchor
   entry a run. A dialogue's source conditions run its turns once
   (anchors.runs).
3. `--summarise`: the anchor-summary entry, recall per condition by gap and
   the semantic source paired against uniform and random, and its tables.

With --evictions, the conditions are eviction at each level (#160) and the
semantic source there, both run at the same code, and summarised against
each other (rq3_summary.EVICTION_NAMES):

    .venv/bin/python tools/anchors.py --issue 160 --evictions
    .venv/bin/python tools/anchors.py --issue 160 --evictions --summarise

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
from summarise_rq3 import print_evictions  # noqa: E402

from microinfer import Engine, anchors, benchlog, rq3, rq3_summary  # noqa: E402

FP16 = rq3.condition("fp16")


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--issue", type=int, required=True,
                        help="the ticket these runs are made for")
    parser.add_argument("--model", default="qwen2.5-1.5b-instruct")
    step = parser.add_mutually_exclusive_group()
    step.add_argument("--screen", action="store_true", help="screen the set at FP16")
    step.add_argument("--summarise", action="store_true",
                      help="log the summary of the screened set's runs")
    parser.add_argument("--conditions", nargs="+", default=None,
                        choices=[*rq3.SOURCE_CONDITIONS, *rq3.EVICTION_CONDITIONS],
                        help="default: every score source at each level; fp16's runs are "
                             "screening's")
    parser.add_argument("--evictions", action="store_true",
                        help="eviction at each level and the semantic source there (#160)")
    parser.add_argument("--per-gap", type=int, default=anchors.SAMPLES_PER_GAP)
    parser.add_argument("--draws", type=int, default=15,
                        help="dialogues a gap drawn, at most, to screen")
    parser.add_argument("--anchor-seed", type=int, default=anchors.ANCHOR_SEED)
    parser.add_argument("--plan-seed", type=int, default=0,
                        help="the random source's seed, with each dialogue's id")
    parser.add_argument("--turns", type=int, default=anchors.TURNS)
    parser.add_argument("--passage-tokens", type=int, default=anchors.PASSAGE_TOKENS)
    parser.add_argument("--log", type=Path, default=benchlog.DEFAULT_LOG)
    args = parser.parse_args(argv)
    if args.evictions:
        if args.screen or args.conditions:
            parser.error("--evictions chooses its conditions, of the screened set; drop "
                         "--screen and --conditions")
        args.conditions = list(rq3_summary.EVICTION_NAMES)
    args.conditions = args.conditions or list(rq3.SOURCE_CONDITIONS)
    of = anchors.AnchorSet(args.model, args.anchor_seed, args.turns, args.passage_tokens)

    if args.summarise:
        benchlog.verify(args.log)
        names = list(dict.fromkeys(args.conditions))
        summary = anchors.summarise(benchlog.read(args.log), of, plan_seed=args.plan_seed,
                                    conditions=[FP16.name, *names])
        entry = benchlog.append(
            "anchor-summary", model=args.model, context_length=None,
            precision_tiers=None,
            config={**of.config(), "plan_seed": args.plan_seed, "issue": args.issue},
            results=summary, log=args.log)
        print(tables(summary))
        if summary["evictions"] is not None:
            print()
            print_evictions(summary["evictions"])
        print(f"\nlogged {entry['sha256'][:12]}")
        return 0

    environment = benchlog.environment(args.log)
    if environment["git_dirty"]:
        parser.error("tracked files have uncommitted changes; commit first, so that "
                     "each entry names the code that produced it")
    commit = environment["git_commit"]
    engine = Engine(REPO / "models" / args.model)
    text = held_out_text()

    def log_run(dialogue, cond, results, **more):
        benchlog.append(
            anchors.KIND, model=args.model,
            context_length=results["plan_positions"] + 1,
            precision_tiers=results["tier_shares"],
            config=anchors.config(dialogue, cond, anchor_seed=of.seed, issue=args.issue, **more),
            results=results,
            log=args.log)

    start = time.perf_counter()
    if args.screen:
        done = {s: right for (s, name), right in anchors.logged(
            of, commit, args.log, screening=True).items() if name == FP16.name}
        drawn = of.dialogues(text, engine.tokenizer, args.draws)
        engine.load_weights()

        def is_right(dialogue) -> bool:
            if dialogue.sample_id in done:
                print(f"{dialogue.sample_id}: logged at this code", flush=True)
                return done[dialogue.sample_id]
            results = anchors.run(engine, dialogue, FP16)
            log_run(dialogue, FP16, results, plan_seed=0, screening=True)
            print(f"{dialogue.sample_id}: {'right' if results['correct'] else 'WRONG'} "
                  f"{results['answer']!r}  ({time.perf_counter() - start:.0f} s)", flush=True)
            return results["correct"]

        try:
            kept, rejected = rq3.screen(drawn, is_right, args.per_gap,
                                        anchors.by_gap(args.turns))
            short = None
        except rq3.ShortScreening as failed:
            kept, rejected, short = failed.kept, failed.rejected, failed
        benchlog.append(
            anchors.SCREENING_KIND, model=args.model, context_length=None,
            precision_tiers={"FP16": 1.0},
            config={**of.config(), "per_gap": args.per_gap, "draws": args.draws,
                    "condition": FP16.name, "experiment": anchors.EXPERIMENT,
                    "issue": args.issue},
            results={"kept": [d.sample_id for d in kept],
                     "rejected": [d.sample_id for d in rejected], "complete": short is None,
                     "short": None if short is None else {
                         anchors.gap_name(g): n for g, n in short.short.items()},
                     "seconds": time.perf_counter() - start},
            log=args.log)
        if short is not None:
            print(f"not complete: {short}")
            return 1
        print(f"kept {len(kept)}; replaced {len(rejected)}: {[d.sample_id for d in rejected]}")
        return 0

    screening = anchors.last_screening(benchlog.read(args.log), of)
    kept = screening["results"]["kept"]
    every = of.dialogues(text, engine.tokenizer, screening["config"]["draws"])
    by_id = {d.sample_id: d for d in every}
    names = list(dict.fromkeys(args.conditions))
    done = anchors.logged(of, commit, args.log, plan_seed=args.plan_seed)
    todo = {s: [n for n in names if (s, n) not in done] for s in kept}
    count = sum(len(v) for v in todo.values())
    print(f"{len(kept) * len(names) - count} of {len(kept) * len(names)} runs logged at this "
          f"code already; {count} to run", flush=True)
    if not count:
        return 0
    engine.load_weights()
    ran = 0
    for sample_id, left in todo.items():
        if not left:
            continue
        dialogue = by_id[sample_id]
        for cond, results in anchors.runs(engine, dialogue, [rq3.condition(n) for n in left],
                                          plan_seed=args.plan_seed):
            log_run(dialogue, cond, results, plan_seed=args.plan_seed)
            ran += 1
            print(f"{ran}/{count} {sample_id} {cond.name}: "
                  f"{'right' if results['correct'] else 'WRONG'} "
                  f"{results['effective_bits']:.2f} bits  {results['answer']!r}  "
                  f"({time.perf_counter() - start:.0f} s)", flush=True)
    return 0


def tables(summary: dict) -> str:
    """The summary's recall, by condition and gap, and its pairs, as markdown."""
    recall = summary["recall"]
    gap_names = list(next(iter(recall.values()))["by_gap"])
    rows = [f"{summary['dialogues']} dialogues of {summary['turns']} turns", "",
            "| Condition | Recall | Wilson 95% | Bits | " + " | ".join(gap_names) + " |",
            "|---|---|---|---|" + "---|" * len(gap_names)]
    for name, r in recall.items():
        low, high = r["wilson_95"]
        cells = [f"{g['correct']}/{g['n']}" for g in r["by_gap"].values()]
        rows.append(f"| {name} | {r['correct']}/{r['n']} | [{low:.2f}, {high:.2f}] | "
                    f"{r['effective_bits']:.2f} | " + " | ".join(cells) + " |")
    rows += ["", "| Level | Pair | Only semantic | Only other | McNemar p | Difference |",
             "|---|---|---|---|---|---|"]
    for level, pairs in summary["paired"].items():
        for pair, p in pairs.items():
            rows.append(f"| {level} | {pair} | {p['semantic_only']} | {p['other_only']} | "
                        f"{p['mcnemar_p']:.3g} | {p['difference']:+.2f} |")
    return "\n".join(rows)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
