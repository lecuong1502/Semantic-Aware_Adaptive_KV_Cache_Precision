"""Run needle samples under ADR-0017's conditions, and log each run (#147).

Every sample chosen, under every condition chosen, in every experiment
chosen: one entry in the benchmark log a run, naming the sample, the
condition, the experiment and the code that produced it, with the plan, the
average effective bits, the answer and whether it is correct (microinfer.rq3).
By default, the whole needle set under every condition in the
question-blind experiment; with --screened, the set tools/screen_needles.py
kept at FP16 (#148). A run already logged at the same code, for the
model, set and plan seed, is not run again, whatever its issue, so that a
run of hours stopped part way, its log committed or not, is taken up by the
same command (rq3.logged). A sample's conditions that hold turn 1 alike
run it once (rq3.runs, #172). Refuses a dirty tree, as the checkpoint
recorder does.
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

from microinfer import Engine, benchlog, needles, rq3  # noqa: E402

#: The needle set's seed, fixed for Milestone 3 (#140).
NEEDLE_SEED = 140


def conditions_for(chosen: list[str] | None, *, screened: bool) -> list[str]:
    """The conditions to run, each once: those chosen, "sources" for every
    score source at each level, or every one, but for the screened set fp16,
    which screening ran (#148)."""
    if chosen is not None:
        named = (rq3.SOURCE_CONDITIONS if name == "sources" else [name] for name in chosen)
        return list(dict.fromkeys(n for names in named for n in names))
    return [c.name for c in rq3.CONDITIONS if not (screened and c.name == "fp16")]


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--issue", type=int, required=True,
                        help="the ticket these runs are made for")
    parser.add_argument("--model", default="qwen2.5-1.5b-instruct")
    parser.add_argument("--conditions", nargs="+", default=None,
                        choices=["sources", *(c.name for c in (*rq3.CONDITIONS, rq3.NO_PLAN))],
                        help="sources: every source at each level (#149); default: every "
                             "condition, but fp16 with --screened, whose FP16 runs screening "
                             "logged")
    parser.add_argument("--experiments", nargs="+", default=["question-blind"],
                        choices=rq3.EXPERIMENTS)
    parser.add_argument("--samples", nargs="+", default=None,
                        help="sample ids, such as d50-0 (default: the whole set)")
    parser.add_argument("--screened", action="store_true",
                        help="the screened set, from the last needle-screening logged for "
                             "the model and seed")
    parser.add_argument("--samples-per-depth", type=int, default=None,
                        help="draw more a depth, for screening's replacements")
    parser.add_argument("--needle-seed", type=int, default=NEEDLE_SEED)
    parser.add_argument("--plan-seed", type=int, default=0,
                        help="the random source's seed, with each sample's id (rq3.sample_seed)")
    parser.add_argument("--haystack-tokens", type=int, default=needles.DEFAULT_HAYSTACK_TOKENS)
    parser.add_argument("--log", type=Path, default=benchlog.DEFAULT_LOG)
    args = parser.parse_args(argv)

    environment = benchlog.environment(args.log)
    if environment["git_dirty"]:
        parser.error("tracked files have uncommitted changes; commit first, so that "
                     "each entry names the code that produced it")
    done = rq3.logged(args.model, args.needle_seed, args.haystack_tokens,
                      environment["git_commit"], args.log, plan_seed=args.plan_seed)

    per_depth = args.samples_per_depth or needles.SAMPLES_PER_DEPTH
    if args.screened:
        if args.samples is not None or args.samples_per_depth is not None:
            parser.error("--screened chooses the samples, and how many a depth are drawn")
        args.samples, per_depth = rq3.screened(args.model, args.needle_seed,
                                               args.haystack_tokens, args.log)
    args.conditions = conditions_for(args.conditions, screened=args.screened)
    args.experiments = list(dict.fromkeys(args.experiments))

    engine = Engine(REPO / "models" / args.model)
    every = needles.needle_set(held_out_text(), engine.tokenizer, seed=args.needle_seed,
                               samples_per_depth=per_depth,
                               haystack_tokens=args.haystack_tokens)
    chosen = every if args.samples is None else [s for s in every if s.sample_id in args.samples]
    missing = set(args.samples or ()) - {s.sample_id for s in chosen}
    if missing:
        parser.error(f"no samples {sorted(missing)} in the set")
    runs = [(sample, experiment, name) for sample in chosen for experiment in args.experiments
            for name in args.conditions]
    todo = [r for r in runs if (r[0].sample_id, r[1], r[2]) not in done]
    print(f"{len(runs) - len(todo)} of {len(runs)} runs logged at this code already; "
          f"{len(todo)} to run", flush=True)
    if not todo:
        return 0

    engine.load_weights()
    start = time.perf_counter()
    # A sample's conditions in one experiment share turn 1 where they hold
    # it alike (rq3.runs, #172); each run is logged as it ends.
    together: dict[tuple, list[str]] = {}
    for sample, experiment, name in todo:
        together.setdefault((sample.sample_id, experiment), []).append(name)
    by_id = {s.sample_id: s for s in chosen}
    ran = 0
    for (sample_id, experiment), names in together.items():
        sample = by_id[sample_id]
        for cond, results in rq3.runs(engine, sample, [rq3.condition(n) for n in names],
                                      experiment, plan_seed=args.plan_seed):
            benchlog.append(
                "rq3-needle", model=args.model, context_length=len(sample.turn_1),
                precision_tiers=results["tier_shares"],
                config=rq3.config(sample, cond, experiment, needle_seed=args.needle_seed,
                                  plan_seed=args.plan_seed,
                                  haystack_tokens=args.haystack_tokens, issue=args.issue),
                results=results, log=args.log)
            ran += 1
            print(f"{ran}/{len(todo)} {sample_id} {experiment} {cond.name}: "
                  f"{'right' if results['correct'] else 'WRONG'} "
                  f"{results['effective_bits']:.2f} bits  {results['answer']!r}  "
                  f"({time.perf_counter() - start:.0f} s)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
