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
run it once (rq3.runs, #172), before the others of the sample. Refuses a dirty tree, as the checkpoint
recorder does.

With --long, the 32K subset (#156): one sample a depth of haystacks of
needles.LONG_HAYSTACK_TOKENS, question-blind, under FP16 and both sources
at 80% (rq3_summary.LONG_CONDITIONS), 30 runs:

    .venv/bin/python tools/rq3.py --issue 156 --long

A sample's prefill near 32K positions takes about 10 minutes on the RTX
4050, and a token decoded there about 1.8 s, so the subset takes hours;
stopped, the same command takes it up.

With --ablations, ADR-0018's quality ablations (#158): the screened set,
question-blind, under the full policy, semantic at 80%, and each of the
nine ablations (rq3.ablations), the layer weights' from the last
layer-sensitivity logged for the model (#145), 500 runs; --haystack-tokens
names the screened set:

    .venv/bin/python tools/rq3.py --issue 158 --ablations

A run takes about 77 s at 8K positions on the RTX 4050, as #149's did, so
the 500 take about 11 hours; they are best run in batches, each stopped
with Ctrl-C and taken up by the same command. The full policy is run again
beside them, at the same code, rather than paired with #149's runs.
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

from microinfer import Engine, benchlog, controller, needles, rq3, rq3_summary  # noqa: E402

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
    parser.add_argument("--long", action="store_true",
                        help="the 32K subset (#156): its samples, conditions and experiment")
    parser.add_argument("--ablations", action="store_true",
                        help="ADR-0018's quality ablations (#158) on the screened set")
    parser.add_argument("--experiments", nargs="+", default=None,
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
    parser.add_argument("--haystack-tokens", type=int, default=None,
                        help=f"default: {needles.DEFAULT_HAYSTACK_TOKENS}")
    parser.add_argument("--log", type=Path, default=benchlog.DEFAULT_LOG)
    args = parser.parse_args(argv)
    if args.long and args.ablations:
        parser.error("--long and --ablations are runs of their own; choose one")
    by_name = {c.name: c for c in (*rq3.CONDITIONS, rq3.NO_PLAN)}
    if args.long or args.ablations:
        chosen = [flag for flag, value in (
            ("--screened", args.screened), ("--samples-per-depth", args.samples_per_depth),
            ("--conditions", args.conditions), ("--experiments", args.experiments),
            ("--haystack-tokens", args.long and args.haystack_tokens),
            ("--samples", args.samples)) if value]
        if chosen:
            parser.error(f"{'--long' if args.long else '--ablations'} chooses its runs; "
                         f"drop {' '.join(chosen)}")
    if args.long:
        args.haystack_tokens, args.samples_per_depth = needles.LONG_HAYSTACK_TOKENS, 1
        args.conditions = list(rq3_summary.LONG_CONDITIONS)
    if args.ablations:
        try:
            weights = controller.logged_layer_weights(args.model, args.log)
        except ValueError as exc:
            parser.error(str(exc))
        ablations = rq3.ablations(weights)
        by_name.update({c.name: c for c in ablations})
        args.conditions = [rq3.FULL_POLICY.name, *(c.name for c in ablations)]
        args.screened = True
    args.haystack_tokens = args.haystack_tokens or needles.DEFAULT_HAYSTACK_TOKENS
    args.experiments = args.experiments or [rq3_summary.CLOSING_EXPERIMENT]

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
    # it alike (rq3.runs, #172), and run in its order, the sharing ones
    # first; each run is logged as it ends.
    together: dict[tuple, list[str]] = {}
    for sample, experiment, name in todo:
        together.setdefault((sample.sample_id, experiment), []).append(name)
    by_id = {s.sample_id: s for s in chosen}
    ran = 0
    for (sample_id, experiment), names in together.items():
        sample = by_id[sample_id]
        for cond, results in rq3.runs(engine, sample, [by_name[n] for n in names],
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
