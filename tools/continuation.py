#!/usr/bin/env python3
"""Run continuation fidelity on WikiText-2, and log each run (#153).

    .venv/bin/python tools/continuation.py --issue 153
    .venv/bin/python tools/continuation.py --issue 153 --summarise

Each passage of the WikiText-2 test split (microinfer.continuation): its
prefix prefilled, its warm tokens forced a position a step, then under each
source condition its plan, and its measured tokens forced, their logits
compared with FP16's. One "rq3-continuation" entry a run, naming the passage,
the condition and the code that produced it, with its top-1 agreement and
KL. A run already logged at the same code is not run again, so that a run
stopped part way is taken up by the same command. --summarise pools the
runs into one "continuation-summary" entry, naming them by their hashes, and
prints it. Refuses a dirty tree, as the checkpoint recorder does.
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
from summarise_rq3 import interval  # noqa: E402

from microinfer import Engine, benchlog, continuation, rq3  # noqa: E402


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--issue", type=int, required=True)
    parser.add_argument("--model", default="qwen2.5-1.5b-instruct")
    parser.add_argument("--conditions", nargs="+", default=list(rq3.SOURCE_CONDITIONS),
                        choices=[*rq3.SOURCE_CONDITIONS, rq3.NO_PLAN.name])
    parser.add_argument("--passages", type=int, default=continuation.PASSAGES)
    parser.add_argument("--prefix-tokens", type=int, default=continuation.PREFIX)
    parser.add_argument("--warm-tokens", type=int, default=continuation.WARM)
    parser.add_argument("--measured-tokens", type=int, default=continuation.MEASURED)
    parser.add_argument("--plan-seed", type=int, default=0)
    parser.add_argument("--summarise", action="store_true",
                        help="pool the runs logged into one summary entry, and run none")
    parser.add_argument("--log", type=Path, default=benchlog.DEFAULT_LOG)
    args = parser.parse_args(argv)
    environment = benchlog.environment(args.log)
    if environment["git_dirty"]:
        parser.error("tracked files have uncommitted changes; commit first, so that "
                     "each entry names the code that produced it")
    shape = continuation.Shape(args.prefix_tokens, args.warm_tokens, args.measured_tokens,
                               args.plan_seed)
    if args.summarise:
        return summarise(parser, args, shape)

    done = continuation.logged(args.model, shape, environment["git_commit"], args.log)
    engine = Engine(REPO / "models" / args.model)
    chosen = continuation.passages(engine.encode(held_out_text()), args.passages, shape)
    todo = [(p, [rq3.condition(n) for n in args.conditions if (p.passage_id, n) not in done])
            for p in chosen]
    todo = [(p, conds) for p, conds in todo if conds]
    left = sum(len(conds) for _, conds in todo)
    print(f"{len(chosen) * len(args.conditions) - left} of {len(chosen) * len(args.conditions)} "
          f"runs logged at this code already; {left} to run", flush=True)
    if not todo:
        return 0
    engine.load_weights()
    start, ran = time.perf_counter(), 0
    for passage, conds in todo:
        for cond, results in continuation.runs(engine, passage, conds,
                                               plan_seed=args.plan_seed):
            benchlog.append(
                continuation.KIND, model=args.model,
                context_length=args.prefix_tokens + args.warm_tokens + args.measured_tokens,
                precision_tiers=results["tier_shares"],
                config=continuation.config(passage, cond, plan_seed=args.plan_seed,
                                           issue=args.issue),
                results=results, log=args.log)
            ran += 1
            print(f"{ran}/{left} {passage.passage_id} {cond.name}: top-1 {results['top1']:.3f}"
                  f"  KL {results['kl_mean']:.4f}  {results['effective_bits']:.2f} bits  "
                  f"({time.perf_counter() - start:.0f} s)", flush=True)
    return 0


def summarise(parser, args, shape) -> int:
    benchlog.verify(args.log)
    try:
        summary = continuation.summarise(benchlog.read(args.log), args.model, shape,
                                         args.passages, args.conditions)
    except ValueError as exc:
        parser.error(str(exc))
    benchlog.append("continuation-summary", model=args.model,
                    context_length=args.prefix_tokens + args.warm_tokens + args.measured_tokens,
                    precision_tiers=None,
                    config={"issue": args.issue, "conditions": args.conditions,
                            "plan_seed": args.plan_seed},
                    results=summary, log=args.log)
    print(f"\n{summary['passages']} passages of {args.prefix_tokens} + {args.warm_tokens} + "
          f"{args.measured_tokens} tokens\n")
    print("| Condition | Top-1 agreement | Bootstrap 95% | Mean KL | Bootstrap 95% | Bits "
          "| Plans short |")
    print("|---|---|---|---|---|---|---|")
    for name, c in summary["by_condition"].items():
        print(f"| {name} | {c['top1']:.4f} | {interval(c['top1_bootstrap_95'], 4)} "
              f"| {c['kl_mean']:.4f} | {interval(c['kl_bootstrap_95'], 4)} "
              f"| {c['effective_bits']:.2f} | {c['plans_short_of_target']} |")
    print("\n| Level | Pair | KL difference | Bootstrap 95% | Passages semantic lower "
          "| Top-1 difference | Bootstrap 95% |")
    print("|---|---|---|---|---|---|---|")
    for level, pairs in summary["paired"].items():
        for pair, p in pairs.items():
            print(f"| {level} | {pair} | {p['kl_difference']:+.4f} "
                  f"| {interval(p['kl_difference_bootstrap_95'], 4)} "
                  f"| {p['semantic_lower_kl']}/{summary['passages']} "
                  f"| {p['top1_difference']:+.4f} "
                  f"| {interval(p['top1_difference_bootstrap_95'], 4)} |")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
