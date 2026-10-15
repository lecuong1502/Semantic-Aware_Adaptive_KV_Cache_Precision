"""Screen the needle set at FP16, and log the screened set (#148).

ADR-0017 replaces a sample FP16 answers wrongly before the comparison, so
that the model's failures are not counted as the cache's. The set is drawn
with up to --draws samples a depth, and each depth's are run at FP16,
question-blind, in their order, until five are answered correctly
(rq3.screen): drawing more a depth leaves the first ones as they are, so
the replacements are the next of the seed. Each FP16 run is an rq3-needle
entry, and the screened set and its replacements one needle-screening
entry, which tools/rq3.py --screened reads. Refuses a dirty tree, as the
checkpoint recorder does.
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

from microinfer import Engine, benchlog, needles, rq3  # noqa: E402

FP16 = rq3.condition("fp16")
EXPERIMENT = "question-blind"


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--issue", type=int, required=True,
                        help="the ticket this screening is made for")
    parser.add_argument("--model", default="qwen2.5-1.5b-instruct")
    parser.add_argument("--needle-seed", type=int, default=NEEDLE_SEED)
    parser.add_argument("--per-depth", type=int, default=needles.SAMPLES_PER_DEPTH,
                        help="samples a depth the screened set keeps")
    parser.add_argument("--draws", type=int, default=15,
                        help="samples a depth drawn, at most, to find them")
    parser.add_argument("--haystack-tokens", type=int, default=needles.DEFAULT_HAYSTACK_TOKENS)
    parser.add_argument("--log", type=Path, default=benchlog.DEFAULT_LOG)
    args = parser.parse_args(argv)

    if benchlog.environment(args.log)["git_dirty"]:
        parser.error("tracked files have uncommitted changes; commit first, so that "
                     "each entry names the code that produced it")

    engine = Engine(REPO / "models" / args.model)
    engine.load_weights()
    drawn = needles.needle_set(held_out_text(), engine.tokenizer, seed=args.needle_seed,
                               samples_per_depth=args.draws,
                               haystack_tokens=args.haystack_tokens)
    start = time.perf_counter()

    def is_right(sample: needles.Sample) -> bool:
        results = rq3.run(engine, sample, FP16, EXPERIMENT)
        benchlog.append(
            "rq3-needle", model=args.model, context_length=len(sample.turn_1),
            precision_tiers=results["tier_shares"],
            config=rq3.config(sample, FP16, EXPERIMENT, needle_seed=args.needle_seed,
                              plan_seed=0, haystack_tokens=args.haystack_tokens,
                              screening=True, issue=args.issue),
            results=results, log=args.log)
        print(f"{sample.sample_id}: {'right' if results['correct'] else 'WRONG'} "
              f"{results['answer']!r}  ({time.perf_counter() - start:.0f} s)", flush=True)
        return results["correct"]

    kept, rejected = rq3.screen(drawn, is_right, per_depth=args.per_depth)
    benchlog.append(
        "needle-screening", model=args.model,
        context_length=sorted(len(s.turn_1) for s in kept)[len(kept) // 2],  # the median
        precision_tiers={"FP16": 1.0},
        config={"needle_seed": args.needle_seed, "per_depth": args.per_depth,
                "draws": args.draws, "haystack_tokens": args.haystack_tokens,
                "condition": FP16.name, "experiment": EXPERIMENT, "issue": args.issue},
        results={"kept": [s.sample_id for s in kept],
                 "rejected": [s.sample_id for s in rejected],
                 "samples_per_depth": args.draws, "seconds": time.perf_counter() - start},
        log=args.log)
    print(f"kept {len(kept)}; replaced {len(rejected)}: {[s.sample_id for s in rejected]}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
