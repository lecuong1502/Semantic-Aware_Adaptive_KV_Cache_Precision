#!/usr/bin/env python3
"""A prefill step's fixed part and part per position, and decode's rate, at long contexts (#204).

    .venv/bin/python tools/step_cost.py --issue 204
    .venv/bin/python tools/step_cost.py --issue 204 --contexts 16384 32768 --kv-tier INT4

Under pressure an adaptive engine sizes each prefill chunk from what the
steps before it cost, a fixed part and a part per position (ADR-0016,
amendment of #135). At long contexts the fixed part is most of a step, so it
sets how quickly the engine can react while it prefills. This measures both
parts as the engine meets them: a conversation is forced to each context in
the engine's chunks, and then steps of --small and --large positions are
forced, --repeat of each, alternating, each one step. With t(n) the median
time of a step of n positions, the part per position is
(t(large) - t(small)) / (large - small), and the fixed part is what t(small)
leaves. The context grows by the steps as they go, a few hundred positions
against tens of thousands; a context the model's window leaves no room for
them at is prefilled short by that room, and logged as prefilled.

Then --decode-steps positions are forced one by one, a decode step's work
each but the greedy choice, and timed one at a time: decode's rate at that
context, from one prefill where tools/throughput.py makes two.

Appends one "prefill-step-cost" and one "decode-step" entry per context.
Refuses a dirty tree.
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from microinfer import Engine, benchlog  # noqa: E402

SEED = 204


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--issue", type=int, required=True,
                        help="the ticket this measurement is made for")
    parser.add_argument("--model", default="qwen2.5-1.5b-instruct")
    parser.add_argument("--kv-tier", default="FP16", choices=Engine.KV_TIERS)
    parser.add_argument("--contexts", nargs="+", type=int, default=[16384, 32768])
    parser.add_argument("--small", type=int, default=16)
    parser.add_argument("--large", type=int, default=64)
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--decode-steps", type=int, default=16)
    parser.add_argument("--log", type=Path, default=benchlog.DEFAULT_LOG)
    args = parser.parse_args(argv)

    if benchlog.environment(args.log)["git_dirty"]:
        parser.error("tracked files have uncommitted changes; commit first, so that "
                     "each entry names the code that produced it")

    engine = Engine(REPO / "models" / args.model, kv_tier=args.kv_tier)
    engine.load_weights()
    rng = np.random.default_rng(SEED)
    vocab = engine.config.vocab_size
    # What the steps add after each context, which must stay in the window.
    room = args.repeat * (args.small + args.large) + args.decode_steps
    window = engine.config.max_position_embeddings
    with engine.conversation() as conversation:
        for asked in sorted(args.contexts):
            context = min(asked, window - room)
            if context < conversation.positions:
                parser.error(f"{asked} positions and the steps after them do not fit "
                             f"the window of {window} after the context before")
            start = time.perf_counter()
            conversation.force(rng.integers(0, vocab, context - conversation.positions,
                                            dtype=np.int32))
            prefilled = time.perf_counter() - start
            times: dict[int, list[float]] = {args.small: [], args.large: []}
            for r in range(args.repeat):
                for n in ((args.small, args.large) if r % 2 == 0 else (args.large, args.small)):
                    ids = rng.integers(0, vocab, n, dtype=np.int32)
                    start = time.perf_counter()
                    conversation.force(ids)
                    times[n].append(time.perf_counter() - start)
            small, large = (statistics.median(times[n]) for n in (args.small, args.large))
            per_position = (large - small) / (args.large - args.small)
            fixed = small - args.small * per_position
            print(f"{context}: fixed {fixed * 1e3:.1f} ms, {per_position * 1e3:.2f} ms a "
                  f"position; steps of {args.small} {small * 1e3:.1f} ms, of {args.large} "
                  f"{large * 1e3:.1f} ms (prefilled in {prefilled:.0f} s)")
            benchlog.append(
                "prefill-step-cost", model=args.model, context_length=context,
                precision_tiers={args.kv_tier: 1.0},
                config={"issue": args.issue, "asked_context": asked,
                        "small": args.small, "large": args.large,
                        "repeat": args.repeat, "statistic": "median",
                        "prompt": "random token ids", "seed": SEED,
                        "prefill_chunk": engine.prefill_chunk,
                        "model": "t(n) = fixed + n * per_position"},
                results={"fixed_seconds": fixed, "per_position_seconds": per_position,
                         "step_seconds": {str(n): times[n] for n in times},
                         "prefill_seconds": prefilled},
                log=args.log)
            steps = []
            for _ in range(args.decode_steps):
                ids = rng.integers(0, vocab, 1, dtype=np.int32)
                start = time.perf_counter()
                conversation.force(ids, one_by_one=True)
                steps.append(time.perf_counter() - start)
            step = statistics.median(steps)
            print(f"{context}: decode step {step * 1e3:.1f} ms, {1 / step:.2f} tokens/s")
            benchlog.append(
                "decode-step", model=args.model, context_length=context,
                precision_tiers={args.kv_tier: 1.0},
                config={"issue": args.issue, "asked_context": asked,
                        "steps": args.decode_steps,
                        "statistic": "median", "prompt": "random token ids", "seed": SEED,
                        "step": "one position forced, as decode runs it, without its "
                                "greedy choice"},
                results={"step_seconds": step, "tokens_per_second": 1 / step,
                         "steps_seconds": steps},
                log=args.log)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
