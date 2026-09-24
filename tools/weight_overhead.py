#!/usr/bin/env python3
"""What loading the weights takes from the driver, against what they hold (#23).

    .venv/bin/python tools/weight_overhead.py --issue 23 [--models qwen2.5-0.5b-instruct ...]

For each model, the weights are loaded into the engine's one arena and three
figures are recorded: the bytes the tensors hold (Footprint.weights), the
arena's size (weight_layout, from config.json, with every tensor aligned),
and what the driver took for this process across the load
(nvml.own_used_bytes), which no other process moves. Then the engine is
dropped and what this process holds afterwards is recorded too: the whole
arena should be back. One entry per model. Refuses a dirty tree.
"""

from __future__ import annotations

import argparse
import gc
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from microinfer import Engine, benchlog, nvml  # noqa: E402

MIB = 2**20


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--issue", type=int, required=True,
                        help="the ticket this measurement is made for")
    parser.add_argument("--models", nargs="+",
                        default=["qwen2.5-0.5b-instruct", "qwen2.5-1.5b-instruct"])
    parser.add_argument("--log", type=Path, default=benchlog.DEFAULT_LOG)
    args = parser.parse_args(argv)

    if benchlog.environment(args.log)["git_dirty"]:
        parser.error("tracked files have uncommitted changes; commit first, so that "
                     "each entry names the code that produced it")

    for name in args.models:
        engine = Engine(REPO / "models" / name)
        gc.collect()
        before = nvml.own_used_bytes()
        engine.load_weights()
        taken = nvml.own_used_bytes() - before
        claimed = engine.footprint().weights
        arena = engine.weight_arena.nbytes
        tensors = len(engine.tensors)
        del engine
        gc.collect()
        left = nvml.own_used_bytes() - before
        results = {"tensors": tensors, "allocations": 1, "claimed_bytes": claimed,
                   "arena_bytes": arena, "taken_bytes": taken,
                   "overhead": taken / claimed - 1, "held_after_free_bytes": left}
        print(f"{name}: {tensors} tensors, claimed {claimed / MIB:.1f} MiB, arena "
              f"{arena / MIB:.1f} MiB, taken {taken / MIB:.1f} MiB "
              f"({100 * results['overhead']:.2f}% over), {left / MIB:.1f} MiB left after free")
        benchlog.append(
            "weight-allocation", model=name, context_length=None, precision_tiers=None,
            config={"measured": "nvml.own_used_bytes across load_weights, and after the "
                                "engine is dropped", "layout": "weight_layout, one arena",
                    "issue": args.issue},
            results=results, log=args.log)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
