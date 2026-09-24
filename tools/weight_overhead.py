#!/usr/bin/env python3
"""What loading the weights takes from the driver, one arena against one
allocation per tensor (#23).

    .venv/bin/python tools/weight_overhead.py --issue 23 [--models qwen2.5-0.5b-instruct ...]

For each model the checkpoint is put on the device twice, and what the driver
took for this process each time is read (nvml.own_used_bytes), which no
other process moves, from a settled start, with the CUDA context already
made (nvml.settled_own_used_bytes):

- one allocation per tensor, as before #23: each tensor uploaded on its own;
- one arena, as Engine.load_weights does it now.

Both are set against the bytes the tensors hold. The same instrument reads
both, so the difference between them is the arena's saving, and nothing
else. After each, everything is dropped, and what this process still holds
beyond its start is recorded: the whole of it should be back. One entry per
model. Refuses a dirty tree.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from microinfer import Engine, _microinfer, benchlog, nvml, weights  # noqa: E402

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
        path = REPO / "models" / name / "model.safetensors"

        before = nvml.settled_own_used_bytes()
        separate = [_microinfer.upload_fp16(np.ascontiguousarray(v, np.float32).reshape(-1))
                    for _, v in weights.iter_tensors(path)]
        per_tensor = nvml.own_used_bytes() - before
        del separate
        per_tensor_left = nvml.settled_own_used_bytes() - before

        engine = Engine(REPO / "models" / name)
        before = nvml.settled_own_used_bytes()
        engine.load_weights()
        taken = nvml.own_used_bytes() - before
        claimed = engine.footprint().weights
        arena = engine.weight_arena.nbytes
        tensors = len(engine.tensors)
        del engine
        left = nvml.settled_own_used_bytes() - before

        results = {"tensors": tensors, "claimed_bytes": claimed, "arena_bytes": arena,
                   "arena_taken_bytes": taken, "arena_overhead": taken / claimed - 1,
                   "arena_left_after_free_bytes": left,
                   "per_tensor_taken_bytes": per_tensor,
                   "per_tensor_overhead": per_tensor / claimed - 1,
                   "per_tensor_left_after_free_bytes": per_tensor_left,
                   "saved_bytes": per_tensor - taken}
        print(f"{name}: {tensors} tensors, {claimed / MIB:.1f} MiB claimed. One per tensor: "
              f"{per_tensor / MIB:.1f} MiB taken ({100 * results['per_tensor_overhead']:.2f}% "
              f"over). One arena: {taken / MIB:.1f} MiB ({100 * results['arena_overhead']:.2f}% "
              f"over). Saved {results['saved_bytes'] / MIB:.1f} MiB. Left after freeing: "
              f"{per_tensor_left / MIB:.1f} and {left / MIB:.1f} MiB")
        benchlog.append(
            "weight-allocation", model=name, context_length=None, precision_tiers=None,
            config={"measured": "nvml.own_used_bytes from nvml.settled_own_used_bytes, "
                                "across each way of loading and after dropping it",
                    "per_tensor": "upload_fp16 per tensor, as before #23",
                    "arena": "Engine.load_weights, weight_layout", "issue": args.issue},
            results=results, log=args.log)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
