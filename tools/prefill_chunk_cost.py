#!/usr/bin/env python3
"""What a prefill chunk costs at long contexts, on the tensor cores and before (#213).

    .venv/bin/python tools/prefill_chunk_cost.py --issue 213
        [--kv-tiers FP16 INT8 INT4 INT2 mixed] [--contexts 8192 16384 32768]
        [--chunk 512] [--repeat 3]

A pressure event during prefill waits for the chunk in progress, so a chunk's
time is how late the first event of an episode is met (ADR-0016). This
measures it in the engine, as prefill meets it: a conversation is forced to
each context in the engine's chunks, and then chunks of --chunk positions
are forced, --repeat by each of the two kernels, alternating: prefill's on
the tensor cores, and the one that computed it before, which a decode step
still runs (device.prefill_on_tensor_cores). The context grows by the chunks
as they go; one the model's window leaves no room for them at is prefilled
short by that room, and logged as prefilled.

`mixed` is a cache born from a map that cycles the four tiers page by page,
each layer's row offset by one.

Appends one "prefill-chunk-cost" entry per tier. Refuses a dirty tree.
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
sys.path.insert(0, str(REPO / "tools"))

from decode_agreement import mixed_map  # noqa: E402

from microinfer import Engine, _microinfer, benchlog  # noqa: E402

TIERS = ("FP16", "INT8", "INT4", "INT2", "mixed")
#: The two kernels, and whether each is on the tensor cores.
KERNELS = {"before": False, "tensor_cores": True}
SEED = 213
on_tensor_cores = _microinfer.device.prefill_on_tensor_cores


def measure(engine: Engine, args: argparse.Namespace) -> dict[str, dict]:
    """Each context's chunk times by kernel."""
    rng = np.random.default_rng(SEED)
    vocab = engine.config.vocab_size
    # What the chunks add after each context, which must stay in the window.
    room = args.repeat * len(KERNELS) * args.chunk
    results = {}
    with engine.conversation() as conversation:
        for asked in sorted(args.contexts):
            context = min(asked, engine.config.context_window - room)
            if context < conversation.positions:
                raise ValueError(f"{asked} positions and the chunks after them do not fit the "
                                 f"window after the context before")
            conversation.force(rng.integers(0, vocab, context - conversation.positions,
                                            dtype=np.int32))
            seconds: dict[str, list[float]] = {kernel: [] for kernel in KERNELS}
            for r in range(args.repeat):
                for kernel in list(KERNELS)[::1 if r % 2 == 0 else -1]:
                    ids = rng.integers(0, vocab, args.chunk, dtype=np.int32)
                    asked_before = on_tensor_cores(KERNELS[kernel])
                    try:
                        start = time.perf_counter()
                        conversation.force(ids)
                        seconds[kernel].append(time.perf_counter() - start)
                    finally:
                        on_tensor_cores(asked_before)
            medians = {k: statistics.median(s) for k, s in seconds.items()}
            results[str(asked)] = {"prefilled": context, "chunk_seconds": seconds,
                                   "median_seconds": medians,
                                   "speedup": medians["before"] / medians["tensor_cores"]}
            print(f"{asked} (from {context}): before {medians['before'] * 1e3:.0f} ms, tensor "
                  f"cores {medians['tensor_cores'] * 1e3:.0f} ms a chunk of {args.chunk}",
                  flush=True)
    return results


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--issue", type=int, required=True,
                        help="the ticket this measurement is made for")
    parser.add_argument("--model", default="qwen2.5-1.5b-instruct")
    parser.add_argument("--kv-tiers", nargs="+", default=list(TIERS), choices=TIERS)
    parser.add_argument("--contexts", nargs="+", type=int, default=[8192, 16384, 32768])
    parser.add_argument("--chunk", type=int, default=512)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--log", type=Path, default=benchlog.DEFAULT_LOG)
    args = parser.parse_args(argv)
    if benchlog.environment(args.log)["git_dirty"]:
        parser.error("tracked files have uncommitted changes; commit first, so that "
                     "each entry names the code that produced it")
    engine = Engine(REPO / "models" / args.model)
    if args.chunk > engine.prefill_chunk:
        parser.error(f"a chunk is at most the engine's, {engine.prefill_chunk}")
    engine.load_weights()
    pages = engine.config.context_window // _microinfer.device.page_tokens + 1
    for tier in args.kv_tiers:
        mixed = tier == "mixed"
        engine.kv_tier = "FP16" if mixed else tier
        engine.kv_tier_map = mixed_map(engine.config.num_hidden_layers, pages) if mixed else None
        print(tier, flush=True)
        results = measure(engine, args)
        benchlog.append(
            "prefill-chunk-cost", model=args.model,
            context_length={"min": min(args.contexts), "max": max(args.contexts)},
            precision_tiers={t: 0.25 for t in TIERS[:4]} if mixed else {tier: 1.0},
            config={"issue": args.issue, "contexts": args.contexts, "chunk": args.chunk,
                    "repeat": args.repeat, "statistic": "median", "alternating": True,
                    "prompt": "random token ids", "seed": SEED,
                    "tier_map": "four tiers cycled by page" if mixed else None,
                    "kernels": {"before": "device.prefill_on_tensor_cores(False)",
                                "tensor_cores": "device.prefill_on_tensor_cores(True)"}},
            results=results, log=args.log)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
