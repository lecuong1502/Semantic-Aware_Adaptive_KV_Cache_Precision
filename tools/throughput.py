#!/usr/bin/env python3
"""Prefill and decode throughput, measured and logged separately (#15).

    .venv/bin/python tools/throughput.py --model qwen2.5-0.5b-instruct --lengths 512 2048 8192
    .venv/bin/python tools/throughput.py --model qwen2.5-1.5b-instruct --lengths 32768 --repeat 1

They are two different regimes, so they are two entries per length:
- **Prefill** runs many positions through each projection at once and is
  bound by compute. It is reported as prompt tokens per second, for a prompt
  of each length, prefilled in chunks of the engine's prefill_chunk, and
  timed with the one greedy choice after it.
- **Decode** runs one position per step and reads every weight each time,
  so it is bound by memory bandwidth. It is reported as generated tokens per
  second, after a prompt of each length, since each step's attention reads
  the whole cache.

Each prefill entry also records the peak footprint, which is how a 32K-token
prompt on the 1.5B model is shown to fit on a 6 GiB card. Prompts are
random token ids: throughput depends on length, not content. Refuses a dirty
tree, as the checkpoint recorder does.
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

SEED = 15
DECODE_STEPS = 32


def timed(fn, repeat: int, warmup: int) -> float:
    """Median wall time of `repeat` calls, after `warmup` untimed ones:
    cuBLAS loads kernels for a new GEMM shape on first use. A 32K-token
    prefill takes minutes, against milliseconds of loading, so long lengths
    are measured without one."""
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(repeat):
        start = time.perf_counter()
        fn()
        times.append(time.perf_counter() - start)
    return statistics.median(times)


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", default="qwen2.5-0.5b-instruct")
    parser.add_argument("--lengths", type=int, nargs="+", default=[512, 2048, 8192])
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--log", type=Path, default=benchlog.DEFAULT_LOG)
    args = parser.parse_args(argv)

    if benchlog.environment(args.log)["git_dirty"]:
        parser.error("tracked files have uncommitted changes; commit first")

    engine = Engine(REPO / "models" / args.model)
    engine.load_weights()
    rng = np.random.default_rng(SEED)
    method = {"repeat": args.repeat, "warmup": args.warmup, "statistic": "median", "seed": SEED,
              "prompt": "random token ids", "prefill_chunk": engine.prefill_chunk,
              "kv_cache": engine.kv_cache}
    tiers = {"FP16": 1.0}

    for length in args.lengths:
        ids = rng.integers(1000, engine.config.vocab_size - 1000, length).astype(np.int32)

        # Prefill, and the one greedy choice after it: no logits for every
        # position, which at 32K tokens would be 20 GB of host memory.
        engine.reset_peak()
        seconds = timed(lambda: engine.generate(ids, 1, stop_at_eos=False),
                        args.repeat, args.warmup)
        peak = engine.peak_footprint()
        prefill = {"tokens": length, "seconds": seconds, "tokens_per_second": length / seconds,
                   "peak": {"weights": peak.weights, "kv_cache": peak.kv_cache,
                            "workspace": peak.workspace, "engine_total": peak.engine_total,
                            "unaccounted": peak.unaccounted, "device_free": peak.device_free,
                            "device_total": peak.device_total}}
        print(f"prefill {length:6} tokens: {seconds:8.2f} s, {length / seconds:9.1f} tok/s, "
              f"free at peak {peak.device_free / 2**20:.0f} MiB")
        benchlog.append("prefill-throughput", model=args.model, context_length=length,
                        precision_tiers=tiers, config=method, results=prefill, log=args.log)

        # Decode: generation of one token (prefill and its greedy choice)
        # and of DECODE_STEPS more are both timed; the difference is the
        # steps alone. The prompt is shortened, if it must be, so that the
        # steps stay inside the model's context window.
        window = engine.config.max_position_embeddings
        prompt = ids[: min(length, window - DECODE_STEPS)]
        prefill_only = seconds if len(prompt) == length else timed(
            lambda: engine.generate(prompt, 1, stop_at_eos=False), args.repeat, args.warmup)
        with_steps = timed(lambda: engine.generate(prompt, DECODE_STEPS + 1, stop_at_eos=False),
                           args.repeat, args.warmup)
        steps_seconds = max(with_steps - prefill_only, 1e-9)
        decode = {"prompt_tokens": len(prompt), "steps": DECODE_STEPS, "seconds": steps_seconds,
                  "tokens_per_second": DECODE_STEPS / steps_seconds}
        print(f"decode  after {length:6}: {DECODE_STEPS / steps_seconds:9.1f} tok/s")
        benchlog.append("decode-throughput", model=args.model,
                        context_length=len(prompt) + DECODE_STEPS,
                        precision_tiers=tiers, config={**method, "steps": DECODE_STEPS},
                        results=decode, log=args.log)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
