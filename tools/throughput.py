#!/usr/bin/env python3
"""Prefill and decode throughput, measured and logged separately (#15).

    .venv/bin/python tools/throughput.py --model qwen2.5-0.5b-instruct --lengths 512 2048 8192
    .venv/bin/python tools/throughput.py --model qwen2.5-1.5b-instruct --lengths 32768 --repeat 1
    .venv/bin/python tools/throughput.py --kv-tier INT4 ...
    .venv/bin/python tools/throughput.py --monitor --lengths 512 --repeat 10

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

--monitor measures what the VRAM pressure monitor costs decoding (#62)
instead: decode throughput with the engine's monitor running and without,
--repeat times each, interleaved and in alternating order so that drift
falls on both alike. It logs one "monitor-overhead" entry per length, with
both distributions and whether the difference of their means is within
noise: no more than twice its standard error, sqrt(s_on^2/n + s_off^2/n).
A transition also reads the own/others split, which the timed runs may never
see, so the entry carries that read's own cost, timed apart.
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

from microinfer import Engine, benchlog, monitor  # noqa: E402
from microinfer.contention import spread  # noqa: E402

SEED = 15
DECODE_STEPS = 32
#: How many times the monitor's own/others split is read to time one.
SPLIT_READS = 50


def timed(fn, repeat: int, warmup: int) -> float:
    """Median wall time of `repeat` calls, after `warmup` untimed ones:
    cuBLAS loads kernels for a new GEMM shape on first use. A 32K-token
    prefill takes minutes, against milliseconds of loading, so long lengths
    are measured without one."""
    for _ in range(warmup):
        fn()
    return statistics.median(seconds(fn) for _ in range(repeat))


def seconds(fn) -> float:
    """The wall time of one call."""
    start = time.perf_counter()
    fn()
    return time.perf_counter() - start


def first_token(engine: Engine, prompt: np.ndarray):
    """Prefill `prompt` and choose one token: what a decode timing subtracts."""
    return lambda: engine.generate(prompt, 1, stop_at_eos=False)


def with_steps(engine: Engine, prompt: np.ndarray):
    """The same, and DECODE_STEPS more tokens decoded."""
    return lambda: engine.generate(prompt, DECODE_STEPS + 1, stop_at_eos=False)


def decode_prompt(engine: Engine, ids: np.ndarray) -> np.ndarray:
    """`ids`, shortened if it must be so that DECODE_STEPS more stay inside
    the model's context window."""
    return ids[: min(len(ids), engine.config.max_position_embeddings - DECODE_STEPS)]


def decode_rate(engine: Engine, prompt: np.ndarray) -> float:
    """Decode tokens per second after `prompt`, from one timing of each."""
    steps = seconds(with_steps(engine, prompt)) - seconds(first_token(engine, prompt))
    return DECODE_STEPS / max(steps, 1e-9)


def monitor_overhead(engine: Engine, prompt: np.ndarray, repeat: int, warmup: int) -> dict:
    """Decode throughput with the pressure monitor on and off, `repeat`
    times each, interleaved, the order alternating."""
    for _ in range(warmup):
        decode_rate(engine, prompt)
    rates: dict[bool, list[float]] = {False: [], True: []}
    events, errors = 0, []
    for r in range(repeat):
        for on in ((False, True) if r % 2 == 0 else (True, False)):
            if on:
                engine.start_monitor()
            try:
                rates[on].append(decode_rate(engine, prompt))
            finally:
                engine.stop_monitor()
            if on:
                events += len(engine.pressure_events)
                if engine.monitor_error is not None:
                    errors.append(str(engine.monitor_error))

    def summary(values: list[float]) -> dict:
        return {"runs": values, "mean": statistics.mean(values),
                "stdev": statistics.stdev(values) if len(values) > 1 else 0.0}

    off, on = summary(rates[False]), summary(rates[True])
    noise = 2 * (off["stdev"] ** 2 / repeat + on["stdev"] ** 2 / repeat) ** 0.5
    difference = on["mean"] - off["mean"]
    return {"steps": DECODE_STEPS, "off": off, "on": on, "difference": difference,
            "difference_percent": 100 * difference / off["mean"], "noise": noise,
            "within_noise": abs(difference) <= noise,
            "events_recorded": events, "monitor_errors": errors,
            # A transition also reads the own/others split: rare, and timed
            # here, since the runs above may see none.
            "split_read_ms": spread([seconds(monitor.nvml_split) * 1e3
                                     for _ in range(SPLIT_READS)])}


def log_monitor_overhead(engine: Engine, ids: np.ndarray, args, method: dict,
                         tiers: dict) -> None:
    """Measure and log what the pressure monitor costs decoding after `ids`."""
    prompt = decode_prompt(engine, ids)
    results = monitor_overhead(engine, prompt, args.repeat, args.warmup)
    thresholds = monitor.DEFAULT
    config = {**method, "statistic": "mean", "steps": DECODE_STEPS,
              "order": "interleaved, alternating", "poll_s": monitor.POLL_S,
              "thresholds": {"red_below_bytes": thresholds.red_below_bytes,
                             "yellow_below_bytes": thresholds.yellow_below_bytes,
                             "persist_polls": thresholds.persist_polls,
                             "thresholds_adr": 13}}
    print(f"decode after {len(prompt):6}: {results['off']['mean']:8.1f} tok/s without "
          f"the monitor, {results['on']['mean']:8.1f} with "
          f"({results['difference_percent']:+.2f}%, noise +-{results['noise']:.1f}): "
          f"{'within' if results['within_noise'] else 'OUTSIDE'} noise; a split read "
          f"{results['split_read_ms']['median']:.2f} ms")
    benchlog.append("monitor-overhead", model=args.model,
                    context_length=len(prompt) + DECODE_STEPS, precision_tiers=tiers,
                    config=config, results=results, log=args.log)


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", default="qwen2.5-0.5b-instruct")
    parser.add_argument("--lengths", type=int, nargs="+", default=[512, 2048, 8192])
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--kv-tier", default="FP16", choices=Engine.KV_TIERS,
                        help="the static tier the cache is held at (#18)")
    parser.add_argument("--prefill-only", action="store_true",
                        help="log prefill alone; at 32K each decode timing costs two "
                             "more prefills")
    parser.add_argument("--monitor", action="store_true",
                        help="measure the pressure monitor's cost to decoding, instead")
    parser.add_argument("--log", type=Path, default=benchlog.DEFAULT_LOG)
    args = parser.parse_args(argv)
    if args.monitor and args.repeat < 2:
        parser.error("--monitor compares spreads, and needs --repeat 2 or more")

    if benchlog.environment(args.log)["git_dirty"]:
        parser.error("tracked files have uncommitted changes; commit first")

    engine = Engine(REPO / "models" / args.model, kv_tier=args.kv_tier)
    engine.load_weights()
    rng = np.random.default_rng(SEED)
    method = {"repeat": args.repeat, "warmup": args.warmup, "statistic": "median", "seed": SEED,
              "prompt": "random token ids", "prefill_chunk": engine.prefill_chunk,
              "kv_cache": engine.kv_cache, "kv_tier": engine.kv_tier}
    tiers = {engine.kv_tier: 1.0}

    for length in args.lengths:
        ids = rng.integers(1000, engine.config.vocab_size - 1000, length).astype(np.int32)

        if args.monitor:
            log_monitor_overhead(engine, ids, args, method, tiers)
            continue

        # Prefill, and the one greedy choice after it: no logits for every
        # position, which at 32K tokens would be 20 GB of host memory.
        engine.reset_peak()
        prefill_seconds = timed(first_token(engine, ids), args.repeat, args.warmup)
        peak = engine.peak_footprint()
        prefill = {"tokens": length, "seconds": prefill_seconds,
                   "tokens_per_second": length / prefill_seconds,
                   "peak": {"weights": peak.weights, "kv_cache": peak.kv_cache,
                            "workspace": peak.workspace, "engine_total": peak.engine_total,
                            "unaccounted": peak.unaccounted, "device_free": peak.device_free,
                            "device_total": peak.device_total}}
        print(f"prefill {length:6} tokens: {prefill_seconds:8.2f} s, "
              f"{length / prefill_seconds:9.1f} tok/s, "
              f"free at peak {peak.device_free / 2**20:.0f} MiB")
        benchlog.append("prefill-throughput", model=args.model, context_length=length,
                        precision_tiers=tiers, config=method, results=prefill, log=args.log)

        if args.prefill_only:
            continue

        # Decode: generation of one token (prefill and its greedy choice)
        # and of DECODE_STEPS more are both timed; the difference is the
        # steps alone. The prompt is shortened, if it must be, so that the
        # steps stay inside the model's context window.
        prompt = decode_prompt(engine, ids)
        prefill_only = prefill_seconds if len(prompt) == length else timed(
            first_token(engine, prompt), args.repeat, args.warmup)
        steps_seconds = max(timed(with_steps(engine, prompt), args.repeat, args.warmup)
                            - prefill_only, 1e-9)
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
