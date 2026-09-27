#!/usr/bin/env python3
"""Hold the engine in a generation until told to stop, or until it runs out of
memory (#51, #52).

    .venv/bin/python tools/hold_engine.py --status hold.json
        [--model qwen2.5-1.5b-instruct] [--context 32768] [--decode-positions 1024]

RQ1's with-engine runs: the engine loads the model, prefills its window less
--decode-positions at FP16, then decodes continuously, going back to the end
of the prompt each time it reaches the end of the window (Engine.hold), so
that from the first pass on it holds the full context. It stops on SIGINT or
SIGTERM within one prefill chunk or decoding step.

If contention takes the memory the hold needs, the allocation that fails is
caught, and the failure recorded in the status file: RQ1's with-engine runs
exist to observe it. The tool then releases what it holds and exits with
code 3, the outcome observed rather than a fault of the tool.

How rarely that can happen is itself a finding. The paged cache takes device
memory only as positions arrive, a 2 MiB granule at a time, and a pass goes
back over pages already held: once the first pass has reached the end of
the window, a hold allocates nothing but a few bytes of token ids a step.
Contention that arrives after that can only fail the engine at those
allocations, and it is the other processes that meet it instead. A hold
whose cache is still growing, in the first pass or with a larger
--decode-positions, allocates a granule each time the pages of every layer
fill one: on Qwen2.5-1.5B a 32-position page row is 896 KiB, so every 73
positions, about every 110 s at 32K.

The status file is rewritten, whole, four times a second:

    {"state": "loading" | "prefilling" | "decoding" | "stopped" |
       "out_of_memory" | "failed",
     "position": the position the last token was decoded at, which goes
       back to the end of the prompt with every pass,
     "held_positions": the most positions the cache has held, which it
       still holds,
     "tokens_per_second": over the last second, or the last two updates if
       they are further apart, falling towards 0 while none arrives,
     "tokens": prefilled or decoded so far in this state,
     "pid", "model", "context", "prompt_positions", "t_mono_ns", "t_wall",
     "error": what failed, with "failed",
     "failure": with "out_of_memory", {"t_mono_ns", "t_wall": when it was
       caught; "phase": prefilling or decoding; "position" and
       "held_positions" then; "headroom_bytes": the device's free memory at
       the failure, as the allocator read it before giving anything back,
       or by NVML when caught if the allocation does not say
       ("headroom_at": "failure" or "caught"); "allocation": the allocation
       that failed, as the extension names it; "held_after_release_bytes":
       what the process still held on the device once the hold had
       released its cache and weights, by NVML: the CUDA context}}

A reader, the recorder or the scenario driver, knows the engine is ready when
the state is "decoding", and that the file is stale if t_mono_ns stops
moving. "stopped" and "out_of_memory" mean the hold ended and its cache and
weights are released; the CUDA context, about 90 MiB, goes when the process
exits, so a reader that needs every byte back waits for the pid to leave
NVML. "failed" promises nothing about memory. The prompt is random token ids: what is held
and how fast it decodes depend on length, not content.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import re
import signal
import sys
import threading
import time
from collections import deque
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from microinfer import Engine, OutOfMemory, nvml  # noqa: E402
from microinfer.engine import DECODING, PREFILLING  # noqa: E402

SEED = 51
WRITE_SECONDS = 0.25
LOADING, STOPPED, FAILED = "loading", "stopped", "failed"
OUT_OF_MEMORY = "out_of_memory"
#: The exit code of a hold that ran out of memory.
EXIT_OUT_OF_MEMORY = 3
SECOND_NS = 1_000_000_000


class Status:
    """The hold's state, written to a file by a thread of its own, so that a
    long prefill chunk or a slow step does not hold it back."""

    def __init__(self, path: Path, fields: dict):
        self._path, self._fields = path, fields
        self._lock = threading.Lock()
        self._state, self._position, self._held, self._tokens = LOADING, 0, 0, 0
        self._extra: dict = {}
        self._recent: deque[tuple[int, int]] = deque()  # (t_mono_ns, tokens), the last second
        self._done = threading.Event()
        self._thread = threading.Thread(target=self._run, name="hold-status", daemon=True)
        self._write()
        self._thread.start()

    def update(self, state: str, position: int) -> None:
        now = time.monotonic_ns()
        with self._lock:
            if state != self._state:
                self._state, self._tokens = state, 0
                self._recent.clear()
            self._tokens += 1 if state == DECODING else position - self._position
            self._position, self._held = position, max(self._held, position)
            self._recent.append((now, self._tokens))
            # The last second, but never less than the update before this one:
            # a prefill chunk can take longer than a second.
            while len(self._recent) > 2 and self._recent[1][0] <= now - SECOND_NS:
                self._recent.popleft()

    def record_failure(self, allocation: str) -> dict:
        """What running out looked like, taken the moment it was caught. The
        headroom is the allocator's reading at the failure where the message
        carries one, since giving back what the failed call took, and any
        other process's next allocation, move it before it can be caught."""
        at_failure = re.search(r"(\d+) bytes were free", allocation)
        with self._lock:
            return {"t_mono_ns": time.monotonic_ns(), "t_wall": time.time(),
                    "phase": self._state, "position": self._position,
                    "held_positions": self._held,
                    "headroom_bytes": (int(at_failure.group(1)) if at_failure
                                       else nvml.memory().free),
                    "headroom_at": "failure" if at_failure else "caught",
                    "allocation": allocation}

    def close(self, state: str, *, error: str | None = None,
              failure: dict | None = None) -> None:
        """Stop the thread and write the final state: with the error of a
        hold that failed, or what running out looked like."""
        self._done.set()
        self._thread.join()
        with self._lock:
            self._state = state
            self._extra = {k: v for k, v in (("error", error), ("failure", failure))
                           if v is not None}
            self._recent.clear()
        self._write()

    def _rate(self, now: int) -> float:
        """Tokens per second over the updates kept. Once the next update is
        overdue, longer in coming than the updates took on average, the time
        since the first counts too, so a stalled decode reads as slowing."""
        if len(self._recent) < 2:
            return 0.0
        (t0, n0), (t1, n1) = self._recent[0], self._recent[-1]
        overdue = now - t1 > (t1 - t0) / (len(self._recent) - 1)
        return (n1 - n0) * 1e9 / ((now if overdue else t1) - t0)

    def _write(self) -> None:
        now = time.monotonic_ns()
        with self._lock:
            status = {"state": self._state, "position": self._position,
                      "held_positions": self._held, "tokens_per_second": self._rate(now),
                      "tokens": self._tokens, **self._fields, "t_mono_ns": now,
                      "t_wall": time.time()}
            status.update(self._extra)
        tmp = self._path.with_name(self._path.name + ".tmp")
        tmp.write_text(json.dumps(status) + "\n")
        os.replace(tmp, self._path)  # whole or not at all, for a reader at any moment

    def _run(self) -> None:
        while not self._done.wait(WRITE_SECONDS):
            self._write()


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--status", type=Path, required=True, help="the JSON status file")
    parser.add_argument("--model", default="qwen2.5-1.5b-instruct")
    parser.add_argument("--context", type=int, default=None,
                        help="positions held; the model's window by default")
    parser.add_argument("--decode-positions", type=int, default=1024,
                        help="the span decoded again and again after the prompt")
    args = parser.parse_args(argv)

    stop = threading.Event()
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: stop.set())

    engine = Engine(REPO / "models" / args.model, kv_tier="FP16")
    context = args.context or engine.config.max_position_embeddings
    prompt_positions = context - args.decode_positions
    if not 0 < prompt_positions < context:
        parser.error(f"--decode-positions must be between 1 and the context, {context}, less one")
    status = Status(args.status, {"pid": os.getpid(), "model": args.model, "context": context,
                                  "prompt_positions": prompt_positions})
    announced = False

    def report(state: str, position: int, token: int | None) -> None:
        nonlocal announced
        status.update(state, position)
        if state == DECODING and not announced:
            announced = True
            print(f"ready: prefilled {prompt_positions} positions, decoding", flush=True)

    failure = None
    try:
        engine.load_weights()
        rng = np.random.default_rng(SEED)
        prompt = rng.integers(1000, engine.config.vocab_size - 1000,
                              prompt_positions).astype(np.int32)
        if not stop.is_set():
            engine.hold(prompt, context=context, stop=stop, report=report)
    except OutOfMemory as exc:
        failure = status.record_failure(str(exc))
        # Leaving this block drops the exception, and with its traceback the
        # frames that still hold the cache.
    except Exception as exc:
        status.close(FAILED, error=f"{type(exc).__name__}: {exc}")
        raise
    # The cache went with the hold; the weights go with the engine.
    del engine
    gc.collect()
    if failure is not None:
        failure["held_after_release_bytes"] = nvml.own_used_bytes()
        status.close(OUT_OF_MEMORY, failure=failure)
        print(f"out of memory at position {failure['position']}: {failure['allocation']}",
              flush=True)
        return EXIT_OUT_OF_MEMORY
    status.close(STOPPED)
    print("stopped", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
