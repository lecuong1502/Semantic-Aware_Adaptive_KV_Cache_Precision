"""Device-wide memory, sampled on a fixed schedule into a compressed CSV (#46).

RQ1's instrument: how free video memory moves while the desktop is used. The
recorder reads the driver's account of free and used memory (NVML) at a fixed
rate, 50 Hz by default, twice the rate the pressure monitor will poll at, so
that a spike the monitor misses can be seen in the trace.

**The schedule is absolute.** Sample k is due at start + k * period, however
long sample k - 1 took. A slow read delays only itself and the schedule does
not drift. A read that overruns whole periods skips their deadlines and
counts them as missed, rather than bunching samples to catch up.

**The file survives an interrupt.** It is gzip-compressed CSV, flushed at
least once a second, so a recorder killed without warning leaves a file that
reads back to within a second of the kill. A recording that ends cleanly
closes with a `# complete` line, and read() reports whether it is there.

**It takes no device memory.** NVML needs no CUDA context, so a process that
records is not a GPU process at all, and cannot be part of the contention it
records.

This is measurement, not inference, so it computes in NumPy on the host
(ADR-0002, amendment on its scope).
"""

from __future__ import annotations

import gzip
import threading
import time
from pathlib import Path
from typing import Callable

import numpy as np

from . import nvml

#: One row per sample: monotonic time, wall-clock time, the driver's free and
#: used bytes, and how long the query took.
COLUMNS = ("t_mono_ns", "t_wall", "free_bytes", "used_bytes", "query_ns")
_DTYPE = np.dtype([("t_mono_ns", np.int64), ("t_wall", np.float64), ("free_bytes", np.int64),
                   ("used_bytes", np.int64), ("query_ns", np.int64)])

FORMAT = "microinfer contention trace, device memory, v1"
FLUSH_SECONDS = 1.0

Reader = Callable[[], tuple[int, int]]


def nvml_reader() -> tuple[int, int]:
    """Free and used device memory, by the driver's account."""
    m = nvml.memory()
    return m.free, m.used


def device_meta() -> dict[str, object]:
    """What a recording says about where it was made."""
    return {"device": nvml.device_name(), "driver": nvml.driver_version(),
            "total_bytes": nvml.memory().total}


def record(path: str | Path | None, rate_hz: float = 50.0, *, duration: float | None = None,
           stop: threading.Event | None = None, reader: Reader | None = None,
           meta: dict | None = None) -> dict:
    """Sample `reader` every 1 / rate_hz seconds until `duration` has passed or
    `stop` is set, writing each sample to `path` if one is given. Returns the
    summary of what was recorded (see summarise)."""
    if rate_hz <= 0:
        raise ValueError(f"rate_hz must be positive, got {rate_hz}")
    if reader is None:
        reader = nvml_reader
        meta = {**device_meta(), **(meta or {})}
    period = 1e9 / rate_hz
    end = None if duration is None else int(duration * 1e9)
    rows: list[tuple] = []

    out = gzip.open(path, "wt", newline="") if path is not None else None
    try:
        if out is not None:
            out.write(f"# {FORMAT}\n")
            for key, value in {**(meta or {}), "rate_hz": f"{rate_hz:g}"}.items():
                out.write(f"# {key}={value}\n")
            out.write(",".join(COLUMNS) + "\n")
        start = time.monotonic_ns()
        last_flush = start
        k = missed = 0
        while end is None or k * period < end:
            wait = (start + k * period - time.monotonic_ns()) / 1e9
            if stop is not None:
                if stop.wait(max(wait, 0.0)):
                    break
            elif wait > 0:
                time.sleep(wait)
            t0 = time.monotonic_ns()
            wall = time.time()
            free, used = reader()
            t1 = time.monotonic_ns()
            row = (t0, wall, free, used, t1 - t0)
            rows.append(row)
            if out is not None:
                out.write(f"{t0},{wall:.6f},{free},{used},{t1 - t0}\n")
                if t1 - last_flush >= FLUSH_SECONDS * 1e9:
                    out.flush()
                    last_flush = t1
            # The next deadline not yet passed; any skipped are missed.
            due = int((t1 - start) // period) + 1
            missed += max(due - (k + 1), 0)
            k = max(k + 1, due)
        if out is not None:
            out.write(f"# complete samples={len(rows)} missed={missed}\n")
    finally:
        if out is not None:
            out.close()
    samples = np.array(rows, dtype=_DTYPE)
    return {**summarise(samples), "missed": missed, "rate_hz": rate_hz}


def read(path: str | Path) -> tuple[dict, np.ndarray]:
    """A recording's metadata and samples. A recording cut short, by a kill or
    a crash, reads back to its last flush, and its metadata says
    complete=False."""
    meta: dict[str, object] = {"complete": False}
    rows: list[tuple] = []
    try:
        with gzip.open(path, "rt", newline="") as f:
            for line in f:
                if not line.endswith("\n"):
                    break  # the last line was cut mid-write
                if line.startswith("# complete"):
                    meta["complete"] = True
                elif line.startswith("# ") and "=" in line:
                    key, value = line[2:].rstrip("\n").split("=", 1)
                    meta[key] = value
                elif line[0].isdigit():
                    a, b, c, d, e = line.rstrip("\n").split(",")
                    rows.append((int(a), float(b), int(c), int(d), int(e)))
    except (EOFError, OSError):
        pass  # the gzip stream ends without its trailer: the recorder was killed
    return meta, np.array(rows, dtype=_DTYPE)


def summarise(samples: np.ndarray) -> dict:
    """How the schedule was kept: the samples taken, the achieved rate, the
    spread of the periods between them, and what each query cost."""
    n = len(samples)
    if n < 2:
        return {"samples": n}
    periods = np.diff(samples["t_mono_ns"]) / 1e6
    query = samples["query_ns"] / 1e3
    span = (samples["t_mono_ns"][-1] - samples["t_mono_ns"][0]) / 1e9
    return {"samples": n, "seconds": span, "achieved_hz": (n - 1) / span,
            "period_ms": {"mean": float(periods.mean()), "std": float(periods.std()),
                          "p50": float(np.percentile(periods, 50)),
                          "p99": float(np.percentile(periods, 99)), "max": float(periods.max())},
            "query_us": {"median": float(np.median(query)),
                         "p99": float(np.percentile(query, 99)), "max": float(query.max())}}
