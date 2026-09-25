"""Device memory, and who holds it, sampled on fixed schedules into compressed
CSV (#46, #47).

RQ1's instrument: how free video memory moves while the desktop is used, and
which process moved it. The recorder keeps two streams, on one clock:

- **device memory**, the driver's free and used bytes (NVML), at 50 Hz by
  default: twice the rate the pressure monitor will poll at, so that a spike
  the monitor misses can be seen in the trace;
- **processes**, at 5 Hz by default, in a second file beside the first: the
  memory each GPU process holds, and the GPU's P-state and clocks. It is what
  attributes a spike to the process that caused it. A process query costs far
  more than a memory query, so it runs on its own thread rather than delaying
  the 50 Hz samples.

Both streams time their samples with time.monotonic_ns, one clock for the
whole machine, so a row of one lines up with the rows of the other by
timestamp alone.

**The schedules are absolute.** Sample k is due at start + k * period, however
long sample k - 1 took. A slow read delays only itself and the schedule does
not drift. A read that overruns whole periods skips their deadlines and
counts them as missed, rather than bunching samples to catch up.

**The files survive an interrupt.** Each is gzip-compressed CSV, flushed with
the first sample taken a second or more after the last flush. Samples arrive
every period, so a recorder killed without warning leaves files that read
back to within about a second of the kill. A recording that ends cleanly
closes each with a line of its own, `#! complete`, carrying its counts, and
the readers report whether it is there. Only a stream cut short reads as
incomplete: a missing file, or one that is not a trace, is an error.

**It takes no device memory, and gives the GPU no work.** NVML needs no CUDA
context, so a process that records is not a GPU process at all, and cannot
be part of the contention it records. NVML's queries are answered from the
driver's bookkeeping; they launch nothing on the device.

The device samples are also kept in host memory, for the summary: 40 bytes
each, about 14 MiB for a two-hour recording at 50 Hz.

This is measurement, not inference, so it computes in NumPy on the host
(ADR-0002, amendment on its scope).
"""

from __future__ import annotations

import csv
import gzip
import io
import math
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

import numpy as np

from . import nvml

#: One row per device sample: monotonic time, wall-clock time, the driver's
#: free and used bytes, and how long the query took.
_DTYPE = np.dtype([("t_mono_ns", np.int64), ("t_wall", np.float64), ("free_bytes", np.int64),
                   ("used_bytes", np.int64), ("query_ns", np.int64)])
COLUMNS = _DTYPE.names

#: The first line of every device trace; read() refuses a file without it.
FORMAT = "microinfer contention trace, device memory, v1"

#: The processes stream holds two kinds of row, told apart by their first
#: field. Each sample writes one state row, the GPU's P-state and clocks,
#: then one process row for every process on the GPU. A sample with no
#: process still leaves its state row, so no sample goes unrecorded. A value
#: the driver does not report is written as -1.
STATE_ROW, PROCESS_ROW = "state", "process"
PROCESS_FORMAT = "microinfer contention trace, processes, v1"
STATE_COLUMNS = ("t_mono_ns", "t_wall", "query_ns", "pstate", "graphics_mhz", "sm_mhz",
                 "memory_mhz")
PROCESS_COLUMNS = ("t_mono_ns", "t_wall", "pid", "kind", "used_bytes", "name")
_STATE_DTYPE = np.dtype([("t_mono_ns", np.int64), ("t_wall", np.float64), ("query_ns", np.int64),
                         ("pstate", np.int32), ("graphics_mhz", np.int32), ("sm_mhz", np.int32),
                         ("memory_mhz", np.int32)])
_PROCESS_DTYPE = np.dtype([("t_mono_ns", np.int64), ("t_wall", np.float64), ("pid", np.int64),
                           ("kind", object), ("used_bytes", np.int64), ("name", object)])

#: The line a clean stop closes a trace with, apart from the "# key=value"
#: metadata so that no key can be taken for it.
_COMPLETE = "#! complete"
FLUSH_SECONDS = 1.0

Reader = Callable[[], tuple[int, int]]


@dataclass(frozen=True)
class ProcessSample:
    """What one process sample reads: the GPU's P-state and clocks, None
    where the GPU does not report them, and the processes on it."""

    pstate: int | None
    graphics_mhz: int | None
    sm_mhz: int | None
    memory_mhz: int | None
    processes: list[nvml.GpuProcess]


ProcessReader = Callable[[], ProcessSample]


def nvml_reader() -> tuple[int, int]:
    """Free and used device memory, by the driver's account."""
    m = nvml.memory()
    return m.free, m.used


def nvml_process_reader() -> ProcessSample:
    """The GPU's P-state and clocks, and every process on it with what it
    holds, by the driver's account."""
    mhz = nvml.clocks()
    return ProcessSample(nvml.performance_state(), mhz["graphics"], mhz["sm"], mhz["memory"],
                         nvml.processes())


def device_meta() -> dict[str, object]:
    """What a recording says about where it was made."""
    return {"device": nvml.device_name(), "driver": nvml.driver_version(),
            "total_bytes": nvml.memory().total}


def processes_path(path: str | Path) -> Path:
    """Where the processes stream of the recording at `path` is written:
    beside it, trace.csv.gz becoming trace.procs.csv.gz."""
    path = Path(path)
    name = path.name
    for suffix in (".csv.gz", ".gz", ".csv"):
        if name.endswith(suffix):
            return path.with_name(name[: -len(suffix)] + ".procs" + suffix)
    return path.with_name(name + ".procs")


# -- the schedule and the files, shared by both streams ------------------------------------


def _schedule(rate_hz: float, duration: float | None, stop: threading.Event | None,
              sample: Callable[[], None]) -> int:
    """Call `sample` at start + k / rate_hz seconds, k = 0, 1, ..., until
    `duration` has passed or `stop` is set. Returns the deadlines missed:
    every deadline in [0, duration) is either sampled or missed."""
    if rate_hz <= 0:
        raise ValueError(f"rate_hz must be positive, got {rate_hz}")
    period = 1e9 / rate_hz
    deadlines = None if duration is None else math.ceil(duration * 1e9 / period)
    start = time.monotonic_ns()
    k = missed = 0
    while deadlines is None or k < deadlines:
        wait = (start + k * period - time.monotonic_ns()) / 1e9
        if stop is not None:
            if stop.wait(max(wait, 0.0)):
                break
        elif wait > 0:
            time.sleep(wait)
        sample()
        # The next deadline not yet passed; any skipped are missed.
        due = int((time.monotonic_ns() - start) // period) + 1
        if deadlines is not None:
            due = min(due, deadlines)
        missed += max(due - (k + 1), 0)
        k = max(k + 1, due)
    return missed


class _TraceWriter:
    """A trace file: its format line, its metadata, its column line, rows as
    CSV, a flush at least a second apart while rows arrive, and, on a clean
    close, the closing line with its counts."""

    def __init__(self, path: str | Path, fmt: str, meta: dict, columns: Iterable[str]):
        self._file = gzip.open(path, "wt", newline="")
        self._csv = csv.writer(self._file, lineterminator="\n")
        self._file.write(f"# {fmt}\n")
        for key, value in meta.items():
            self._file.write(f"# {key}={value}\n")
        self._file.write(",".join(columns) + "\n")
        self._last_flush = time.monotonic_ns()

    def row(self, *values) -> None:
        self._csv.writerow(values)
        now = time.monotonic_ns()
        if now - self._last_flush >= FLUSH_SECONDS * 1e9:
            self._file.flush()
            self._last_flush = now

    def close(self, counts: dict | None) -> None:
        """Close the file; with `counts`, as a clean stop, writing them on the
        closing line. Without, the file is left as a kill would leave it."""
        if counts is not None:
            fields = " ".join(f"{k}={v}" for k, v in counts.items())
            self._file.write(f"{_COMPLETE} {fields}\n")
        self._file.close()


def _read_trace(path: str | Path, fmt: str) -> tuple[dict, list[list[str]]]:
    """A trace's metadata and its rows, as lists of fields. A trace cut
    short, by a kill or a crash, reads back to its last whole line, and its
    metadata says complete=False; a clean one's says complete=True and carries
    its counts. A missing file raises FileNotFoundError, one that is not gzip
    raises OSError, and one that is not a trace of this format raises
    ValueError."""
    meta: dict[str, object] = {"complete": False}
    lines: list[str] = []
    with gzip.open(path, "rt", newline="") as f:
        try:
            first = f.readline()
            if first.rstrip("\n") != f"# {fmt}":
                raise ValueError(f"{path} is not a contention trace of the kind {fmt!r}: "
                                 f"it begins {first[:60]!r}")
            for line in f:
                if not line.endswith("\n"):
                    break  # the last line was cut mid-write
                if line.startswith(_COMPLETE):
                    meta["complete"] = True
                    for field in line[len(_COMPLETE):].split():
                        key, value = field.split("=", 1)
                        meta[key] = int(value)
                elif line.startswith("# ") and "=" in line:
                    key, value = line[2:].rstrip("\n").split("=", 1)
                    meta[key] = value
                else:
                    lines.append(line)
        except EOFError:
            pass  # the stream ends without its trailer: the recorder was killed
    rows = list(csv.reader(io.StringIO("".join(lines[1:]))))  # lines[0] names the columns
    return meta, rows


# -- recording -----------------------------------------------------------------------------


def record(path: str | Path | None, rate_hz: float = 50.0, *, duration: float | None = None,
           stop: threading.Event | None = None, reader: Reader | None = None,
           meta: dict | None = None, processes_rate_hz: float = 5.0,
           process_reader: ProcessReader | None = None) -> dict:
    """Sample `reader` every 1 / rate_hz seconds until `duration` has passed or
    `stop` is set, writing each sample to `path` if one is given. Returns the
    summary of what was recorded (see summarise).

    Alongside, on its own thread and the same clock, `process_reader` is
    sampled every 1 / processes_rate_hz seconds into processes_path(path).
    With the default readers both streams read NVML; a caller that passes its
    own `reader` records a processes stream only if it passes a
    `process_reader` too. processes_rate_hz=0 turns the stream off.

    The two streams fail together. If either raises, the recording stops
    within a device period, neither file is closed as complete, and the
    error is raised here."""
    if reader is None:
        reader = nvml_reader
        if process_reader is None:
            process_reader = nvml_process_reader
        if path is not None:
            meta = {**device_meta(), **(meta or {})}
    meta = dict(meta or {})
    rows: list[tuple] = []
    out = None
    if path is not None:
        out = _TraceWriter(path, FORMAT, {**meta, "rate_hz": f"{rate_hz:g}"}, COLUMNS)
    stream = None
    if process_reader is not None and processes_rate_hz > 0:
        stream = _ProcessStream(None if path is None else processes_path(path),
                                processes_rate_hz, duration, process_reader, meta)

    def sample() -> None:
        if stream is not None and stream.error is not None:
            raise RuntimeError("the processes stream failed") from stream.error
        t0 = time.monotonic_ns()
        wall = time.time()
        free, used = reader()
        t1 = time.monotonic_ns()
        rows.append((t0, wall, free, used, t1 - t0))
        if out is not None:
            out.row(t0, f"{wall:.6f}", free, used, t1 - t0)

    missed = None
    try:
        if stream is not None:
            stream.start()
        missed = _schedule(rate_hz, duration, stop, sample)
    finally:
        if stream is not None:
            stream.end(clean=missed is not None)
            if stream.error is not None:
                missed = None  # it failed after the device stream's last check
        if out is not None:
            out.close(None if missed is None else {"samples": len(rows), "missed": missed})
    if stream is not None and stream.error is not None:
        raise RuntimeError("the processes stream failed") from stream.error
    samples = np.array(rows, dtype=_DTYPE)
    summary = {**summarise(samples), "missed": missed, "rate_hz": rate_hz}
    if stream is not None:
        summary["processes"] = stream.summary
    return summary


class _ProcessStream(threading.Thread):
    """The processes stream: one state row and one row per process at every
    sample, until the device stream ends it. An error is kept in `error` for
    the device stream to raise, and ends this stream with its file left
    incomplete."""

    def __init__(self, path: Path | None, rate_hz: float, duration: float | None,
                 reader: ProcessReader, meta: dict):
        super().__init__(name="recorder-processes", daemon=True)
        self._path, self._rate_hz, self._duration = path, rate_hz, duration
        self._reader, self._meta = reader, meta
        self._done = threading.Event()
        self._clean = True
        self.error: BaseException | None = None
        self.summary: dict = {}

    def end(self, clean: bool) -> None:
        """Stop the stream and wait for it; `clean` says whether the device
        stream ended cleanly, and so whether this one may close as complete."""
        self._clean = clean
        self._done.set()
        self.join()

    def run(self) -> None:
        try:
            self._record()
        except Exception as exc:  # noqa: BLE001 - raised by the device stream
            self.error = exc

    def _record(self) -> None:
        out = None
        if self._path is not None:
            # A row's first field says which kind it is; the metadata names
            # the columns of each kind.
            out = _TraceWriter(self._path, PROCESS_FORMAT,
                               {**self._meta, "rate_hz": f"{self._rate_hz:g}",
                                "state_columns": ",".join(STATE_COLUMNS),
                                "process_columns": ",".join(PROCESS_COLUMNS)},
                               ("row", "fields..."))
        query_ns: list[int] = []
        process_rows = 0

        def unknown(value: int | None) -> int:
            return -1 if value is None else value

        def sample() -> None:
            nonlocal process_rows
            t0 = time.monotonic_ns()
            wall = f"{time.time():.6f}"
            got = self._reader()
            t1 = time.monotonic_ns()
            query_ns.append(t1 - t0)
            if out is not None:
                out.row(STATE_ROW, t0, wall, t1 - t0, unknown(got.pstate),
                        unknown(got.graphics_mhz), unknown(got.sm_mhz), unknown(got.memory_mhz))
                for p in got.processes:
                    out.row(PROCESS_ROW, t0, wall, p.pid, p.kind, unknown(p.used_bytes),
                            p.name or "")
                    process_rows += 1

        missed = None
        try:
            missed = _schedule(self._rate_hz, self._duration, self._done, sample)
        finally:
            if out is not None:
                out.close({"samples": len(query_ns), "missed": missed,
                           "process_rows": process_rows}
                          if missed is not None and self._clean else None)
        query_ms = np.array(query_ns) / 1e6
        self.summary = {"samples": len(query_ms), "missed": missed, "rate_hz": self._rate_hz}
        if len(query_ms):
            self.summary["query_ms"] = {"median": float(np.median(query_ms)),
                                        "p99": float(np.percentile(query_ms, 99)),
                                        "max": float(query_ms.max())}


# -- reading ------------------------------------------------------------------------------


def read(path: str | Path) -> tuple[dict, np.ndarray]:
    """A device recording's metadata and samples (see _read_trace for what a
    recording cut short reads as, and what raises)."""
    meta, rows = _read_trace(path, FORMAT)
    return meta, np.array([(int(a), float(b), int(c), int(d), int(e)) for a, b, c, d, e in rows],
                          dtype=_DTYPE)


def read_processes(path: str | Path) -> tuple[dict, np.ndarray, np.ndarray]:
    """A processes recording's metadata, its state rows (P-state and clocks,
    one per sample) and its process rows (one per process per sample). A
    value the driver did not report, a size, P-state or clock, is -1."""
    meta, rows = _read_trace(path, PROCESS_FORMAT)
    states = [(int(r[1]), float(r[2]), int(r[3]), int(r[4]), int(r[5]), int(r[6]), int(r[7]))
              for r in rows if r[0] == STATE_ROW]
    procs = [(int(r[1]), float(r[2]), int(r[3]), r[4], int(r[5]), r[6])
             for r in rows if r[0] == PROCESS_ROW]
    return (meta, np.array(states, dtype=_STATE_DTYPE),
            np.array(procs, dtype=_PROCESS_DTYPE) if procs
            else np.zeros(0, dtype=_PROCESS_DTYPE))


def summarise(samples: np.ndarray) -> dict:
    """How the schedule was kept: the samples taken, the achieved rate, the
    spread of the periods between them, and what each query cost."""
    n = len(samples)
    if n < 2:
        return {"samples": n}
    periods = np.diff(samples["t_mono_ns"]) / 1e6
    query = samples["query_ns"] / 1e3
    span = (samples["t_mono_ns"][-1] - samples["t_mono_ns"][0]) / 1e9
    return {"samples": n, "span_seconds": span, "achieved_hz": (n - 1) / span,
            "period_ms": {"mean": float(periods.mean()), "std": float(periods.std()),
                          "p50": float(np.percentile(periods, 50)),
                          "p99": float(np.percentile(periods, 99)), "max": float(periods.max())},
            "query_us": {"median": float(np.median(query)),
                         "p99": float(np.percentile(query, 99)), "max": float(query.max())}}
