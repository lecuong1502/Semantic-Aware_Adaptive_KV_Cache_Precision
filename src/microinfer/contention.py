"""RQ1's results from a contention recording (#50).

A recording's spikes (spikes.py) become RQ1's results here:

- **Attribution.** Each spike is laid against the 5 Hz processes stream and
  names the process most responsible: the one whose memory gained most
  between the last process sample before the spike's rise began and the
  samples taken during its peak, provided that gain is at least a quarter of
  the spike's amplitude. It says what share of the amplitude that process
  took, so a spike that two processes share still names the larger. Where no
  process can be named, it says why (Unnamed).
- **Statistics.** Per labelled action, every action the labels name, those
  with no spike included: the distributions of amplitude and rise time over
  all its spikes, and of duration and recovery over those that recovered
  (median, P90, max); the duration of a censored spike is only a lower
  bound. For an unlabelled recording, a passive session, rates per hour of
  recorded time, gaps left out, once there are ten minutes of it.
- **Sensitivity.** Spike counts at 32, 64 and 128 MiB, the release at half
  of each, in total and per action.
- **The log.** log_recording appends one benchmark-log entry per recording,
  with every spike and its attribution, and the sha256 of each of the
  recording's files, so that a published trace can be matched to the
  statistics drawn from it.

This is measurement, not inference, so it computes in NumPy on the host
(ADR-0002, amendment on its scope).
"""

from __future__ import annotations

import enum
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import get_args

import numpy as np

from . import benchlog, recorder, spikes
from .footprint import MIB

SENSITIVITY_BYTES = (32 * MIB, 64 * MIB, 128 * MIB)
#: The share of a spike's amplitude a process must have gained to be named.
DEFAULT_MIN_SHARE = 0.25
#: The recorded time below which a rate per hour would say more about chance
#: than about the desktop.
MIN_RATE_SECONDS = 600.0
NO_ACTION = "(between actions)"
NO_PROCESS = "(none)"


class Unnamed(str, enum.Enum):
    """Why a spike names no process."""

    NO_STREAM = "the recording has no processes stream"
    NO_SAMPLE_BEFORE = "no process sample before the spike, or it listed no process"
    NO_SAMPLE_DURING = "no process sample during the spike's peak"
    UNREPORTED = "the driver did not report what the processes that gained held"
    TOO_SMALL = "no process gained a quarter of the amplitude"


@dataclass(frozen=True)
class Attribution:
    """The process a spike is attributed to, or why none is."""

    pid: int | None = None
    name: str | None = None
    gain_bytes: int | None = None  # what it gained over the spike
    share: float | None = None  # gain_bytes over the spike's amplitude
    unnamed: Unnamed | None = None


@dataclass(frozen=True)
class ProcessStream:
    """A recording's processes stream: its state rows, one per sample, and
    its process rows (recorder.read_processes)."""

    states: np.ndarray
    procs: np.ndarray

    @classmethod
    def absent(cls) -> ProcessStream:
        return cls(np.zeros(0, dtype=recorder.STATE_DTYPE),
                   np.zeros(0, dtype=recorder.PROCESS_DTYPE))


def attribute(spike: spikes.Spike, stream: ProcessStream, *,
              min_share: float = DEFAULT_MIN_SHARE) -> Attribution:
    """The process most responsible for `spike`.

    A process's gain is the most it held in a sample during the peak, less
    what it held in the last sample before the rise began. A process not in
    that sample, or there under the same pid but another name (the pid
    reused), gained all it holds. A process whose holding the driver did not
    report cannot be credited, and if it could have been the one, nothing is
    named."""
    if not len(stream.states):
        return Attribution(unnamed=Unnamed.NO_STREAM)
    times = stream.states["t_mono_ns"]
    earlier = times[times <= spike.rise_start_ns]
    during = times[(times >= spike.peak_start_ns) & (times <= spike.peak_end_ns)]
    listed = stream.procs[stream.procs["t_mono_ns"] == earlier[-1]] if len(earlier) else []
    if not len(listed):
        return Attribution(unnamed=Unnamed.NO_SAMPLE_BEFORE)
    if not len(during):
        return Attribution(unnamed=Unnamed.NO_SAMPLE_DURING)

    before = {(int(r["pid"]), str(r["name"])): int(r["used_bytes"]) for r in listed}
    peak = stream.procs[np.isin(stream.procs["t_mono_ns"], during)]
    gains: dict[tuple[int, str], int | None] = {}
    for r in peak:
        key, used = (int(r["pid"]), str(r["name"])), int(r["used_bytes"])
        held = before.get(key, 0)
        if used < 0 or held < 0 or gains.get(key, 0) is None:
            gains[key] = None  # unreported, now or before
        else:
            gains[key] = max(gains.get(key, used - held), used - held)
    known = {k: g for k, g in gains.items() if g is not None}
    floor = min_share * spike.amplitude_bytes
    best = max(known, key=known.get, default=None)
    if best is None or known[best] < floor:
        unreported = len(known) < len(gains)
        return Attribution(unnamed=Unnamed.UNREPORTED if unreported else Unnamed.TOO_SMALL)
    pid, name = best
    return Attribution(pid=pid, name=name, gain_bytes=known[best],
                       share=known[best] / spike.amplitude_bytes)


def _spread(values: list[float]) -> dict | None:
    """Median, P90 and max, or None if there are no values."""
    if not values:
        return None
    a = np.array(values, dtype=float)
    return {"median": float(np.median(a)), "p90": float(np.percentile(a, 90)),
            "max": float(a.max())}


def _ms(seconds: list[float | None]) -> dict | None:
    return _spread([s * 1e3 for s in seconds if s is not None])  # None: not measured


def _distributions(group: list[tuple[spikes.Spike, Attribution]]) -> dict:
    found = [s for s, _ in group]
    recovered = [s for s in found if s.recovered]
    endings = {e: sum(s.ending == e for s in found) for e in get_args(spikes.Ending)}
    return {"count": len(found), "endings": endings,
            "amplitude_mib": _spread([s.amplitude_bytes / MIB for s in found]),
            "rise_ms": _ms([s.rise_s for s in found]),
            "duration_ms": _ms([s.duration_s for s in recovered]),
            "recovery_ms": _ms([s.recovery_s for s in recovered]),
            "attributed": _tally(a.name or NO_PROCESS for _, a in group)}


def _tally(names) -> dict[str, int]:
    counts: dict[str, int] = {}
    for n in names:
        counts[n] = counts.get(n, 0) + 1
    return dict(sorted(counts.items()))


def recorded_seconds(t_mono_ns: np.ndarray, max_gap_s: float = spikes.DEFAULT_MAX_GAP_S) -> float:
    """The time a trace covers, its gaps left out."""
    steps = np.diff(np.asarray(t_mono_ns)) / 1e9
    return float(steps[steps <= max_gap_s].sum())


def analyse(samples: np.ndarray, stream: ProcessStream, labels: np.ndarray | None, *,
            window_s: float = spikes.DEFAULT_WINDOW_S,
            threshold_bytes: int = spikes.DEFAULT_THRESHOLD_BYTES,
            min_share: float = DEFAULT_MIN_SHARE) -> dict:
    """RQ1's results for one recording: its device samples, its processes
    stream and its labels (None, or empty, for a passive session). JSON as
    it stands, for the benchmark log."""
    t, free = samples["t_mono_ns"], samples["free_bytes"]

    def find(threshold: int) -> list[spikes.Spike]:
        return spikes.find_spikes(t, free, window_s=window_s, threshold_bytes=threshold)

    found = find(threshold_bytes)
    named = [attribute(s, stream, min_share=min_share) for s in found]
    labelled = labels is not None and len(labels) > 0

    def actions_of(group: list[spikes.Spike]) -> list[str]:
        if not labelled:
            return [""] * len(group)
        at = recorder.actions_in_progress(np.array([s.start_ns for s in group], np.int64),
                                          labels)
        return [a or NO_ACTION for a in at]

    actions = actions_of(found)
    seconds = recorded_seconds(t)
    results: dict = {
        "samples": int(len(t)), "recorded_hours": seconds / 3600,
        "threshold_mib": threshold_bytes / MIB, "process_stream": len(stream.states) > 0,
        "all": _distributions(list(zip(found, named))),
        "sensitivity_mib": {}, "spikes": [],
    }
    for threshold in SENSITIVITY_BYTES:
        at = found if threshold == threshold_bytes else find(threshold)
        entry = {"count": len(at)}
        if labelled:
            entry["by_action"] = _tally(actions_of(at))
        results["sensitivity_mib"][f"{threshold / MIB:g}"] = entry

    if labelled:
        names = sorted(set(labels["action"]) | set(actions))
        results["by_action"] = {
            a: _distributions([(s, n) for s, n, x in zip(found, named, actions) if x == a])
            for a in names}
    else:
        results["per_hour"] = None if seconds < MIN_RATE_SECONDS else {
            "spikes": len(found) * 3600 / seconds,
            "by_process": {n: c * 3600 / seconds
                           for n, c in results["all"]["attributed"].items()}}

    first = int(t[0]) if len(t) else 0
    for s, a, action in zip(found, named, actions):
        results["spikes"].append({
            "start_s": (s.start_ns - first) / 1e9, "ending": s.ending,
            "amplitude_mib": s.amplitude_bytes / MIB,
            "rise_ms": None if s.rise_s is None else s.rise_s * 1e3,
            "duration_ms": s.duration_s * 1e3,
            "recovery_ms": None if s.recovery_s is None else s.recovery_s * 1e3,
            **({"action": action} if labelled else {}),
            "process": a.name, "pid": a.pid, "share": a.share,
            "unnamed": None if a.unnamed is None else a.unnamed.value})
    return results


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def log_recording(path: str | Path, *, issue: int, log: str | Path = benchlog.DEFAULT_LOG,
                  window_s: float = spikes.DEFAULT_WINDOW_S,
                  threshold_bytes: int = spikes.DEFAULT_THRESHOLD_BYTES,
                  min_share: float = DEFAULT_MIN_SHARE) -> dict:
    """Analyse the recording at `path`, with its processes and labels files
    if they are beside it, and append its results to the benchmark log as a
    "contention-trace" entry carrying each file's sha256. Returns the entry.
    A file that is not a recording raises before anything is logged."""
    path = Path(path)
    meta, samples = recorder.read(path)
    files = [path]
    stream, labels = ProcessStream.absent(), None
    if (p := recorder.processes_path(path)).exists():
        _, states, procs = recorder.read_processes(p)
        stream = ProcessStream(states, procs)
        files.append(p)
    if (p := recorder.labels_path(path)).exists():
        _, labels = recorder.read_labels(p)
        files.append(p)

    results = {**analyse(samples, stream, labels, window_s=window_s,
                         threshold_bytes=threshold_bytes, min_share=min_share),
               "complete": bool(meta["complete"]), "missed": meta.get("missed")}
    context = str(meta.get("context_length", ""))
    return benchlog.append(
        "contention-trace", model=meta.get("model"),
        # What the recording says ran beside it; its precision stays in its
        # metadata below, as a claim about the run, not a tier split measured here.
        context_length=int(context) if context.isdigit() else None,
        precision_tiers=None,
        config={"issue": issue, "trace": path.name,
                "files": {f.name: _sha256(f) for f in files},
                "recording": {k: v for k, v in meta.items()
                              if k not in ("complete", "samples", "missed")},
                "window_s": window_s, "threshold_mib": threshold_bytes / MIB,
                "min_duration_s": spikes.DEFAULT_MIN_DURATION_S, "min_share": min_share},
        results=results, log=log)
