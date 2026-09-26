"""RQ1's results from a contention recording (#50).

A recording's spikes (spikes.py) become RQ1's results here:

- **Attribution.** Each spike is laid against the 5 Hz processes stream and
  names the process whose memory rose with it: the one that gained most
  between the last process sample before the spike's rise began and the
  samples taken while it lasted, if that gain is at least half the spike's
  amplitude. Otherwise it names no one, and says why: no process sample fell
  inside the spike, which one shorter than a 5 Hz period can escape; the
  driver did not report what a process held; or no process gained enough to
  explain it, as when the memory went to something NVML does not list.
- **Statistics.** A labelled recording gives, per action, the distributions
  of amplitude, rise time, duration and recovery (median, P90, max). An
  unlabelled one, a passive session, gives rates per hour.
- **Sensitivity.** The spikes are counted again at 32, 64 and 128 MiB.
- **The log.** log_recording appends one benchmark-log entry per recording,
  carrying the sha256 of each of its files, so that a published trace can be
  matched to the statistics drawn from it.

This is measurement, not inference, so it computes in NumPy on the host
(ADR-0002, amendment on its scope).
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import benchlog, recorder, spikes

MiB = 2**20
SENSITIVITY_MIB = (32, 64, 128)
#: The share of a spike's amplitude a process must have gained to be named.
DEFAULT_MIN_SHARE = 0.5
NO_ACTION = "(between actions)"
NO_PROCESS = "(none)"


@dataclass(frozen=True)
class Attribution:
    """The process a spike is attributed to, or why none is."""

    pid: int | None = None
    name: str | None = None
    rise_bytes: int | None = None  # what it gained over the spike
    share: float | None = None  # rise_bytes over the spike's amplitude
    reason: str | None = None  # why no process is named


def attribute(spike: spikes.Spike, states: np.ndarray, procs: np.ndarray, *,
              min_share: float = DEFAULT_MIN_SHARE) -> Attribution:
    """The process most responsible for `spike`, from a processes stream's
    state rows (one per sample) and process rows (recorder.read_processes)."""
    rise_began = (spike.start_ns if spike.rise_s is None
                  else min(spike.start_ns, spike.peak_start_ns - int(spike.rise_s * 1e9)))
    sample_times = states["t_mono_ns"]
    earlier = sample_times[sample_times <= rise_began]
    during = sample_times[(sample_times >= spike.start_ns)
                          & (sample_times <= spike.start_ns + spike.duration_s * 1e9)]
    if not len(earlier):
        return Attribution(reason="no process sample before the spike")
    if not len(during):
        return Attribution(reason="no process sample during the spike")

    before = {int(r["pid"]): int(r["used_bytes"]) for r in procs[procs["t_mono_ns"] == earlier[-1]]}
    inside = procs[np.isin(procs["t_mono_ns"], during)]
    unreported = set(inside["pid"][inside["used_bytes"] < 0]) | {
        pid for pid, used in before.items() if used < 0}
    best = None
    for pid in set(int(p) for p in inside["pid"]) - {int(p) for p in unreported}:
        rows = inside[inside["pid"] == pid]
        gain = int(rows["used_bytes"].max()) - before.get(pid, 0)  # new: it held nothing
        if best is None or gain > best[1]:
            best = (pid, gain, str(rows["name"][0]))
    if best is None or best[1] < min_share * spike.amplitude_bytes:
        reason = f"no process's memory rose by {min_share:.0%} of the amplitude"
        if unreported:
            reason += "; the driver did not report what some processes held"
        return Attribution(reason=reason)
    pid, gain, name = best
    return Attribution(pid=pid, name=name, rise_bytes=gain,
                       share=gain / spike.amplitude_bytes)


def _spread(values: list[float]) -> dict | None:
    """Median, P90 and max, or None if there are no values."""
    if not values:
        return None
    a = np.array(values, dtype=float)
    return {"median": float(np.median(a)), "p90": float(np.percentile(a, 90)),
            "max": float(a.max())}


def _distributions(group: list[spikes.Spike]) -> dict:
    def ms(seconds: list[float | None]) -> dict | None:
        return _spread([s * 1e3 for s in seconds if s is not None])  # None: not measured

    return {"count": len(group),
            "amplitude_mib": _spread([s.amplitude_bytes / MiB for s in group]),
            "rise_ms": ms([s.rise_s for s in group]),
            "duration_ms": ms([s.duration_s for s in group]),
            "recovery_ms": ms([s.recovery_s for s in group])}


def analyse(samples: np.ndarray, states: np.ndarray, procs: np.ndarray,
            labels: np.ndarray | None, *, threshold_mib: int = 64,
            min_share: float = DEFAULT_MIN_SHARE) -> dict:
    """RQ1's results for one recording: its device samples, its processes
    stream and its labels (None, or empty, for a passive session). JSON as
    it stands, for the benchmark log."""
    t, free = samples["t_mono_ns"], samples["free_bytes"]
    found = spikes.find_spikes(t, free, threshold_bytes=threshold_mib * MiB)
    span_hours = float(t[-1] - t[0]) / 3.6e12 if len(t) > 1 else 0.0
    named = [attribute(s, states, procs, min_share=min_share) for s in found]

    attributed: dict[str, int] = {}
    for a in named:
        attributed[a.name or NO_PROCESS] = attributed.get(a.name or NO_PROCESS, 0) + 1
    endings = {e: sum(s.ending == e for s in found) for e in ("recovered", "lasting", "censored")}
    results = {
        "samples": int(len(t)), "span_hours": span_hours,
        "threshold_mib": threshold_mib, "spikes": len(found),
        "sensitivity_mib": {str(m): len(found) if m == threshold_mib
                            else len(spikes.find_spikes(t, free, threshold_bytes=m * MiB))
                            for m in SENSITIVITY_MIB},
        "endings": endings, "attributed": attributed,
        "all": _distributions(found),
    }
    if labels is not None and len(labels):
        # A spike belongs to the action in progress when it began.
        actions = recorder.actions_in_progress(
            np.array([s.start_ns for s in found], dtype=np.int64), labels)
        by_action: dict[str, list[spikes.Spike]] = {}
        for s, action in zip(found, actions):
            by_action.setdefault(action or NO_ACTION, []).append(s)
        results["by_action"] = {a: _distributions(g) for a, g in sorted(by_action.items())}
    elif span_hours > 0:
        results["per_hour"] = {"spikes": len(found) / span_hours,
                               "by_process": {n: c / span_hours
                                              for n, c in sorted(attributed.items())}}
    return results


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def log_recording(path: str | Path, *, issue: int, log: str | Path = benchlog.DEFAULT_LOG,
                  threshold_mib: int = 64, min_share: float = DEFAULT_MIN_SHARE) -> dict:
    """Analyse the recording at `path`, with its processes and labels files
    if they are beside it, and append its results to the benchmark log as a
    "contention-trace" entry carrying each file's sha256. Returns the entry.
    A file that is not a recording raises before anything is logged."""
    path = Path(path)
    meta, samples = recorder.read(path)
    files = [path]
    states = np.zeros(0, dtype=recorder.STATE_DTYPE)
    procs = np.zeros(0, dtype=recorder.PROCESS_DTYPE)
    labels = None
    if (p := recorder.processes_path(path)).exists():
        _, states, procs = recorder.read_processes(p)
        files.append(p)
    if (p := recorder.labels_path(path)).exists():
        _, labels = recorder.read_labels(p)
        files.append(p)

    results = {**analyse(samples, states, procs, labels, threshold_mib=threshold_mib,
                         min_share=min_share),
               "complete": bool(meta["complete"]), "missed": meta.get("missed")}
    precision = meta.get("precision")
    return benchlog.append(
        "contention-trace", model=meta.get("model"),
        context_length=int(meta["context_length"]) if "context_length" in meta else None,
        precision_tiers={precision: 1.0} if precision else None,
        config={"issue": issue, "trace": path.name,
                "files": {f.name: _sha256(f) for f in files},
                "recording": {k: v for k, v in meta.items()
                              if k not in ("complete", "samples", "missed")},
                "window_s": spikes.DEFAULT_WINDOW_S, "threshold_mib": threshold_mib,
                "min_duration_s": spikes.DEFAULT_MIN_DURATION_S, "min_share": min_share},
        results=results, log=log)
