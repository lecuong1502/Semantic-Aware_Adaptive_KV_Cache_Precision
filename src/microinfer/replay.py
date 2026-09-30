"""Replaying a recorded trace on the contention simulator (#60).

The simulator (#58) takes memory on a schedule; #59 makes schedules from
patterns, and this makes one from an RQ1 recording, so that a real episode
of contention can be played again on demand.

**What is replayed is what the others held.** The device stream, at 50 Hz,
is the driver's used memory: every process's, the driver's own, and in a
with-engine recording the engine's. The engine is what RQ2 and RQ3 will run
beside a replay, not part of the contention, so its memory is taken off; its
pid comes from the hold's status beside the recording. The process samples
that say what it held come at 5 Hz, and taken off as they stand they would
leave its every change, its weights' 3 GB among them, in the others' memory
for up to 200 ms. So between the two process samples that saw a change, the
device's own steps in its direction are the engine's, until the change is
spent; another process's step in that window, if the same way, is taken for
the engine's, an error no larger than the change and no longer than 200 ms.
Before the first process sample and after the last, what the engine held is
not known, and those device samples are left out: the engine's exit, when a
recording ends, falls there. What is left is every
other process's, whole-desktop noise included, at the device stream's own
rate, so a spike keeps the rise it was recorded with. Its level is taken
above its least in the chosen window: a replay can add memory to a desktop,
not take away what the desktop holds.

**A gap holds the level before it.** The recorder skips deadlines it misses
(recorder.py); the schedule keeps the last level through them rather than
invent what free memory did.

**A replay is compared with its original.** replay() runs the simulator as a
process of its own while the recorder records the device, as RQ1 did; after
an idle lead with the simulator's context in place, the schedule begins.
compare() measures the recording of the replay against what the original
held, and states whether it is within these bounds:

- **sample by sample**, P90 of the error at most ERROR_BOUND_MIB: an idle
  desktop's noise, a few MiB (ADR-0007), in both recordings, and a granule
  of rounding. Most of what is left is a ramp's samples, a few ms apart;
- **spike by spike**, at spikes.py's definition: the same spikes, each
  beginning within START_BOUND_MS, two samples, and its amplitude within
  AMPLITUDE_BOUND_MIB.

This is measurement, not inference, so it computes in NumPy on the host
(ADR-0002, amendment on its scope), and the recording process takes no
device memory: the simulator is the only process here with a CUDA context.
"""

from __future__ import annotations

import json
import signal
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import patterns, recorder, spikes
from .contention import spread
from .footprint import MIB
from .schedule import Schedule

ERROR_BOUND_MIB = 16.0
START_BOUND_MS = 40.0
AMPLITUDE_BOUND_MIB = 16.0
#: How far apart a spike and its replay may begin and still be paired.
MATCH_S = 0.5
#: The idle lead a replay records before its schedule begins, and the part
#: of it the desktop's own level is measured over.
LEAD_S = 3.0
IDLE_S = 1.0
TAIL_S = 2.0
SIMULATOR = Path(__file__).resolve().parents[2] / "tools" / "simulate_contention.py"


@dataclass(frozen=True)
class Held:
    """What the other processes held above their least, per device sample,
    at seconds from the window's start, and the engine taken off, if any."""

    t_s: np.ndarray
    bytes: np.ndarray
    engine_pid: int | None = None


def _engine_bytes(path: Path, t_ns: np.ndarray, used: np.ndarray,
                  pid: int) -> tuple[np.ndarray, np.ndarray]:
    """The engine's memory at each device sample, 0 where the process
    samples do not list it, and where it is known: between the first process
    sample and the last. Between two process samples that disagree, the
    engine's change is the device's own steps in its direction, taken in
    turn until the change is spent: most of what the device did then was
    the engine's."""
    procs_path = recorder.processes_path(path)
    if not procs_path.exists():
        raise ValueError(f"{path} ran beside the engine (pid {pid}), and its memory cannot be "
                         f"taken off without {procs_path.name}")
    _, states, procs = recorder.read_processes(procs_path)
    if not len(states):
        raise ValueError(f"{procs_path.name} has no samples")
    engine = procs[procs["pid"] == pid]
    if (engine["used_bytes"] < 0).any():
        raise ValueError(f"the driver did not report what the engine (pid {pid}) held")
    at = dict(zip(engine["t_mono_ns"].tolist(), engine["used_bytes"].tolist()))
    times = states["t_mono_ns"]
    # A process sample says what was held at some moment of its query.
    seen = times + states["query_ns"]
    per_state = np.array([at.get(int(t), 0) for t in times], dtype=np.int64)
    out = per_state[np.clip(np.searchsorted(seen, t_ns, side="right") - 1, 0, None)]
    for k in np.nonzero(np.diff(per_state))[0] + 1:
        # The device samples after the process query that saw the old value
        # began, up to the end of the one that saw the new.
        lo, hi = np.searchsorted(t_ns, [times[k - 1], seen[k]], side="right")
        change, level = int(per_state[k] - per_state[k - 1]), int(per_state[k - 1])
        for i in range(lo, hi):
            step = int(used[i] - used[i - 1]) if i else 0
            if (step > 0) == (change > 0):
                took = min(abs(step), abs(change)) * (1 if change > 0 else -1)
                level, change = level + took, change - took
            out[i] = level
        if hi > lo:
            out[hi - 1] = per_state[k]  # whatever the device did not show, by the new sample
    return out, (t_ns >= seen[0]) & (t_ns <= seen[-1])


def held(path: str | Path, *, start_s: float = 0.0, end_s: float | None = None,
         engine_pid: int | None = None) -> Held:
    """What the processes other than the engine held over the recording at
    `path`, from `start_s` to `end_s` seconds after its first sample. The
    engine is the pid given, or the one in the hold's status beside the
    recording; without either, nothing is taken off."""
    path = Path(path)
    _, samples = recorder.read(path)
    if end_s is not None and end_s <= start_s:
        raise ValueError(f"a window ends after it starts; got {start_s} to {end_s} s")
    if engine_pid is None and (status := recorder.companion(path, ".hold.json")).exists():
        engine_pid = int(json.loads(status.read_text())["pid"])
    t, used = samples["t_mono_ns"], samples["used_bytes"]
    s = (t - t[0]) / 1e9 if len(t) else np.zeros(0)
    keep = (s >= start_s) & (s <= (np.inf if end_s is None else end_s))
    if engine_pid is not None:
        engine, known = _engine_bytes(path, t, used, engine_pid)
        used, keep = used - engine, keep & known
    if keep.sum() < 2:
        raise ValueError(f"{path} has fewer than two samples from {start_s} to {end_s} s"
                         + ("" if engine_pid is None else " that a process sample brackets"))
    used = used[keep]
    return Held((t[keep] - t[keep][0]) / 1e9, used - used.min(), engine_pid)


def to_schedule(h: Held) -> Schedule:
    """The simulator's schedule for `h`: each sample's level, in whole
    granules, from its time until the next's, and then 0, one period after
    the last."""
    end = round(float(h.t_s[-1] + np.median(np.diff(h.t_s))), 6)
    return Schedule(patterns.sampled(h.t_s, h.bytes).points + [(end, 0)])


def from_recording(path: str | Path, **window) -> Schedule:
    """The schedule that replays the recording at `path` (see held)."""
    return to_schedule(held(path, **window))


# -- replaying -----------------------------------------------------------------------------


def replay(schedule: Schedule, out: str | Path, *, lead_s: float = LEAD_S,
           tail_s: float = TAIL_S, stop: threading.Event | None = None) -> int:
    """Record the device to `out`, as RQ1 did, while the simulator, a process
    of its own, takes `schedule` after `lead_s` of taking nothing; stop
    `tail_s` after its last point. The schedule and the simulator's events
    are written beside `out`. Returns when the schedule began, on the
    recorder's clock."""
    out = Path(out)
    shifted = Schedule([(0.0, 0)] + [(t + lead_s, b) for t, b in schedule.points])
    schedule_path = recorder.companion(out, ".schedule.json")
    events_path = recorder.companion(out, ".events.jsonl")
    schedule.save(schedule_path)
    shifted_path = recorder.companion(out, ".shifted.json")
    shifted.save(shifted_path)
    events_path.unlink(missing_ok=True)

    done = threading.Event()
    failure: list[BaseException] = []

    def record() -> None:
        try:
            recorder.record(out, stop=done)
        except BaseException as exc:  # noqa: BLE001 - raised by the caller's thread
            failure.append(exc)

    recording = threading.Thread(target=record, name="replay-recorder")
    recording.start()
    proc = subprocess.Popen([sys.executable, str(SIMULATOR), "--schedule", str(shifted_path),
                             "--events", str(events_path)],
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        if proc.stdout.readline().strip() != "ready":
            raise RuntimeError(f"the simulator did not start: {proc.stderr.read()}")
        deadline = time.monotonic() + shifted.duration_s + 30.0
        while not _applied(events_path, len(shifted.points)):
            if failure or proc.poll() is not None or time.monotonic() > deadline or (
                    stop is not None and stop.is_set()):
                raise RuntimeError("the replay stopped before its schedule ended" +
                                   (f": {failure[0]}" if failure else ""))
            time.sleep(0.05)
        (stop or threading.Event()).wait(tail_s)
    finally:
        if proc.poll() is None:
            proc.send_signal(signal.SIGTERM)
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
        done.set()
        recording.join()
        shifted_path.unlink(missing_ok=True)
    if failure:
        raise failure[0]
    first = json.loads(events_path.read_text().splitlines()[0])
    return first["scheduled_ns"] + int(lead_s * 1e9)


def _applied(events: Path, points: int) -> bool:
    return events.exists() and len(events.read_text().splitlines()) >= points


# -- comparing -----------------------------------------------------------------------------


def _step_at(h: Held, s: np.ndarray) -> np.ndarray:
    """`h`'s level at each of `s` seconds, from its last sample at or before;
    0 before its first."""
    index = np.searchsorted(h.t_s, s, side="right") - 1
    return np.where(index >= 0, h.bytes[np.clip(index, 0, None)], 0)


def _pair(original: list[spikes.Spike], replayed: list[spikes.Spike]):
    """Each original spike with the unpaired replayed one that began nearest
    it, within MATCH_S."""
    free = list(replayed)
    pairs = []
    for o in original:
        near = min(free, key=lambda r: abs(r.start_ns - o.start_ns), default=None)
        if near is not None and abs(near.start_ns - o.start_ns) <= MATCH_S * 1e9:
            pairs.append((o, near))
            free.remove(near)
    return pairs


def compare(original: Held, samples: np.ndarray, started_ns: int, *,
            idle_s: float = IDLE_S) -> dict:
    """The recording of a replay, its device `samples`, against the
    `original` it replayed from `started_ns`. The desktop's own level is the
    median used memory over the `idle_s` before the start; what the replay
    took is the used memory above it. Spikes are sought in both over the
    replayed time alone, so in both from one baseline window (spikes.py)
    after its start. JSON as it stands, for the log."""
    t = samples["t_mono_ns"]
    idle = samples["used_bytes"][(t >= started_ns - idle_s * 1e9) & (t < started_ns)]
    if not len(idle):
        raise ValueError("the replay's recording has no samples before its schedule began, "
                         "to measure the desktop's own level by")
    base = int(np.median(idle))
    s = (t - started_ns) / 1e9
    during = (s >= 0) & (s <= original.t_s[-1])
    taken = samples["used_bytes"][during] - base
    error = np.abs(taken - _step_at(original, s[during])) / MIB

    found = spikes.find_spikes((original.t_s * 1e9).astype(np.int64), -original.bytes)
    again = spikes.find_spikes(t[during] - started_ns, samples["free_bytes"][during])
    pairs = _pair(found, again)
    starts = [abs(r.start_ns - o.start_ns) / 1e6 for o, r in pairs]
    amplitudes = [abs(r.amplitude_bytes - o.amplitude_bytes) / MIB for o, r in pairs]
    rises = [abs(r.rise_s - o.rise_s) * 1e3 for o, r in pairs
             if r.rise_s is not None and o.rise_s is not None]
    error_mib = spread(error.tolist())
    return {
        "samples": int(during.sum()), "idle_used_mib": base / MIB,
        "error_mib": error_mib,
        "spikes": {"original": len(found), "replayed": len(again), "matched": len(pairs),
                   "start_offset_ms": spread(starts),
                   "amplitude_error_mib": spread(amplitudes),
                   "rise_error_ms": spread(rises)},
        "bounds": {"error_p90_mib": ERROR_BOUND_MIB, "start_ms": START_BOUND_MS,
                   "amplitude_mib": AMPLITUDE_BOUND_MIB},
        "within": {
            "error": error_mib is not None and error_mib["p90"] <= ERROR_BOUND_MIB,
            "spikes": len(pairs) == len(found) == len(again),
            "start": all(x <= START_BOUND_MS for x in starts),
            "amplitude": all(x <= AMPLITUDE_BOUND_MIB for x in amplitudes)},
    }
