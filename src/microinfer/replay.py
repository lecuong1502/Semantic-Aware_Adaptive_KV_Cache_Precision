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
device's own changes in its direction are the engine's, until the engine's
is spent; another process's change in that window, if the same way, is
taken for the engine's, an error no larger than the change and no longer than 200 ms.
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
- **spike by spike**: each of the original's spikes, found as spikes.py
  finds them, measured the same way on both over its own span: its
  amplitude, the peak above the median of the window before its rise, and
  its edge, when it first reaches half that amplitude. The replay's edge
  within START_BOUND_MS, two samples, and its amplitude within
  AMPLITUDE_BOUND_MIB. The replay's spikes are not sought on their own: a
  spike that crosses the threshold by a MiB in one crosses it in the other
  only by chance, and a lasting drop missed moves the baseline every later
  spike is found against.

The desktop is part of both device traces, so what it does during a replay
counts as error. Where the replay's processes stream is given, compare()
also says how far the simulator's own memory strayed from the schedule, and
how far the rest of the desktop moved: whether a replay that missed its
bounds was the simulator's fault or the desktop's.

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

from . import nvml, patterns, recorder, spikes
from .contention import spread
from .footprint import MIB
from .schedule import Schedule

ERROR_BOUND_MIB = 16.0
START_BOUND_MS = 40.0
AMPLITUDE_BOUND_MIB = 16.0
#: The idle lead a replay records before its schedule begins, and the part
#: of it the desktop's own level is measured over.
LEAD_S = 3.0
IDLE_S = 1.0
TAIL_S = 2.0
#: How long past its schedule's end a replay waits for the simulator's last
#: change, how long for it to exit once stopped, and how often it looks.
REPLAY_SLACK_S = 30.0
EXIT_WAIT_S = 30.0
POLL_S = 0.05
SIMULATOR = Path(__file__).resolve().parents[2] / "tools" / "simulate_contention.py"


@dataclass(frozen=True)
class OthersHeld:
    """What the other processes held above their least, per device sample,
    at seconds from the window's start, and the engine taken off, if any."""

    t_s: np.ndarray
    bytes: np.ndarray
    engine_pid: int | None = None
    #: The headroom the original had with the others at their least, the
    #: engine as it stood: the median of free memory and `bytes` over the
    #: window. A replay that leaves this much, before its schedule takes
    #: anything, gives the machine the original's headroom throughout.
    headroom_bytes: int | None = None
    #: The samples' times on the original's clock, to find them there again.
    t_mono_ns: np.ndarray | None = None


def _engine_bytes(path: Path, t_ns: np.ndarray, used: np.ndarray,
                  pid: int) -> tuple[np.ndarray, np.ndarray]:
    """The engine's memory at each device sample, 0 where the process
    samples do not list it, and where it is known: between the first process
    sample and the last. Between two process samples that disagree, the
    engine's change is the device's own changes in its direction, taken in
    turn until the engine's is spent: most of what the device did then was
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
        left, level = int(per_state[k] - per_state[k - 1]), int(per_state[k - 1])
        for i in range(lo, hi):
            moved = int(used[i] - used[i - 1]) if i else 0
            if moved and (moved > 0) == (left > 0):
                took = min(abs(moved), abs(left)) * (1 if left > 0 else -1)
                level, left = level + took, left - took
            out[i] = level
        if hi > lo:
            out[hi - 1] = per_state[k]  # whatever the device did not show, by the new sample
    return out, (t_ns >= seen[0]) & (t_ns <= seen[-1])


def _engine_pid(path: Path, engine_pid: int | None) -> int | None:
    """The engine's pid: the one given, or the hold's status beside the
    recording's; None if neither."""
    if engine_pid is None and (status := recorder.companion(path, ".hold.json")).exists():
        engine_pid = int(json.loads(status.read_text())["pid"])
    return engine_pid


def _window(t_ns: np.ndarray, start_s: float, end_s: float | None) -> np.ndarray:
    if end_s is not None and end_s <= start_s:
        raise ValueError(f"a window ends after it starts; got {start_s} to {end_s} s")
    s = (t_ns - t_ns[0]) / 1e9 if len(t_ns) else np.zeros(0)
    return (s >= start_s) & (s <= (np.inf if end_s is None else end_s))


def held(path: str | Path, *, start_s: float = 0.0, end_s: float | None = None,
         engine_pid: int | None = None) -> OthersHeld:
    """What the processes other than the engine held over the recording at
    `path`, from `start_s` to `end_s` seconds after its first sample. The
    engine is the pid given, or the one in the hold's status beside the
    recording; without either, nothing is taken off."""
    path = Path(path)
    _, samples = recorder.read(path)
    engine_pid = _engine_pid(path, engine_pid)
    t, used = samples["t_mono_ns"], samples["used_bytes"]
    keep = _window(t, start_s, end_s)
    if engine_pid is not None:
        engine, known = _engine_bytes(path, t, used, engine_pid)
        used, keep = used - engine, keep & known
    if keep.sum() < 2:
        raise ValueError(f"{path} has fewer than two samples from {start_s} to {end_s} s"
                         + ("" if engine_pid is None else " that a process sample brackets"))
    used = used[keep]
    above = used - used.min()
    headroom = int(np.median(samples["free_bytes"][keep] + above))
    return OthersHeld((t[keep] - t[keep][0]) / 1e9, above, engine_pid, headroom, t[keep])


#: The span a with-engine scenario labels its engine's prefill with
#: (tools/run_scenario.py): the engine grows through it.
ENGINE_PREFILL = "engine-prefill"


def scenario_start_s(path: str | Path) -> float:
    """When the scenario of the recording at `path` began its first span
    after the engine's prefill had ended, in seconds from the recording's
    first sample: from there the engine holds what it holds. 0 without
    labels, or a prefill."""
    labels_path = recorder.labels_path(path)
    if not labels_path.exists():
        return 0.0
    _, labels = recorder.read_labels(labels_path)
    _, samples = recorder.read(path)
    t, event, action = labels["t_mono_ns"], labels["event"], labels["action"]
    prefilled = t[(event == recorder.END) & (action == ENGINE_PREFILL)]
    after = int(prefilled.max()) if len(prefilled) else int(samples["t_mono_ns"][0])
    starts = t[(event == recorder.START) & (action != ENGINE_PREFILL) & (t >= after)]
    return (int(starts.min() if len(starts) else after) - int(samples["t_mono_ns"][0])) / 1e9


def engine_residual(path: str | Path, *, start_s: float = 0.0, end_s: float | None = None,
                    engine_pid: int | None = None) -> dict | None:
    """How far what held() leaves once the engine is taken off strays from
    the sum of the other processes the 5 Hz stream lists, at each of its
    samples in the window: the difference less its median, the driver's own
    memory, in MiB as median, P90 and max. A check on the engine's
    subtraction, on the recording itself; None where nothing is taken off.
    Samples with a process the driver did not report are left out."""
    path = Path(path)
    engine_pid = _engine_pid(path, engine_pid)
    if engine_pid is None:
        return None
    _, samples = recorder.read(path)
    t, used = samples["t_mono_ns"], samples["used_bytes"]
    engine, known = _engine_bytes(path, t, used, engine_pid)
    _, states, procs = recorder.read_processes(recorder.processes_path(path))
    times = states["t_mono_ns"]
    others = procs[procs["pid"] != engine_pid]
    at = np.searchsorted(times, others["t_mono_ns"])
    listed = np.zeros(len(times), dtype=np.int64)
    np.add.at(listed, at, np.maximum(others["used_bytes"], 0))
    unreported = np.zeros(len(times), dtype=bool)
    unreported[at[others["used_bytes"] < 0]] = True
    # The device sample nearest the middle of each process query.
    middle = times + states["query_ns"] // 2
    after = np.clip(np.searchsorted(t, middle), 1, len(t) - 1)
    nearest = np.where(middle - t[after - 1] <= t[after] - middle, after - 1, after)
    rel = (times - t[0]) / 1e9
    take = (~unreported & known[nearest] & (rel >= start_s)
            & (rel <= (np.inf if end_s is None else end_s)))
    if not take.any():
        return None
    difference = (used - engine)[nearest[take]] - listed[take]
    return spread((np.abs(difference - np.median(difference)) / MIB).tolist())


def to_schedule(h: OthersHeld, base_bytes: int = 0) -> Schedule:
    """The simulator's schedule for `h`: each sample's level, on `base_bytes`
    taken throughout, in whole granules, from its time until the next's, and
    then 0, one period after the last."""
    end = round(float(h.t_s[-1] + np.median(np.diff(h.t_s))), 6)
    return Schedule(patterns.sampled(h.t_s, h.bytes + base_bytes).points + [(end, 0)])


def from_recording(path: str | Path, **window) -> Schedule:
    """The schedule that replays the recording at `path` (see held)."""
    return to_schedule(held(path, **window))


# -- replaying -----------------------------------------------------------------------------


def replay(schedule: Schedule, out: str | Path, *, lead_s: float = LEAD_S,
           tail_s: float = TAIL_S, stop: threading.Event | None = None) -> int:
    """Record the device to `out`, as RQ1 did, while the simulator, a process
    of its own, takes `schedule` after `lead_s` of taking nothing; stop
    `tail_s` after its last point. The schedule, the simulator's events, and
    when the schedule began with the simulator's pid (.replay.json) are
    written beside `out`. Returns when the schedule began, on the
    recorder's clock."""
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
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
        deadline = time.monotonic() + shifted.duration_s + REPLAY_SLACK_S
        while not _applied(events_path, len(shifted.points)):
            if failure or proc.poll() is not None or time.monotonic() > deadline or (
                    stop is not None and stop.is_set()):
                raise RuntimeError("the replay stopped before its schedule ended" +
                                   (f": {failure[0]}" if failure else ""))
            time.sleep(POLL_S)
        (stop or threading.Event()).wait(tail_s)
    finally:
        if proc.poll() is None:
            proc.send_signal(signal.SIGTERM)
        try:
            proc.wait(timeout=EXIT_WAIT_S)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
        done.set()
        recording.join()
        shifted_path.unlink(missing_ok=True)
    if failure:
        raise failure[0]
    first = json.loads(events_path.read_text().splitlines()[0])
    started = first["scheduled_ns"] + int(lead_s * 1e9)
    recorder.companion(out, ".replay.json").write_text(json.dumps(
        {"started_ns": started, "simulator_pid": proc.pid, "lead_s": lead_s}) + "\n")
    return started


def simulator_context_bytes(tmp: str | Path) -> int:
    """What the simulator has before it takes anything: its CUDA context,
    by NVML's account of its process. Measured by starting it once on a
    schedule of nothing, with its schedule written in `tmp`. A caller that
    sets a base from headroom read before the simulator starts must leave
    this out too: the context is taken as it starts."""
    path = Path(tmp) / "nothing.schedule.json"
    Schedule([(0.0, 0)]).save(path)
    proc = subprocess.Popen([sys.executable, str(SIMULATOR), "--schedule", str(path)],
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        if proc.stdout.readline().strip() != "ready":
            raise RuntimeError(f"the simulator did not start: {proc.stderr.read()}")
        return next((p.used_bytes or 0 for p in nvml.processes() if p.pid == proc.pid), 0)
    finally:
        proc.send_signal(signal.SIGTERM)
        try:
            proc.wait(timeout=EXIT_WAIT_S)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
        path.unlink(missing_ok=True)


def files(out: str | Path) -> list[Path]:
    """Every file replay() writes for the recording at `out`: the trace, its
    processes stream, the schedule, the simulator's events and when the
    schedule began."""
    out = Path(out)
    return [out, recorder.processes_path(out), recorder.companion(out, ".schedule.json"),
            recorder.companion(out, ".events.jsonl"), recorder.companion(out, ".replay.json")]


def _applied(events: Path, points: int) -> bool:
    return events.exists() and len(events.read_text().splitlines()) >= points


# -- comparing -----------------------------------------------------------------------------


def level_at(h: OthersHeld, s: np.ndarray) -> np.ndarray:
    """`h`'s level at each of `s` seconds, from its last sample at or before;
    0 before its first."""
    index = np.searchsorted(h.t_s, s, side="right") - 1
    return np.where(index >= 0, h.bytes[np.clip(index, 0, None)], 0)


def _measure(t_s: np.ndarray, level: np.ndarray, spike: spikes.Spike,
             window_s: float) -> tuple[float, float] | None:
    """`spike`'s amplitude and edge on the series (t_s, level), in bytes and
    seconds: the peak over its span above the median of the window before
    its rise, and the first time from its rise it reaches half of that,
    interpolated. None where the series has no sample before the rise or
    during the span."""
    rise, start = spike.rise_start_ns / 1e9, spike.start_ns / 1e9
    stop = spike.end_ns / 1e9 if spike.end_ns is not None else start + window_s
    before = level[(t_s >= rise - window_s) & (t_s < rise)]
    span = (t_s >= rise) & (t_s <= stop)
    if not len(before) or not span.any():
        return None
    base = float(np.median(before))
    amplitude = float(level[span].max()) - base
    ts, ls = t_s[span], level[span] - base
    k = int(np.argmax(ls >= amplitude / 2))
    if k == 0:
        return amplitude, float(ts[0])
    t0, t1, l0, l1 = ts[k - 1], ts[k], ls[k - 1], ls[k]
    return amplitude, float(t0 + (t1 - t0) * (amplitude / 2 - l0) / (l1 - l0))


def _streams(processes, simulator_pid: int, started_ns: int, idle_s: float,
             original: OthersHeld) -> dict:
    """From the replay's processes stream: how far the simulator's own
    memory strayed from the schedule, and how far the rest of the desktop
    moved from its idle level, over the replay, in MiB."""
    states, procs = processes
    times = states["t_mono_ns"]
    at = np.searchsorted(times, procs["t_mono_ns"])
    mine = procs["pid"] == simulator_pid
    simulator = np.zeros(len(times), dtype=np.int64)
    desktop = np.zeros(len(times), dtype=np.int64)
    np.add.at(simulator, at[mine], np.maximum(procs["used_bytes"][mine], 0))
    np.add.at(desktop, at[~mine], np.maximum(procs["used_bytes"][~mine], 0))
    s = (times - started_ns) / 1e9
    idle = (s >= -idle_s) & (s < 0)
    during = (s >= 0) & (s <= original.t_s[-1])
    if not idle.any() or not during.any():
        return {"simulator_error_mib": None, "desktop_moved_mib": None}
    strayed = simulator[during] - np.median(simulator[idle]) - level_at(original, s[during])
    moved = desktop[during] - np.median(desktop[idle])
    return {"simulator_error_mib": spread((np.abs(strayed) / MIB).tolist()),
            "desktop_moved_mib": spread((np.abs(moved) / MIB).tolist())}


def compare(original: OthersHeld, samples: np.ndarray, started_ns: int, *,
            idle_s: float = IDLE_S, processes=None, simulator_pid: int | None = None) -> dict:
    """The recording of a replay, its device `samples`, against the
    `original` it replayed from `started_ns`. The desktop's own level is the
    median used memory over the `idle_s` before the start; what the replay
    took is the used memory above it, so whatever the desktop itself does
    over the replay counts as error: replay windows of a few minutes, around
    the actions of interest, rather than whole recordings. `processes`, the
    replay's (states, procs), with the simulator's pid, adds the simulator's
    own error and the desktop's movement. JSON as it stands, for the log."""
    t = samples["t_mono_ns"]
    idle = samples["used_bytes"][(t >= started_ns - idle_s * 1e9) & (t < started_ns)]
    if not len(idle):
        raise ValueError("the replay's recording has no samples before its schedule began, "
                         "to measure the desktop's own level by")
    base = int(np.median(idle))
    s = (t - started_ns) / 1e9
    during = (s >= 0) & (s <= original.t_s[-1])
    taken = samples["used_bytes"] - base
    error = np.abs(taken[during] - level_at(original, s[during])) / MIB

    window_s = spikes.DEFAULT_WINDOW_S
    found = spikes.find_spikes((original.t_s * 1e9).astype(np.int64), -original.bytes)
    starts, amplitudes, unmeasured = [], [], 0
    for spike in found:
        was = _measure(original.t_s, original.bytes, spike, window_s)
        now = _measure(s, taken, spike, window_s)
        if was is None or now is None:
            unmeasured += 1
            continue
        amplitudes.append(abs(now[0] - was[0]) / MIB)
        starts.append(abs(now[1] - was[1]) * 1e3)
    error_mib = spread(error.tolist())
    results = {
        "samples": int(during.sum()), "idle_used_mib": base / MIB,
        "error_mib": error_mib,
        "spikes": {"original": len(found), "unmeasured": unmeasured,
                   "edge_offset_ms": spread(starts),
                   "amplitude_error_mib": spread(amplitudes)},
        "bounds": {"error_p90_mib": ERROR_BOUND_MIB, "start_ms": START_BOUND_MS,
                   "amplitude_mib": AMPLITUDE_BOUND_MIB},
        "within": {
            "error": error_mib is not None and error_mib["p90"] <= ERROR_BOUND_MIB,
            "spikes": unmeasured == 0,
            "start": all(x <= START_BOUND_MS for x in starts),
            "amplitude": all(x <= AMPLITUDE_BOUND_MIB for x in amplitudes)},
    }
    if processes is not None and simulator_pid is not None:
        results.update(_streams(processes, simulator_pid, started_ns, idle_s, original))
    return results
