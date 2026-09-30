"""Replaying a recorded trace on the contention simulator (#60).

A recording becomes the schedule of what the processes other than the
engine held, sample by sample; a replay is recorded as RQ1 recorded, and
compared with the original, sample by sample and spike by spike. Tested on a
recording written here, whose every byte is known, and by replaying one on
the device.
"""

import json

import numpy as np
import pytest

from microinfer import recorder, replay

MiB = 2**20
PERIOD_NS = 20_000_000  # 50 Hz
TOTAL = 6 * 2**30


def others(t_s):
    """What the other processes hold in the recording written here: 500 MiB,
    and 200 MiB more from 8 s to 10 s, rising over 100 ms."""
    t_s = np.asarray(t_s, dtype=float)
    return (500 * MiB + 200 * MiB * np.clip((t_s - 8.0) / 0.1, 0, 1) * (t_s < 10.0)).astype(
        np.int64)


def engine(t_s):
    """The engine: 1000 MiB, then 1100 from 3.9 s, between two 5 Hz samples."""
    return np.where(np.asarray(t_s) >= 3.9, 1100 * MiB, 1000 * MiB)


def write_recording(path, seconds=20.0, t0_ns=10**12):
    """A with-engine recording: device samples at 50 Hz, the processes at
    5 Hz, and the hold's status naming the engine's pid."""
    t = t0_ns + np.arange(int(seconds * 50)) * PERIOD_NS
    used = others((t - t0_ns) / 1e9) + engine((t - t0_ns) / 1e9) + 400 * MiB  # the driver's own
    out = recorder._TraceWriter(path, recorder.FORMAT, {}, recorder.COLUMNS)
    for ti, u in zip(t, used):
        out.row(int(ti), 0.0, TOTAL - int(u), int(u), 1000)
    out.close({"samples": len(t), "missed": 0})
    procs = recorder._TraceWriter(recorder.processes_path(path), recorder.PROCESS_FORMAT, {},
                                  ("row", "fields..."))
    for ti in t[::10]:
        s = (ti - t0_ns) / 1e9
        procs.row(recorder.STATE_ROW, int(ti), 0.0, 1000, 0, 0, 0, 0)
        procs.row(recorder.PROCESS_ROW, int(ti), 0.0, 42, "compute", int(engine(s)), "python")
        procs.row(recorder.PROCESS_ROW, int(ti), 0.0, 7, "graphics", int(others(s)), "chrome")
    procs.close({"samples": len(t) // 10, "missed": 0, "process_rows": len(t) // 5})
    recorder.companion(path, ".hold.json").write_text(json.dumps({"pid": 42}))


def replay_samples(held, started_ns, idle_bytes=700 * MiB, lead_s=2.0, phase_ns=7_000_000):
    """The device samples a perfect replay of `held` would record: an idle
    desktop's `idle_bytes`, and from `started_ns` what the schedule takes."""
    t = started_ns - int(lead_s * 1e9) + phase_ns + np.arange(
        int((lead_s + held.t_s[-1] + 1) * 50)) * PERIOD_NS
    used = idle_bytes + replay.level_at(held, (t - started_ns) / 1e9)
    samples = np.zeros(len(t), dtype=recorder.DEVICE_DTYPE)
    samples["t_mono_ns"], samples["used_bytes"], samples["free_bytes"] = t, used, TOTAL - used
    return samples


def test_a_recording_becomes_what_the_others_held_and_a_replay_is_compared_with_it(tmp_path):
    """The device's used memory less the engine's, each change of the
    engine's taken where the device showed it, between two 5 Hz samples,
    above its least in the window: the others' 200 MiB spike, not the
    engine's growth. The schedule is that in whole granules, from the
    window's start, ending at 0. A replay that took it matches within the
    bounds; one that missed the spike does not."""
    path = tmp_path / "rec.csv.gz"
    write_recording(path)

    held = replay.held(path, start_s=2.0, end_s=15.0)
    assert held.t_s[0] == 0.0 and held.t_s[-1] == pytest.approx(13.0, abs=0.02)
    assert np.array_equal(held.bytes, others(np.round(held.t_s + 2.0, 6)) - 500 * MiB)

    schedule = replay.to_schedule(held)
    assert schedule.points[0] == (0.0, 0) and schedule.points[-1][1] == 0
    assert schedule.at(7.0) == 200 * MiB and schedule.at(5.9) == 0 and schedule.at(8.1) == 0
    assert schedule.duration_s == pytest.approx(13.02, abs=0.001)
    on_base = replay.to_schedule(held, base_bytes=1024 * MiB)
    assert on_base.at(0.0) == 1024 * MiB and on_base.at(7.0) == 1224 * MiB
    assert on_base.points[-1] == (schedule.duration_s, 0)

    everything = replay.held(path)  # no window; the engine's pid from the hold's status
    assert everything.bytes.min() == 0 and everything.bytes.max() == 200 * MiB
    without_engine = replay.held(path, engine_pid=0)  # a pid that is not there subtracts nothing
    assert without_engine.bytes.max() == 300 * MiB  # its 100 MiB growth, and the spike

    started = 5 * 10**12
    good = replay.compare(held, replay_samples(held, started), started)
    assert all(good["within"].values()), good
    assert good["spikes"]["original"] == 1 and good["spikes"]["unmeasured"] == 0
    assert good["spikes"]["amplitude_error_mib"]["max"] == 0
    assert good["error_mib"]["median"] == 0

    flat = replay.OthersHeld(held.t_s, np.zeros_like(held.bytes))
    missed = replay.compare(held, replay_samples(flat, started), started)
    assert not missed["within"]["amplitude"]
    assert missed["spikes"]["amplitude_error_mib"]["max"] == pytest.approx(200, abs=1)

    with pytest.raises(ValueError, match="before"):  # no idle lead to measure the desktop by
        replay.compare(held, replay_samples(held, started, lead_s=0.0), started)
    with pytest.raises(ValueError, match="ends after it starts"):
        replay.held(path, start_s=12.0, end_s=11.0)
    # The processes stream agrees with what is left once the engine is off.
    assert replay.engine_residual(path)["max"] == 0

    # The headroom the original had with the others at their least, the
    # engine as it stood: 6 GiB less 1100 MiB, 500 and the driver's 400.
    assert replay.held(path, start_s=5.0).headroom_bytes == TOTAL - 2000 * MiB
    # Where a scenario begins: its first span after the engine's prefill.
    labels = recorder._TraceWriter(recorder.labels_path(path), recorder.LABEL_FORMAT, {},
                                   recorder.LABEL_COLUMNS)
    for at_s, event, action in ((0.0, recorder.START, replay.ENGINE_PREFILL),
                                (1.5, recorder.END, replay.ENGINE_PREFILL),
                                (1.5, recorder.START, "idle"), (3.0, recorder.END, "idle")):
        labels.row(10**12 + int(at_s * 1e9), 0.0, event, action)
    labels.close({"labels": 4, "rejected": 0})
    assert replay.scenario_start_s(path) == pytest.approx(1.5)


def test_a_replay_on_the_device_is_recorded_with_the_schedule_in_place(tmp_path):
    """The simulator, a process of its own, takes the schedule while the
    recorder records; the recording of the replay shows its step when the
    schedule says. Only that is asserted: how well a replay matches is
    measured on an idle desktop, and logged (tools/replay_contention.py)."""
    out = tmp_path / "new" / "replay.csv.gz"  # its directory made as it is written
    steps = np.arange(100)
    held = replay.OthersHeld(steps * 0.02, np.where(steps >= 25, 256 * MiB, 0).astype(np.int64))
    started = replay.replay(replay.to_schedule(held), out, lead_s=1.0, tail_s=0.5)
    # The lead's two points, the step, and the end.
    assert len(recorder.companion(out, ".events.jsonl").read_text().splitlines()) == 4
    _, samples = recorder.read(out)
    s = (samples["t_mono_ns"] - started) / 1e9
    used = samples["used_bytes"]
    before, during = np.median(used[(s > 0) & (s < 0.45)]), np.median(used[(s > 0.7) & (s < 1.95)])
    assert abs(during - before - 256 * MiB) <= replay.AMPLITUDE_BOUND_MIB * MiB
    # The simulator's own memory, by the processes stream, is the schedule's.
    pid = json.loads(recorder.companion(out, ".replay.json").read_text())["simulator_pid"]
    _, states, procs = recorder.read_processes(recorder.processes_path(out))
    result = replay.compare(held, samples, started, processes=(states, procs),
                            simulator_pid=pid)
    assert result["simulator_error_mib"]["p90"] <= 2
