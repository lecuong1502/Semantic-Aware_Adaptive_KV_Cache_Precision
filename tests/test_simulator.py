"""The contention simulator: device memory taken on a schedule of (time,
bytes taken), by a process of its own (#58).

It is the other process a pressure monitor must see: its own CUDA context,
memory only, a granule at a time through the engine's VMM allocator, so that
memory given back reaches the driver at once. Tested in process for what it
takes, and as a process for what NVML sees of it from outside.
"""

import contextlib
import json
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

from conftest import each
from microinfer import _microinfer, nvml, simulator

REPO = Path(__file__).resolve().parent.parent
TOOL = REPO / "tools" / "simulate_contention.py"
GRANULE = _microinfer.granule_bytes()
MiB = 2**20


def test_a_schedule_is_steps_in_time_the_last_kept_read_and_written_whole(tmp_path):
    """Bytes taken from each point until the next, and the last point's from
    then on; the times start at 0 and increase, and nothing is negative. It
    round-trips through JSON."""
    schedule = simulator.Schedule([(0.0, 0), (0.5, 64 * MiB), (1.25, 300 * MiB), (2.0, 32 * MiB)])
    assert schedule.at(0.4) == 0 and schedule.at(0.5) == 64 * MiB
    assert schedule.at(1.9) == 300 * MiB and schedule.at(99) == 32 * MiB
    assert schedule.duration_s == 2.0 and schedule.peak_bytes == 300 * MiB
    path = tmp_path / "s.json"
    schedule.save(path)
    assert simulator.Schedule.load(path) == schedule

    def refused(points):
        with pytest.raises(ValueError):
            simulator.Schedule(points)

    each([([],), ([(0.5, 0)],), ([(0, 0), (0, 1)],), ([(0, 0), (1, -1)],)], refused)


def test_a_reservation_takes_whole_granules_and_nvml_sees_them_and_their_return():
    """Asked for any number of bytes it takes the nearest whole number of
    granules, up to its capacity, and this process's memory by NVML follows
    it to within one granule, going up and coming down; released, it has
    taken nothing."""
    base = nvml.settled_own_used_bytes()
    reservation = simulator.Reservation()

    def takes(target):
        taken = reservation.take(target)
        assert taken % GRANULE == 0 and abs(taken - target) <= GRANULE / 2
        assert abs(nvml.own_used_bytes() - base - taken) <= GRANULE

    try:
        each([64 * MiB, 3 * GRANULE + 1, 300 * MiB, 10 * MiB, 0], takes)
        reservation.take(100 * MiB)
    finally:
        reservation.release()
    assert abs(nvml.own_used_bytes() - base) <= GRANULE

    small = simulator.Reservation(capacity_bytes=4 * GRANULE)
    assert small.take(100 * MiB) == 4 * GRANULE  # its capacity, not a crash
    small.release()
    with pytest.raises(ValueError, match="granule"):
        simulator.Reservation(capacity_bytes=0)


def test_running_a_schedule_keeps_its_times_its_last_level_and_its_latencies():
    """Each point is applied at its time from the start, however long the one
    before took, and the last level is kept until stopped; each change
    reports the bytes asked and taken, and its latency until the calls
    returned and until an NVML reading showed it, that reading included."""
    schedule = simulator.Schedule([(0.0, 0), (0.2, 64 * MiB), (0.5, 192 * MiB), (0.8, 32 * MiB)])
    reservation = simulator.Reservation()
    stop = threading.Event()
    threading.Timer(1.3, stop.set).start()
    started = time.monotonic()
    try:
        events = simulator.run(schedule, reservation, stop)
        assert time.monotonic() - started >= 1.2  # the last level, kept until stopped
        assert reservation.taken_bytes == 32 * MiB
    finally:
        reservation.release()
    assert [e["target_bytes"] for e in events] == [0, 64 * MiB, 192 * MiB, 32 * MiB]
    offsets = [(e["scheduled_ns"] - events[0]["scheduled_ns"]) / 1e9 for e in events]
    assert offsets == pytest.approx([0.0, 0.2, 0.5, 0.8], abs=1e-6)
    for e in events:
        assert e["seen"] and e["taken_bytes"] == e["target_bytes"] and e["shortfall_bytes"] == 0
        assert 0 <= e["late_s"] < 0.2 and e["api_ms"] < e["nvml_ms"] < 1000


def test_a_calibration_measures_taking_and_giving_back_per_size():
    """Per size and direction, every change made, how late it began, and how
    long until the calls returned and until NVML showed it."""
    results = simulator.calibrate(2, sizes=(GRANULE, 64 * MiB), step_s=0.15)
    assert set(results) == {str(GRANULE), str(64 * MiB)}
    for directions in results.values():
        for stats in directions.values():
            assert stats["changes"] == 2 and stats["unseen"] == 0
            assert stats["api_ms"]["median"] < stats["nvml_ms"]["median"]
            assert stats["nvml_ms"]["median"] <= stats["nvml_ms"]["max"] < 1000


@contextlib.contextmanager
def simulating(tmp_path, points, name="events.jsonl"):
    """The simulator as a process, on `points`, once it says it is ready; it
    is never left behind."""
    schedule = tmp_path / "schedule.json"
    simulator.Schedule(points).save(schedule)
    proc = subprocess.Popen([sys.executable, str(TOOL), "--schedule", str(schedule),
                             "--events", str(tmp_path / name)],
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        assert proc.stdout.readline().strip() == "ready", proc.stderr.read()
        yield proc
    finally:
        if proc.poll() is None:
            proc.kill()
        proc.communicate(timeout=10)


def taken_by(pid):
    return next((p.used_bytes for p in nvml.processes() if p.pid == pid), None)


def wait_until(condition, timeout=30.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if condition():
            return True
        time.sleep(0.05)
    return False


def test_as_a_process_it_is_another_gpu_process_that_gives_everything_back(tmp_path):
    """From outside, NVML sees the simulator as a process of its own, with
    what its schedule says; stopped, it logs every change and exits; killed,
    whether with its memory taken or while taking it, everything it took is
    gone with it."""
    with simulating(tmp_path, [(0.0, 0), (0.5, 200 * MiB)]) as proc:
        assert proc.pid != os.getpid()
        events = tmp_path / "events.jsonl"
        assert wait_until(lambda: len(events.read_text().splitlines()) >= 2)
        context = json.loads(events.read_text().splitlines()[0])["nvml_bytes"]
        assert abs(taken_by(proc.pid) - context - 200 * MiB) <= GRANULE
        proc.send_signal(signal.SIGTERM)
        assert proc.wait(timeout=30) == 0
    assert [json.loads(line)["target_bytes"] for line in events.read_text().splitlines()] == [
        0, 200 * MiB]
    assert wait_until(lambda: taken_by(proc.pid) is None, 10)

    # Killed while taking a gigabyte a granule at a time, and after.
    for kill_after in (0.03, 1.0):
        with simulating(tmp_path, [(0.0, 1024 * MiB)], name=f"kill-{kill_after}.jsonl") as proc:
            time.sleep(kill_after)
            proc.send_signal(signal.SIGKILL)
            proc.wait(timeout=30)
        assert wait_until(lambda: taken_by(proc.pid) is None, 10), kill_after
