"""The contention simulator: device memory held on a schedule of (time, bytes
held), by a process of its own (#58).

It is the other process a pressure monitor must see: its own CUDA context,
memory only, one-granule pages through the engine's VMM allocator, so that a
release reaches the driver at once. Tested in process for what it holds, and
as a process for what NVML sees of it from outside.
"""

import json
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

from microinfer import _microinfer, nvml, simulator

REPO = Path(__file__).resolve().parent.parent
TOOL = REPO / "tools" / "simulate_contention.py"
GRANULE = _microinfer.granule_bytes()
MiB = 2**20


def test_a_schedule_is_steps_in_time_read_and_written_whole(tmp_path):
    """Bytes held from each point until the next; the times start at 0 and
    increase, and nothing is negative. It round-trips through JSON."""
    schedule = simulator.Schedule([(0.0, 0), (0.5, 64 * MiB), (1.25, 300 * MiB), (2.0, 0)])
    assert schedule.at(0.4) == 0 and schedule.at(0.5) == 64 * MiB
    assert schedule.at(1.9) == 300 * MiB and schedule.at(99) == 0
    assert schedule.duration_s == 2.0 and schedule.peak_bytes == 300 * MiB
    path = tmp_path / "s.json"
    schedule.save(path)
    assert simulator.Schedule.load(path) == schedule
    for bad in ([], [(0.5, 0)], [(0, 0), (0, 1)], [(0, 0), (1, -1)]):
        with pytest.raises(ValueError):
            simulator.Schedule(bad)


def test_the_holder_holds_whole_granules_and_nvml_sees_them_and_their_return():
    """Asked for any number of bytes it holds the nearest whole number of
    granules, and this process's memory by NVML follows it to within one
    granule, going up and coming down; released, it holds nothing."""
    base = nvml.settled_own_used_bytes()
    holder = simulator.Holder()
    try:
        for target in (64 * MiB, 3 * GRANULE + 1, 700 * MiB, 10 * MiB, 0):
            held = holder.hold(target)
            assert held % GRANULE == 0 and abs(held - target) <= GRANULE / 2, target
            assert abs(nvml.own_used_bytes() - base - held) <= GRANULE, target
        holder.hold(200 * MiB)
    finally:
        holder.release()
    assert abs(nvml.own_used_bytes() - base) <= GRANULE


def test_running_a_schedule_keeps_its_times_and_says_how_long_each_change_took():
    """Each point is applied at its time from the start, however long the one
    before took; each change reports the bytes asked and held, and the
    latency until the call returned and until NVML showed it."""
    schedule = simulator.Schedule([(0.0, 0), (0.2, 128 * MiB), (0.5, 512 * MiB), (0.8, 0)])
    holder = simulator.Holder()
    try:
        events = simulator.run(schedule, holder, threading.Event())
    finally:
        holder.release()
    assert [e["target_bytes"] for e in events] == [0, 128 * MiB, 512 * MiB, 0]
    offsets = [(e["scheduled_ns"] - events[0]["scheduled_ns"]) / 1e9 for e in events]
    assert offsets == pytest.approx([0.0, 0.2, 0.5, 0.8], abs=1e-6)
    for e in events:
        assert 0 <= e["late_s"] < 0.05 and e["held_bytes"] == e["target_bytes"]
        assert 0 <= e["api_ms"] <= e["nvml_ms"] < 1000


def run_tool(*args):
    return subprocess.Popen([sys.executable, str(TOOL), *args], stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, text=True)


def held_by(pid):
    return next((p.used_bytes for p in nvml.processes() if p.pid == pid), None)


def test_as_a_process_it_is_another_gpu_process_that_gives_everything_back(tmp_path):
    """From outside, NVML sees the simulator as a process of its own, holding
    what its schedule says; stopped, it logs every change and exits; killed,
    everything it held is gone with it."""
    schedule = tmp_path / "s.json"
    simulator.Schedule([(0.0, 0), (1.0, 400 * MiB), (60.0, 0)]).save(schedule)
    events = tmp_path / "events.jsonl"
    proc = run_tool("--schedule", str(schedule), "--events", str(events))
    try:
        end = time.monotonic() + 30
        while time.monotonic() < end and not (events.exists() and len(
                events.read_text().splitlines()) >= 2):
            time.sleep(0.1)
        used = held_by(proc.pid)
        assert used is not None and proc.pid != __import__("os").getpid()
        context = json.loads(events.read_text().splitlines()[0])["nvml_bytes"]
        assert abs(used - context - 400 * MiB) <= GRANULE
        proc.send_signal(signal.SIGTERM)
        assert proc.wait(timeout=30) == 0
    finally:
        if proc.poll() is None:
            proc.kill()
        proc.communicate(timeout=10)
    lines = [json.loads(line) for line in events.read_text().splitlines()]
    assert [e["target_bytes"] for e in lines] == [0, 400 * MiB]
    assert held_by(proc.pid) is None

    proc = run_tool("--schedule", str(schedule), "--events", str(tmp_path / "e2.jsonl"))
    try:
        end = time.monotonic() + 30
        while time.monotonic() < end and (held_by(proc.pid) or 0) < 400 * MiB:
            time.sleep(0.1)
        assert held_by(proc.pid) >= 400 * MiB
        proc.send_signal(signal.SIGKILL)
        proc.wait(timeout=30)
    finally:
        if proc.poll() is None:
            proc.kill()
        proc.communicate(timeout=10)
    end = time.monotonic() + 10
    while time.monotonic() < end and held_by(proc.pid) is not None:
        time.sleep(0.1)
    assert held_by(proc.pid) is None


def test_a_calibration_measures_holding_and_releasing():
    """Per direction, every change made, and how long until NVML showed it."""
    results = simulator.calibrate(2, sizes=(GRANULE, 64 * MiB), step_s=0.15)
    for direction in ("hold", "release"):
        stats = results[direction]
        assert stats["changes"] == 4 and stats["unseen"] == 0
        assert 0 <= stats["api_ms"]["median"] <= stats["nvml_ms"]["median"]
        assert stats["nvml_ms"]["median"] <= stats["nvml_ms"]["max"] < 1000
