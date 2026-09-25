"""The contention recorder: device-wide memory at 50 Hz (#46), who holds it
at 5 Hz (#47), and the actions a scenario marks while it runs (#48).

RQ1's instrument. It samples the driver's account of free and used device
memory on a fixed schedule and writes a compressed CSV that can be read back
even if the recorder is killed partway through. It runs as a process of its
own and takes no device memory: NVML needs no CUDA context.

The tests that need no GPU drive the sampling loop with an injected reader,
so what they check is the loop and the file, not the driver. Those that read
NVML check what only the real thing can show: the rate this machine sustains,
and that the recorder holds nothing on the device.
"""

import contextlib
import gzip
import math
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np
import pytest

from microinfer import nvml, recorder

REPO = Path(__file__).resolve().parent.parent
TOOL = REPO / "tools" / "record_contention.py"
LABEL_TOOL = REPO / "tools" / "label_contention.py"


class FakeMemory:
    """A reader whose free memory falls by one MiB a sample."""

    total = 6 * 2**30

    def __init__(self):
        self.calls = 0

    def __call__(self):
        self.calls += 1
        free = self.total - self.calls * 2**20
        return free, self.total - free


def test_a_recording_reads_back_what_was_sampled(tmp_path):
    """Samples, metadata whatever its keys, and the closing line's counts all
    come back. A caller's own reader records no processes stream unless it
    brings a process reader too."""
    path = tmp_path / "trace.csv.gz"
    reader = FakeMemory()
    stats = recorder.record(path, rate_hz=200, duration=0.25, reader=reader,
                            meta={"total_bytes": reader.total, "completed_by": "someone",
                                  "note": "a=b"})
    meta, samples = recorder.read(path)
    assert meta["rate_hz"] == "200" and meta["completed_by"] == "someone"
    assert meta["note"] == "a=b" and meta["complete"]
    assert len(samples) == stats["samples"] == meta["samples"] == reader.calls
    assert meta["missed"] == stats["missed"]
    assert list(samples.dtype.names) == list(recorder.COLUMNS)
    np.testing.assert_array_equal(samples["free_bytes"],
                                  reader.total - np.arange(1, reader.calls + 1) * 2**20)
    assert np.all(samples["free_bytes"] + samples["used_bytes"] == reader.total)
    assert np.all(np.diff(samples["t_mono_ns"]) > 0)
    assert np.all(samples["query_ns"] >= 0)
    assert "processes" not in stats and not recorder.processes_path(path).exists()


def test_the_schedule_holds_its_rate_without_drift():
    """Deadlines are absolute: a slow sample delays only itself. Over a second
    at 100 Hz the count is the rate times the time, and the mean period is
    the target, however long each read took."""
    def slow_reader():
        time.sleep(0.003)
        return 1, 1

    stats = recorder.record(None, rate_hz=100, duration=1.0, reader=slow_reader)
    assert 97 <= stats["samples"] <= 101
    assert abs(stats["period_ms"]["mean"] - 10.0) < 0.3


def test_every_deadline_is_either_sampled_or_missed_and_none_past_the_end():
    """A reader slower than the period misses deadlines, and a read that
    overruns the end must not count the deadlines after it: over 0.1 s at
    100 Hz there are ten deadlines, and each is either a sample or a miss."""
    def slower_than_period():
        time.sleep(0.025)
        return 1, 1

    rate, duration = 100, 0.1
    stats = recorder.record(None, rate_hz=rate, duration=duration, reader=slower_than_period)
    assert stats["missed"] > 0
    assert stats["samples"] + stats["missed"] == math.ceil(rate * duration)


def test_only_a_truncated_recording_reads_as_incomplete(tmp_path):
    """A recording cut short reads back as incomplete; a file that is missing,
    or is not a recording of the kind asked for, is an error, not an
    interrupted recorder."""
    with pytest.raises(FileNotFoundError):
        recorder.read(tmp_path / "absent.csv.gz")
    (tmp_path / "plain.txt").write_text("not gzip\n")
    with pytest.raises(OSError):
        recorder.read(tmp_path / "plain.txt")
    with gzip.open(tmp_path / "other.csv.gz", "wt") as f:
        f.write("a,b,c\n1,2,3\n")
    with pytest.raises(ValueError, match="not a contention trace"):
        recorder.read(tmp_path / "other.csv.gz")
    path = tmp_path / "trace.csv.gz"
    recorder.record(path, rate_hz=100, duration=0.1, reader=FakeMemory(),
                    process_reader=FakeProcesses())
    with pytest.raises(ValueError, match="not a contention trace"):
        recorder.read(recorder.processes_path(path))
    with pytest.raises(ValueError, match="not a contention trace"):
        recorder.read_processes(path)


def test_a_stop_event_ends_the_recording_cleanly(tmp_path):
    path = tmp_path / "trace.csv.gz"
    stop = threading.Event()
    threading.Timer(0.2, stop.set).start()
    recorder.record(path, rate_hz=100, stop=stop, reader=FakeMemory(),
                    process_reader=FakeProcesses())
    meta, samples = recorder.read(path)
    assert meta["complete"] and 15 <= len(samples) <= 25
    assert recorder.read_processes(recorder.processes_path(path))[0]["complete"]


# -- the processes stream -------------------------------------------------------------


def gpu_process(pid, name, used=2**20, kind="graphics"):
    return nvml.GpuProcess(pid=pid, name=name, kind=kind, used_bytes=used)


class FakeProcesses:
    """A process reader whose GPU gains a process at the third sample and
    loses it at the sixth, whose clocks rise with the sample count, and
    whose P-state is unreported at every other sample."""

    def __init__(self):
        self.calls = 0

    def __call__(self):
        self.calls += 1
        k = self.calls
        procs = [gpu_process(100, "/usr/bin/Xorg"),
                 gpu_process(200, 'a, name with "commas"', used=None, kind="compute")]
        if 3 <= k < 6:
            procs.append(gpu_process(300, "game", used=512 * 2**20, kind="compute+graphics"))
        return recorder.ProcessSample(pstate=k % 16 if k % 2 else None, graphics_mhz=k,
                                      sm_mhz=2 * k, memory_mhz=3 * k, processes=procs)


def test_the_processes_stream_reads_back_every_sample(tmp_path):
    """Beside the device trace, on its clock: every sample leaves one state
    row and one row per process. A process that comes and goes is in exactly
    the samples that saw it; names keep their commas and quotes, and what the
    driver did not report reads as -1."""
    path = tmp_path / "trace.csv.gz"
    procs = FakeProcesses()
    stats = recorder.record(path, rate_hz=100, duration=0.5, reader=FakeMemory(),
                            meta={"device": "fake"}, processes_rate_hz=20,
                            process_reader=procs)
    assert recorder.processes_path(path) == tmp_path / "trace.procs.csv.gz"
    meta, states, rows = recorder.read_processes(recorder.processes_path(path))
    assert meta["complete"] and meta["device"] == "fake" and meta["rate_hz"] == "20"
    assert len(states) == procs.calls == stats["processes"]["samples"] == meta["samples"]
    assert meta["process_rows"] == len(rows)

    k = np.arange(1, procs.calls + 1)
    np.testing.assert_array_equal(states["pstate"], np.where(k % 2, k % 16, -1))
    np.testing.assert_array_equal(states["graphics_mhz"], k)
    np.testing.assert_array_equal(states["sm_mhz"], 2 * k)
    np.testing.assert_array_equal(states["memory_mhz"], 3 * k)

    named = rows[rows["pid"] == 200]
    assert len(named) == procs.calls and set(named["name"]) == {'a, name with "commas"'}
    assert set(named["kind"]) == {"compute"} and set(named["used_bytes"]) == {-1}
    game = rows[rows["pid"] == 300]
    np.testing.assert_array_equal(game["t_mono_ns"], states["t_mono_ns"][2:5])
    assert set(game["used_bytes"]) == {512 * 2**20}

    # On the device stream's clock: five device samples to a process sample.
    _, device = recorder.read(path)
    between = np.searchsorted(device["t_mono_ns"], states["t_mono_ns"])
    assert np.all(np.abs(np.diff(between) - 5) <= 1)


def failing_after(calls, reader):
    count = 0

    def read():
        nonlocal count
        count += 1
        if count > calls:
            raise RuntimeError("driver gone")
        return reader()
    return read


@pytest.mark.parametrize("failing", ["device", "processes"])
def test_a_failure_in_either_stream_ends_both_at_once(tmp_path, failing):
    """A stream that fails stops the recording within a period, not at its
    end, and neither file closes as complete: a whole trace beside a broken
    one would pass for a recording it is not."""
    path = tmp_path / "trace.csv.gz"
    device, procs = FakeMemory(), FakeProcesses()
    if failing == "device":
        device = failing_after(30, device)
    else:
        procs = failing_after(3, procs)
    started = time.monotonic()
    with pytest.raises(RuntimeError) as info:
        recorder.record(path, rate_hz=100, duration=30, reader=device,
                        processes_rate_hz=20, process_reader=procs)
    assert time.monotonic() - started < 1.0
    assert "driver gone" in str(info.value) + str(info.value.__cause__)
    assert not recorder.read(path)[0]["complete"]
    assert not recorder.read_processes(recorder.processes_path(path))[0]["complete"]


# -- action labels --------------------------------------------------------------------


def send_when_listening(fifo, event, action):
    recorder.send_label(fifo, event, action, wait=5.0)


def test_labels_sent_while_recording_mark_the_samples_they_span(tmp_path):
    """Labels sent from outside are stamped on the recorder's clock as they
    arrive, and reading back gives each sample the action in progress."""
    path, fifo = tmp_path / "trace.csv.gz", tmp_path / "labels.fifo"
    stop = threading.Event()

    def scenario():
        send_when_listening(fifo, "start", "open a browser, with a comma")
        time.sleep(0.2)
        recorder.send_label(fifo, "end", "open a browser, with a comma")
        time.sleep(0.1)
        recorder.send_label(fifo, "start", "idle")
        time.sleep(0.1)
        stop.set()

    runner = threading.Thread(target=scenario)
    runner.start()
    recorder.record(path, rate_hz=100, stop=stop, reader=FakeMemory(), labels=fifo)
    runner.join()
    assert not fifo.exists()  # made for the recording, and removed after it

    _, samples = recorder.read(path)
    meta, labels = recorder.read_labels(recorder.labels_path(path))
    assert meta["complete"] and meta["labels"] == 3
    assert list(labels["event"]) == ["start", "end", "start"]
    assert list(labels["action"]) == ["open a browser, with a comma"] * 2 + ["idle"]
    t = samples["t_mono_ns"]
    assert t[0] < labels["t_mono_ns"][0] and np.all(np.diff(labels["t_mono_ns"]) > 0)

    actions = recorder.actions_in_progress(t, labels)
    start, end, idle = labels["t_mono_ns"]
    assert set(actions[(t >= start) & (t < end)]) == {"open a browser, with a comma"}
    assert set(actions[t < start]) == set(actions[(t >= end) & (t < idle)]) == {""}
    assert set(actions[t >= idle]) == {"idle"}  # never ended: in progress to the last sample
    assert 15 <= np.sum(actions == "open a browser, with a comma") <= 25


def test_the_action_in_progress_is_the_latest_one_started_and_not_ended():
    labels = np.array([(10, 0.0, "start", "a"), (20, 0.0, "start", "b"),
                       (30, 0.0, "end", "b"), (40, 0.0, "end", "a")],
                      dtype=recorder.LABEL_DTYPE)
    t = np.array([5, 10, 15, 20, 25, 30, 35, 40, 45])
    assert list(recorder.actions_in_progress(t, labels)) == \
        ["", "a", "a", "b", "b", "a", "a", "", ""]
    unopened = np.array([(10, 0.0, "end", "a")], dtype=recorder.LABEL_DTYPE)
    with pytest.raises(ValueError, match="'a' ends without having started"):
        recorder.actions_in_progress(t, unopened)


def test_a_label_with_no_recorder_listening_is_an_error(tmp_path):
    with pytest.raises(recorder.NoRecorder):
        recorder.send_label(tmp_path / "labels.fifo", "start", "x")
    os.mkfifo(tmp_path / "stale.fifo")  # left behind by a killed recorder
    with pytest.raises(recorder.NoRecorder):
        recorder.send_label(tmp_path / "stale.fifo", "start", "x")
    with pytest.raises(ValueError):
        recorder.send_label(tmp_path / "stale.fifo", "begin", "x")
    with pytest.raises(ValueError):
        recorder.send_label(tmp_path / "stale.fifo", "start", "two\nlines")


# -- the recorder as a process, on the real driver ------------------------------------


@contextlib.contextmanager
def running_tool(path, *args):
    """The recorder as a process, never left behind: whatever happens in the
    block, it is stopped on the way out, and its stderr is shown if it failed."""
    proc = subprocess.Popen([sys.executable, str(TOOL), "--out", str(path), *args],
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        yield proc
    finally:
        if proc.poll() is None:
            proc.kill()
        _, err = proc.communicate(timeout=10)
        if proc.returncode not in (0, -signal.SIGKILL):
            print(err, file=sys.stderr)


def wait_for_samples(path, at_least, timeout=10.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if path.exists():
            with contextlib.suppress(ValueError):  # the header is not written yet
                _, samples = recorder.read(path)
                if len(samples) >= at_least:
                    return samples
        time.sleep(0.1)
    raise AssertionError(f"fewer than {at_least} samples in {timeout} s")


def test_a_killed_recorder_leaves_a_readable_file(tmp_path):
    """Killed without warning, the recorder cannot finish its file. Everything
    up to its last flush, at most a second old, still reads back, and the
    file says it is incomplete."""
    path = tmp_path / "trace.csv.gz"
    with running_tool(path) as proc:
        wait_for_samples(path, 60)
        proc.send_signal(signal.SIGKILL)
        proc.wait()
    meta, samples = recorder.read(path)
    assert not meta["complete"]
    assert len(samples) >= 60 and np.all(samples["free_bytes"] > 0)


def test_labels_from_another_process_survive_a_killed_recorder(tmp_path):
    """The labelling tool, run as a scenario would run it, marks a running
    recorder; killed, the recorder still leaves every label it took."""
    path, fifo = tmp_path / "trace.csv.gz", tmp_path / "labels.fifo"

    def label(event, action):
        subprocess.run([sys.executable, str(LABEL_TOOL), str(fifo), event, action,
                        "--wait", "10"], check=True, timeout=30)

    with running_tool(path, "--labels", str(fifo)) as proc:
        wait_for_samples(path, 10)
        label("start", "scroll")
        time.sleep(0.5)
        label("end", "scroll")
        # The device samples since the labels reach the file at its next flush.
        time.sleep(recorder.FLUSH_SECONDS + 0.5)
        proc.send_signal(signal.SIGKILL)
        proc.wait()
    meta, labels = recorder.read_labels(recorder.labels_path(path))
    assert not meta["complete"]
    assert list(labels["event"]) == ["start", "end"] and set(labels["action"]) == {"scroll"}
    _, samples = recorder.read(path)
    assert 10 <= np.sum(recorder.actions_in_progress(samples["t_mono_ns"], labels) == "scroll")


def test_an_interrupted_recorder_finishes_its_files_and_holds_no_device_memory(tmp_path):
    """SIGINT closes both files as complete. NVML needs no CUDA context, so
    the recorder is not a GPU process at all: it cannot be part of the
    contention it records."""
    path = tmp_path / "trace.csv.gz"
    with running_tool(path) as proc:
        wait_for_samples(path, 60)
        assert proc.pid not in {p.pid for p in nvml.processes()}
        proc.send_signal(signal.SIGINT)
        assert proc.wait(timeout=10) == 0
    meta, samples = recorder.read(path)
    assert meta["complete"] and meta["device"] == nvml.device_name()
    assert int(meta["total_bytes"]) == nvml.memory().total
    assert meta["samples"] == len(samples)
    meta, states, rows = recorder.read_processes(recorder.processes_path(path))
    assert meta["complete"] and meta["samples"] == len(states) >= 5
    assert np.all(states["memory_mhz"] != 0)
    assert proc.pid not in set(rows["pid"])


def test_a_gpu_process_is_recorded_only_while_it_lives(tmp_path):
    """A process that takes a CUDA context mid-recording shows up in the
    samples taken while it holds it, and in none after it exits."""
    path = tmp_path / "trace.csv.gz"
    child = ("import time; from microinfer import _microinfer; "
             "_microinfer.device_memory_info(); print('up', flush=True); time.sleep(1.5)")
    with running_tool(path) as proc:
        wait_for_samples(path, 25)
        gpu = subprocess.Popen([sys.executable, "-c", child], stdout=subprocess.PIPE, text=True)
        try:
            assert gpu.stdout.readline().strip() == "up"
            born = time.monotonic_ns()
            assert gpu.wait(timeout=30) == 0
            died = time.monotonic_ns()
        finally:
            if gpu.poll() is None:
                gpu.kill()
        time.sleep(2.0)  # ten process samples after it exits
        proc.send_signal(signal.SIGINT)
        assert proc.wait(timeout=10) == 0
    _, states, rows = recorder.read_processes(recorder.processes_path(path))
    seen = rows["t_mono_ns"][rows["pid"] == gpu.pid]
    assert len(seen) >= 3 and seen.min() < born + 250e6
    # The driver may drop an exiting process a sample late, but no later.
    assert seen.max() < died + 250e6
    assert np.sum(states["t_mono_ns"] > died + 250e6) >= 5


def test_the_driver_sustains_fifty_and_five_hertz():
    """Two seconds on the real driver: the device stream within 2% of 50 Hz,
    the processes stream missing none of its ten deadlines, and each query
    far cheaper than its period."""
    stats = recorder.record(None, rate_hz=50, duration=2.0)
    procs = stats["processes"]
    assert abs(stats["achieved_hz"] - 50) < 1 and stats["missed"] == 0
    assert procs["samples"] + procs["missed"] == 10 and procs["missed"] == 0
    assert stats["query_us"]["p99"] < 5000 and procs["query_ms"]["p99"] < 50
    print(f"\nNVML memory query: median {stats['query_us']['median']:.0f} us, "
          f"p99 {stats['query_us']['p99']:.0f} us; period p99 {stats['period_ms']['p99']:.2f} ms; "
          f"process query: median {procs['query_ms']['median']:.2f} ms")
