"""The contention recorder: device-wide memory at 50 Hz (#46), and who holds
it at 5 Hz (#47).

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
    path = tmp_path / "trace.csv.gz"
    reader = FakeMemory()
    stats = recorder.record(path, rate_hz=200, duration=0.25, reader=reader,
                            meta={"device": "fake", "total_bytes": reader.total})
    meta, samples = recorder.read(path)
    assert meta["device"] == "fake" and meta["rate_hz"] == "200" and meta["complete"]
    assert len(samples) == stats["samples"] == reader.calls
    assert list(samples.dtype.names) == list(recorder.COLUMNS)
    np.testing.assert_array_equal(samples["free_bytes"],
                                  reader.total - np.arange(1, reader.calls + 1) * 2**20)
    assert np.all(samples["free_bytes"] + samples["used_bytes"] == reader.total)
    assert np.all(np.diff(samples["t_mono_ns"]) > 0)
    assert np.all(samples["query_ns"] >= 0)


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


def test_a_reader_slower_than_the_period_is_counted_as_missed_deadlines():
    def slower_than_period():
        time.sleep(0.025)
        return 1, 1

    stats = recorder.record(None, rate_hz=100, duration=0.5, reader=slower_than_period)
    assert stats["missed"] > 0
    assert stats["achieved_hz"] < 50


def test_every_deadline_is_either_sampled_or_missed_and_none_past_the_end():
    """A read that overruns the end must not count the deadlines after it:
    over 0.1 s at 100 Hz there are ten deadlines, and each is either a sample
    or a miss."""
    def slower_than_period():
        time.sleep(0.025)
        return 1, 1

    rate, duration = 100, 0.1
    stats = recorder.record(None, rate_hz=rate, duration=duration, reader=slower_than_period)
    assert stats["samples"] + stats["missed"] == math.ceil(rate * duration)


def test_metadata_round_trips_whatever_its_keys(tmp_path):
    """A key that begins like the closing line is still a key, and the closing
    line's counts come back too."""
    path = tmp_path / "trace.csv.gz"
    stats = recorder.record(path, rate_hz=100, duration=0.1, reader=FakeMemory(),
                            meta={"completed_by": "someone", "note": "a=b"})
    meta, _ = recorder.read(path)
    assert meta["completed_by"] == "someone" and meta["note"] == "a=b"
    assert meta["complete"] and meta["samples"] == stats["samples"]
    assert meta["missed"] == stats["missed"]


def test_only_a_truncated_recording_reads_as_incomplete(tmp_path):
    """A recording cut short reads back as incomplete; a file that is missing,
    or is not a recording at all, is an error, not an interrupted recorder."""
    with pytest.raises(FileNotFoundError):
        recorder.read(tmp_path / "absent.csv.gz")
    (tmp_path / "plain.txt").write_text("not gzip\n")
    with pytest.raises(OSError):
        recorder.read(tmp_path / "plain.txt")
    with gzip.open(tmp_path / "other.csv.gz", "wt") as f:
        f.write("a,b,c\n1,2,3\n")
    with pytest.raises(ValueError, match="not a contention trace"):
        recorder.read(tmp_path / "other.csv.gz")


def test_a_stop_event_ends_the_recording_cleanly(tmp_path):
    path = tmp_path / "trace.csv.gz"
    stop = threading.Event()
    threading.Timer(0.2, stop.set).start()
    recorder.record(path, rate_hz=100, stop=stop, reader=FakeMemory())
    meta, samples = recorder.read(path)
    assert meta["complete"] and 15 <= len(samples) <= 25


# -- the processes stream -------------------------------------------------------------


def gpu_process(pid, name, used=2**20, kind="graphics"):
    return nvml.GpuProcess(pid=pid, name=name, kind=kind, used_bytes=used)


class FakeProcesses:
    """A process reader whose GPU gains a process at the third sample and
    loses it at the sixth, and whose clocks rise with the sample count."""

    def __init__(self):
        self.calls = 0

    def __call__(self):
        self.calls += 1
        procs = [gpu_process(100, "/usr/bin/Xorg"),
                 gpu_process(200, "a, name with \"commas\"", used=None, kind="compute")]
        if 3 <= self.calls < 6:
            procs.append(gpu_process(300, "game", used=512 * 2**20, kind="compute+graphics"))
        return recorder.ProcessSample(pstate=self.calls % 16,
                                      clocks={"graphics": self.calls, "sm": 2 * self.calls,
                                              "memory": 3 * self.calls},
                                      processes=procs)


def test_the_processes_stream_reads_back_every_sample(tmp_path):
    """Every sample leaves one state row, P-state and clocks, and one row per
    process, with names carrying commas and quotes intact and an unreported
    size as -1."""
    path = tmp_path / "trace.csv.gz"
    procs = FakeProcesses()
    stats = recorder.record(path, rate_hz=200, duration=0.5, reader=FakeMemory(),
                            meta={"device": "fake"}, processes_rate_hz=20,
                            process_reader=procs)
    meta, states, rows = recorder.read_processes(recorder.processes_path(path))
    assert meta["complete"] and meta["device"] == "fake" and meta["rate_hz"] == "20"
    assert len(states) == procs.calls == stats["processes"]["samples"] == meta["samples"]
    assert meta["process_rows"] == len(rows)
    k = np.arange(1, procs.calls + 1)
    np.testing.assert_array_equal(states["pstate"], k % 16)
    np.testing.assert_array_equal(states["graphics_mhz"], k)
    np.testing.assert_array_equal(states["sm_mhz"], 2 * k)
    np.testing.assert_array_equal(states["memory_mhz"], 3 * k)
    named = rows[rows["pid"] == 200]
    assert len(named) == procs.calls
    assert set(named["name"]) == {'a, name with "commas"'}
    assert set(named["kind"]) == {"compute"} and set(named["used_bytes"]) == {-1}
    # Every process row belongs to a sample: its timestamp is a state row's.
    assert set(rows["t_mono_ns"]) <= set(states["t_mono_ns"])


def test_a_process_appears_and_disappears_with_the_samples_that_saw_it(tmp_path):
    path = tmp_path / "trace.csv.gz"
    procs = FakeProcesses()
    recorder.record(path, rate_hz=100, duration=0.5, reader=FakeMemory(),
                    processes_rate_hz=20, process_reader=procs)
    _, states, rows = recorder.read_processes(recorder.processes_path(path))
    game = rows[rows["pid"] == 300]
    np.testing.assert_array_equal(game["t_mono_ns"], states["t_mono_ns"][2:5])
    assert set(game["used_bytes"]) == {512 * 2**20}
    assert set(game["kind"]) == {"compute+graphics"}


def test_the_two_streams_share_one_clock(tmp_path):
    """Both streams stamp their samples with the same monotonic clock, so the
    processes stream's samples fall inside the device stream's span, about
    ten device samples apart at 50 and 5 Hz."""
    path = tmp_path / "trace.csv.gz"
    recorder.record(path, rate_hz=50, duration=1.0, reader=FakeMemory(),
                    processes_rate_hz=5, process_reader=FakeProcesses())
    _, device = recorder.read(path)
    _, states, _ = recorder.read_processes(recorder.processes_path(path))
    assert len(device) == 50 and len(states) == 5
    assert device["t_mono_ns"][0] - 20e6 < states["t_mono_ns"][0]
    assert states["t_mono_ns"][-1] < device["t_mono_ns"][-1]
    between = np.searchsorted(device["t_mono_ns"], states["t_mono_ns"])
    assert np.all(np.abs(np.diff(between) - 10) <= 1)


def test_a_failing_process_reader_fails_the_recording(tmp_path):
    """A processes stream that dies must not leave a device trace that looks
    whole beside a processes trace that stopped: the recording raises."""
    def broken():
        raise RuntimeError("driver gone")

    with pytest.raises(RuntimeError, match="processes stream") as info:
        recorder.record(tmp_path / "trace.csv.gz", rate_hz=100, duration=0.2,
                        reader=FakeMemory(), process_reader=broken)
    assert "driver gone" in str(info.value.__cause__)


def test_without_a_process_reader_an_injected_reader_records_no_processes(tmp_path):
    path = tmp_path / "trace.csv.gz"
    stats = recorder.record(path, rate_hz=100, duration=0.1, reader=FakeMemory())
    assert "processes" not in stats and not recorder.processes_path(path).exists()


def test_the_processes_file_sits_beside_the_device_file():
    assert recorder.processes_path("a/trace.csv.gz") == Path("a/trace.procs.csv.gz")
    assert recorder.processes_path("trace.gz") == Path("trace.procs.gz")


def test_a_processes_trace_is_not_read_as_a_device_trace(tmp_path):
    path = tmp_path / "trace.csv.gz"
    recorder.record(path, rate_hz=100, duration=0.1, reader=FakeMemory(),
                    process_reader=FakeProcesses())
    with pytest.raises(ValueError, match="not a contention trace"):
        recorder.read(recorder.processes_path(path))
    with pytest.raises(ValueError, match="not a contention trace"):
        recorder.read_processes(path)


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


def test_an_interrupted_recorder_finishes_its_file(tmp_path):
    path = tmp_path / "trace.csv.gz"
    with running_tool(path) as proc:
        wait_for_samples(path, 20)
        proc.send_signal(signal.SIGINT)
        assert proc.wait(timeout=10) == 0
    meta, samples = recorder.read(path)
    assert meta["complete"] and meta["device"] == nvml.device_name()
    assert int(meta["total_bytes"]) == nvml.memory().total
    assert meta["samples"] == len(samples)


def test_an_interrupted_recorder_finishes_its_processes_file(tmp_path):
    path = tmp_path / "trace.csv.gz"
    with running_tool(path) as proc:
        wait_for_samples(path, 60)
        proc.send_signal(signal.SIGINT)
        assert proc.wait(timeout=10) == 0
    meta, states, rows = recorder.read_processes(recorder.processes_path(path))
    assert meta["complete"] and meta["device"] == nvml.device_name()
    assert meta["samples"] == len(states) >= 5
    assert np.all((states["pstate"] >= 0) & (states["pstate"] <= 15))
    assert np.all(states["memory_mhz"] > 0)
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
        time.sleep(1.0)
        proc.send_signal(signal.SIGINT)
        assert proc.wait(timeout=10) == 0
    _, states, rows = recorder.read_processes(recorder.processes_path(path))
    seen = rows["t_mono_ns"][rows["pid"] == gpu.pid]
    assert len(seen) >= 3, "the process was not recorded while it held the GPU"
    # The driver may drop an exiting process a sample late.
    assert seen.max() < died + 250e6
    after = states["t_mono_ns"][states["t_mono_ns"] > died + 250e6]
    assert len(after) >= 3 and seen.min() < born + 250e6


def test_the_driver_sustains_five_hertz_of_process_samples():
    stats = recorder.record(None, rate_hz=50, duration=2.0)
    procs = stats["processes"]
    assert procs["samples"] == 10 and procs["missed"] == 0
    assert procs["query_ms"]["p99"] < 50
    print(f"\nprocess query: median {procs['query_ms']['median']:.2f} ms, "
          f"p99 {procs['query_ms']['p99']:.2f} ms")


def test_the_recorder_holds_no_device_memory(tmp_path):
    """NVML needs no CUDA context, so the recorder is not a GPU process at all:
    it cannot be part of the contention it records."""
    path = tmp_path / "trace.csv.gz"
    with running_tool(path) as proc:
        wait_for_samples(path, 20)
        assert proc.pid not in {p.pid for p in nvml.processes()}
        proc.send_signal(signal.SIGINT)
        proc.wait(timeout=10)


def test_the_driver_sustains_fifty_hertz():
    """Two seconds on the real driver: the achieved rate is within 2% of 50 Hz
    and one query costs far less than a period."""
    stats = recorder.record(None, rate_hz=50, duration=2.0)
    assert abs(stats["achieved_hz"] - 50) < 1 and stats["missed"] == 0
    assert stats["query_us"]["p99"] < 5000
    print(f"\nNVML query: median {stats['query_us']['median']:.0f} us, "
          f"p99 {stats['query_us']['p99']:.0f} us; period p99 {stats['period_ms']['p99']:.2f} ms")
