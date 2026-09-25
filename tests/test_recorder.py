"""The contention recorder: device-wide memory at 50 Hz (#46).

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
