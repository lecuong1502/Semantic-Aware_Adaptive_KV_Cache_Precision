"""RQ1's results from a recording: spikes attributed to processes, statistics
per action or per hour, sensitivity to the threshold, and the log entry
(#50).

Tested on synthetic streams whose spikes and processes are planted, so that
which process caused each spike is known in advance.
"""

import gzip
import hashlib

import numpy as np
import pytest

from microinfer import benchlog, contention, recorder, spikes

MiB = 2**20
RATE = 50
BASE = 3000 * MiB
T0 = 10**12


def device(seconds):
    t = T0 + (np.arange(int(seconds * RATE)) * 1e9 / RATE).astype(np.int64)
    return t, np.full(len(t), BASE, dtype=np.int64)


def take(t, free, at, seconds, mib):
    """A square spike: `mib` taken `at` seconds in, for `seconds`."""
    s = (t - T0) / 1e9
    free[(s >= at) & (s < at + seconds)] -= mib * MiB


def processes(seconds, held):
    """A 5 Hz processes stream. `held(pid, s)` is what process `pid` holds `s`
    seconds in: None where it is not on the GPU, -1 where the driver does not
    say."""
    t = T0 + (np.arange(int(seconds * 5)) * 2e8).astype(np.int64)
    states = np.zeros(len(t), dtype=recorder.STATE_DTYPE)
    states["t_mono_ns"] = t
    rows = []
    for ti in t:
        for pid, name in ((100, "Xorg"), (200, "chrome"), (300, "game")):
            used = held(pid, (ti - T0) / 1e9)
            if used is not None:
                rows.append((ti, 0.0, pid, "graphics", used, name))
    return states, np.array(rows, dtype=recorder.PROCESS_DTYPE)


def test_each_spike_names_the_process_whose_memory_rose_with_it_or_none():
    """Chrome grows into one spike and a game starts into another: each is
    named, with how much it took. A spike no process's rise explains, and one
    too brief for a 5 Hz sample to fall in, name no one, and say why."""
    t, free = device(60)
    take(t, free, 10, 2.0, 300)  # chrome grows by 300 MiB
    take(t, free, 20, 2.0, 500)  # the game starts, holding 500 MiB
    take(t, free, 30, 2.0, 400)  # nothing on the process list moves
    take(t, free, 40.02, 0.12, 400)  # between two 5 Hz samples, at 40.0 and 40.2 s

    def held(pid, s):
        if pid == 100:
            return 90 * MiB
        if pid == 200:
            return (300 if 10 <= s < 12 else 100) * MiB
        return 500 * MiB if 20 <= s < 22 else None

    states, procs = processes(60, held)
    found = spikes.find_spikes(t, free)
    named = [contention.attribute(s, states, procs) for s in found]
    assert [(a.name, a.pid) for a in named[:2]] == [("chrome", 200), ("game", 300)]
    assert named[0].rise_bytes == 200 * MiB and named[0].share == pytest.approx(200 / 300)
    assert named[1].rise_bytes == 500 * MiB
    assert named[2].name is None and "rose" in named[2].reason
    assert named[3].name is None and "sample" in named[3].reason

    # A driver that does not report a process's memory cannot name it.
    states, procs = processes(60, lambda pid, s: -1 if pid == 200 else held(pid, s))
    assert contention.attribute(found[0], states, procs).name is None


def test_a_recording_s_statistics_per_action_and_per_hour_with_sensitivity():
    """Spikes of 50, 100 and 200 MiB, each under its own action: the actions'
    distributions, the counts at 32, 64 and 128 MiB, and, without labels,
    the rate per hour."""
    t, free = device(3600)
    starts = {"scroll": [10, 20, 30], "video": [100, 110], "idle": [200]}
    size = {"scroll": 100, "video": 200, "idle": 50}
    for action, ats in starts.items():
        for at in ats:
            take(t, free, at, 0.5, size[action])
    labels = np.array(
        [(T0 + int(a * 1e9), 0.0, e, n) for n, a, b in
         (("scroll", 5, 40), ("video", 95, 120), ("idle", 195, 210))
         for e, a in ((recorder.START, a), (recorder.END, b))],
        dtype=recorder.LABEL_DTYPE)
    samples = np.zeros(len(t), dtype=recorder.DTYPE)
    samples["t_mono_ns"], samples["free_bytes"] = t, free
    empty = np.zeros(0, dtype=recorder.PROCESS_DTYPE)
    states = np.zeros(0, dtype=recorder.STATE_DTYPE)

    results = contention.analyse(samples, states, empty, labels)
    assert results["sensitivity_mib"] == {"32": 6, "64": 5, "128": 2}
    assert results["spikes"] == 5 and results["attributed"] == {"(none)": 5}
    scroll = results["by_action"]["scroll"]
    assert scroll["count"] == 3
    assert scroll["amplitude_mib"] == {"median": 100.0, "p90": 100.0, "max": 100.0}
    assert scroll["duration_ms"]["median"] == pytest.approx(500, abs=25)
    assert results["by_action"]["video"]["count"] == 2 and "idle" not in results["by_action"]
    assert "per_hour" not in results

    unlabelled = contention.analyse(samples, states, empty, None)
    assert "by_action" not in unlabelled
    assert unlabelled["per_hour"]["spikes"] == pytest.approx(5, rel=0.01)


def test_a_recording_is_logged_with_the_sha256_of_its_files(tmp_path):
    path = tmp_path / "trace.csv.gz"
    reader_calls = iter(range(10**6))

    def reader():  # a 200 MiB spike from the seventh second, for half a second
        s = next(reader_calls) / 100
        free = BASE - (200 * MiB if 6.0 <= s < 6.5 else 0)
        return free, 6 * 2**30 - free

    recorder.record(path, rate_hz=100, duration=7.5, reader=reader,
                    meta={"device": "fake", "model": "Qwen2.5-1.5B", "context_length": 32768})
    log = tmp_path / "log.jsonl"
    entry = contention.log_recording(path, issue=50, log=log)

    assert benchlog.read(log) == [entry]
    assert entry["kind"] == "contention-trace" and entry["model"] == "Qwen2.5-1.5B"
    assert entry["context_length"] == 32768
    assert entry["config"]["files"] == {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()}
    assert entry["results"]["spikes"] == 1 and entry["results"]["complete"]
    assert "per_hour" in entry["results"]

    with gzip.open(tmp_path / "other.csv.gz", "wt") as f:
        f.write("not a trace\n")
    with pytest.raises(ValueError, match="not a contention trace"):
        contention.log_recording(tmp_path / "other.csv.gz", issue=50, log=log)
    assert len(benchlog.read(log)) == 1
