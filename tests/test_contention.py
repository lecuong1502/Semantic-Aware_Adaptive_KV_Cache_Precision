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
from microinfer.contention import Unnamed

MiB = 2**20
RATE = 50
BASE = 3000 * MiB
T0 = 10**12
NAMES = {100: "Xorg", 200: "chrome", 300: "game", 400: "engine"}


def device(seconds):
    t = T0 + (np.arange(int(seconds * RATE)) * 1e9 / RATE).astype(np.int64)
    return t, np.full(len(t), BASE, dtype=np.int64)


def take(t, free, at, seconds, mib):
    """A square spike: `mib` taken `at` seconds in, for `seconds`."""
    s = np.round((t - T0) / 1e9, 9)
    free[(s >= at) & (s < at + seconds)] -= mib * MiB


def processes(seconds, held, names=NAMES):
    """A 5 Hz processes stream. `held(pid, s)` is what process `pid` holds `s`
    seconds in: None where it is not on the GPU, -1 where the driver does not
    say. `names(pid, s)` may rename a pid, as a reused one would be."""
    t = T0 + (np.arange(int(seconds * 5)) * 2e8).astype(np.int64)
    states = np.zeros(len(t), dtype=recorder.STATE_DTYPE)
    states["t_mono_ns"] = t
    rows = []
    for ti in t:
        s = (ti - T0) / 1e9
        for pid in NAMES:
            used = held(pid, s)
            if used is not None:
                name = names(pid, s) if callable(names) else names[pid]
                rows.append((ti, 0.0, pid, "graphics", used, name))
    return contention.ProcessStream(states, np.array(rows, dtype=recorder.PROCESS_DTYPE))


def test_each_spike_names_the_process_most_responsible_or_says_why_not():
    """Chrome grows into one spike and a game starts into another; the engine
    and chrome share a third, and the larger is named with its share. A pid
    that another program took over is credited with all it holds. A spike no
    process's gain explains, one too brief for a 5 Hz sample to fall in its
    peak, and one whose cause the driver does not report, name no one, and
    say why."""
    t, free = device(80)
    take(t, free, 10, 2.0, 300)  # chrome grows by 300 MiB
    take(t, free, 20, 2.0, 500)  # the game starts, holding 500 MiB
    take(t, free, 30, 2.0, 300)  # the engine takes 180 MiB, chrome 120
    take(t, free, 40, 2.0, 400)  # nothing on the process list moves
    take(t, free, 50.02, 0.12, 400)  # between two 5 Hz samples, at 50.0 and 50.2 s
    take(t, free, 60, 2.0, 400)  # Xorg's pid, reused by a program holding 400 MiB

    def held(pid, s):
        if pid == 100:
            return (400 if 60 <= s < 62 else 90) * MiB
        if pid == 200:
            return (400 if 10 <= s < 12 else 220 if 30 <= s < 32 else 100) * MiB
        if pid == 300:
            return 500 * MiB if 20 <= s < 22 else None
        return (1180 if 30 <= s < 32 else 1000) * MiB

    def names(pid, s):
        return "newcomer" if pid == 100 and 60 <= s < 62 else NAMES[pid]

    stream = processes(80, held, names)
    found = spikes.find_spikes(t, free)
    named = [contention.attribute(s, stream) for s in found]
    assert [(a.name, a.pid) for a in named] == [
        ("chrome", 200), ("game", 300), ("engine", 400), (None, None), (None, None),
        ("newcomer", 100)]
    assert named[0].gain_bytes == 300 * MiB and named[0].share == pytest.approx(1.0)
    assert named[2].share == pytest.approx(180 / 300)
    assert named[3].unnamed is Unnamed.TOO_SMALL
    assert named[4].unnamed is Unnamed.NO_SAMPLE_DURING
    assert named[5].gain_bytes == 400 * MiB

    unreported = processes(80, lambda pid, s: -1 if pid == 200 else held(pid, s), names)
    assert contention.attribute(found[0], unreported).unnamed is Unnamed.UNREPORTED
    assert contention.attribute(found[0], contention.ProcessStream.absent()).unnamed \
        is Unnamed.NO_STREAM


def test_a_recording_s_statistics_per_action_and_per_hour_with_sensitivity():
    """Spikes of 50, 100 and 200 MiB under their actions, one between them
    and one cut off by the end of the trace: every action is reported, the
    quiet one with a count of 0; a censored spike counts, but its duration,
    a lower bound, joins no distribution. Counts at 32, 64 and 128 MiB, in
    total and per action; without labels, rates per hour of recorded time."""
    t, free = device(3600)
    for at in (10, 20, 30):
        take(t, free, at, 0.5, 100)  # scroll
    for at in (100, 110):
        take(t, free, at, 0.5, 200)  # video
    take(t, free, 200, 0.5, 50)  # idle: too shallow at 64 MiB
    take(t, free, 300, 0.5, 100)  # between actions
    take(t, free, 3599.5, 1.0, 100)  # cut off by the end
    labels = np.array(
        [(T0 + int(at * 1e9), 0.0, event, action) for action, a, b in
         (("scroll", 5, 40), ("video", 95, 120), ("idle", 195, 210), ("end", 3590, 3600))
         for event, at in ((recorder.START, a), (recorder.END, b))],
        dtype=recorder.LABEL_DTYPE)
    samples = np.zeros(len(t), dtype=recorder.DEVICE_DTYPE)
    samples["t_mono_ns"], samples["free_bytes"] = t, free
    stream = contention.ProcessStream.absent()

    results = contention.analyse(samples, stream, labels)
    assert results["sensitivity_mib"]["32"] == {
        "count": 8, "by_action": {"(between actions)": 1, "end": 1, "idle": 1, "scroll": 3,
                                  "video": 2}}
    assert results["sensitivity_mib"]["64"]["count"] == 7
    assert results["sensitivity_mib"]["128"]["count"] == 2
    by_action = results["by_action"]
    assert by_action["idle"]["count"] == 0 and by_action["idle"]["amplitude_mib"] is None
    assert by_action["scroll"]["amplitude_mib"] == {"median": 100.0, "p90": 100.0, "max": 100.0}
    assert by_action["scroll"]["duration_ms"]["median"] == pytest.approx(500, abs=25)
    assert by_action["(between actions)"]["count"] == 1
    assert by_action["end"]["endings"]["censored"] == 1 and by_action["end"]["duration_ms"] is None
    assert results["all"]["attributed"] == {"(none)": 7}
    assert len(results["spikes"]) == 7 and results["spikes"][0]["action"] == "scroll"
    assert results["spikes"][0]["unnamed"] == Unnamed.NO_STREAM.value
    assert "per_hour" not in results

    unlabelled = contention.analyse(samples, stream, None)
    assert "by_action" not in unlabelled
    assert unlabelled["per_hour"]["spikes"] == pytest.approx(7, rel=0.01)
    gapped = samples[(t - T0) % int(600e9) >= int(300e9)]  # half the hour missing
    assert contention.analyse(gapped, stream, None)["recorded_hours"] == \
        pytest.approx(0.5, rel=0.01)
    short = samples[: 60 * RATE]
    assert contention.analyse(short, stream, None)["per_hour"] is None  # too short to rate


def test_a_recording_is_logged_with_every_spike_and_the_sha256_of_its_files(tmp_path):
    path = tmp_path / "trace.csv.gz"
    calls = iter(range(10**6))

    def reader():  # a 200 MiB spike from the second second, for half a second
        s = next(calls) / 100
        free = BASE - (200 * MiB if 1.0 <= s < 1.5 else 0)
        return free, 6 * 2**30 - free

    recorder.record(path, rate_hz=100, duration=2.0, reader=reader,
                    meta={"device": "fake", "model": "Qwen2.5-1.5B", "context_length": 32768,
                          "precision": "FP16"})
    log = tmp_path / "log.jsonl"
    entry = contention.log_recording(path, issue=50, log=log, window_s=0.5)

    assert benchlog.read(log) == [entry]
    assert entry["kind"] == "contention-trace" and entry["model"] == "Qwen2.5-1.5B"
    assert entry["context_length"] == 32768 and entry["precision_tiers"] is None
    assert entry["config"]["recording"]["precision"] == "FP16"
    assert entry["config"]["files"] == {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()}
    results = entry["results"]
    assert results["complete"] and not results["process_stream"]
    assert [s["amplitude_mib"] for s in results["spikes"]] == [200.0]

    with gzip.open(tmp_path / "other.csv.gz", "wt") as f:
        f.write("not a trace\n")
    with pytest.raises(ValueError, match="not a contention trace"):
        contention.log_recording(tmp_path / "other.csv.gz", issue=50, log=log)
    assert len(benchlog.read(log)) == 1
