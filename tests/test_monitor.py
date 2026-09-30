"""The VRAM pressure monitor: polling, levels, hysteresis and events (#61).

A background thread reads device headroom on a fixed schedule, classifies it
as GREEN, YELLOW or RED by thresholds in MiB, and reports a new level only
once it has held for K polls. Tested with an injected reader, a script of
headroom readings, so that no GPU is needed and every count is in polls.
"""

import threading
import time

import pytest

from conftest import each
from microinfer import monitor
from microinfer.monitor import GREEN, RED, YELLOW, Thresholds

MiB = 2**20
T = Thresholds(red_below_bytes=512 * MiB, yellow_below_bytes=1024 * MiB, persist_polls=3,
               provisional=False)
K = T.persist_polls


class Script:
    """A reader that returns `readings` in turn, one per poll, and then
    keeps returning the last and says it is done."""

    def __init__(self, readings):
        self.readings, self.polls, self.done = list(readings), 0, threading.Event()

    def __call__(self):
        k = min(self.polls, len(self.readings) - 1)
        self.polls += 1
        if self.polls >= len(self.readings):
            self.done.set()
        return self.readings[k]


def run(readings):
    """Every event the monitor emits over `readings`."""
    script = Script(readings)
    with monitor.Monitor(reader=script, thresholds=T, poll_s=0.001) as m:
        assert script.done.wait(10)
    return m.drain()


def test_headroom_is_classified_by_thresholds_in_mib():
    """Below the red threshold RED, below the yellow YELLOW, else GREEN;
    thresholds that do not nest, or K below 1, are refused."""
    each([(0, RED), (512 * MiB - 1, RED), (512 * MiB, YELLOW), (1024 * MiB - 1, YELLOW),
          (1024 * MiB, GREEN), (6 * 2**30, GREEN)],
         lambda headroom, level: T.classify(headroom) == level or pytest.fail(
             f"{headroom / MiB} MiB is {T.classify(headroom)}, not {level}"))

    def refused(**kwargs):
        with pytest.raises(ValueError):
            Thresholds(**{"red_below_bytes": 1, "yellow_below_bytes": 2, "persist_polls": 1,
                          **kwargs})

    each([{"red_below_bytes": 2}, {"red_below_bytes": -1}, {"persist_polls": 0}],
         lambda kwargs: refused(**kwargs))
    assert monitor.PROVISIONAL.provisional and monitor.Monitor(reader=lambda: 0).thresholds \
        == monitor.PROVISIONAL


def test_a_clean_step_is_reported_within_k_polls_plus_one_and_a_shorter_dip_is_not():
    """The first poll's level is reported at once. A dip of K - 1 polls into
    RED, and one into YELLOW, go unreported; a step that stays is reported
    within K + 1 polls of its first reading, with the headroom that set it.
    Events are drained without blocking, once each."""
    green, yellow, red = 2000 * MiB, 800 * MiB, 100 * MiB
    readings = ([green] * 10 + [red] * (K - 1) + [green] * 10 + [yellow] * (K - 1)
                + [green] * 10 + [red] * 10 + [yellow] * 10 + [green] * 10)
    events = run(readings)
    assert [(e.previous, e.level) for e in events] == [
        (None, GREEN), (GREEN, RED), (RED, YELLOW), (YELLOW, GREEN)]
    assert events[0].poll == 0
    steps = [i for i in range(1, len(readings)) if readings[i] != readings[i - 1]]
    for event, step in zip(events[1:], steps[4:]):
        assert step <= event.poll <= step + K, (event, step)
        assert event.headroom_bytes == readings[event.poll]
    assert all(a.t_mono_ns < b.t_mono_ns for a, b in zip(events, events[1:]))


def test_the_thread_starts_stops_and_reports_a_failed_reader():
    """drain never blocks, even with nothing to drain; stop ends the thread;
    a reader that raises stops the polling, and drain raises its error."""
    m = monitor.Monitor(reader=lambda: 2000 * MiB, thresholds=T, poll_s=0.001)
    assert m.drain() == []  # not started: nothing, at once
    with m:
        time.sleep(0.05)
        started = time.monotonic()
        assert [e.level for e in m.drain()] == [GREEN] and m.drain() == []
        assert time.monotonic() - started < 0.01
        assert m.level == GREEN
    assert not m.running

    def broken():
        raise OSError("NVML went away")

    with pytest.raises(RuntimeError, match="the pressure monitor stopped") as failed:
        with monitor.Monitor(reader=broken, thresholds=T, poll_s=0.001) as m:
            time.sleep(0.05)
            m.drain()
    assert isinstance(failed.value.__cause__, OSError)
