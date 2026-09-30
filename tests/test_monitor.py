"""The VRAM pressure monitor: polling, levels, hysteresis and events (#61).

Headroom is classified as GREEN, YELLOW or RED by thresholds in MiB, and a
new level is reported only once it has held for K polls. The hysteresis is
tested poll by poll on scripts of readings; the thread that polls, with an
injected reader, so that no GPU is needed.
"""

import threading
import time

import pytest

from conftest import each
from microinfer import monitor
from microinfer.footprint import MIB
from microinfer.monitor import GREEN, RED, YELLOW, Thresholds

T = Thresholds(red_below_bytes=512 * MIB, yellow_below_bytes=1024 * MIB, persist_polls=3,
               provisional=False)
K = T.persist_polls
G, Y, R = 2000 * MIB, 800 * MIB, 100 * MIB  # a reading at each level


def test_headroom_is_classified_by_thresholds_in_mib():
    """Below the red threshold RED, below the yellow YELLOW, else GREEN;
    thresholds that do not nest, or K below 1, are refused. The defaults
    say they are provisional."""
    each([(0, RED), (512 * MIB - 1, RED), (512 * MIB, YELLOW), (1024 * MIB - 1, YELLOW),
          (1024 * MIB, GREEN), (6 * 2**30, GREEN)],
         lambda headroom, level: T.classify(headroom) == level or pytest.fail(
             f"{headroom / MIB} MiB is {T.classify(headroom)}, not {level}"))

    def refused(kwargs):
        with pytest.raises(ValueError):
            Thresholds(**{"red_below_bytes": 1, "yellow_below_bytes": 2, "persist_polls": 1,
                          **kwargs})

    each([{"red_below_bytes": 2}, {"red_below_bytes": -1}, {"persist_polls": 0}], refused)
    assert monitor.PROVISIONAL.provisional
    assert monitor.Monitor(reader=lambda: 0).thresholds == monitor.PROVISIONAL


def test_a_level_change_is_reported_on_its_kth_poll_and_a_shorter_drop_is_not():
    """Poll by poll: the first reading's level at once; a clean change on its
    Kth poll, within K + 1 of its first reading; a drop of K - 1 polls never;
    readings that flicker between YELLOW and RED from GREEN as YELLOW, the
    level they kept, and RED once RED alone holds for K."""
    def settles(readings, expected):
        hysteresis = monitor.Hysteresis(T)
        events = [e for i, h in enumerate(readings) if (e := hysteresis.feed(i, h))]
        got = [(e.poll, e.previous, e.level) for e in events]
        assert got == expected, got
        assert all(e.headroom_bytes == readings[e.poll] for e in events)

    each([
        ([G] * 5 + [R] * 5, [(0, None, GREEN), (5 + K - 1, GREEN, RED)]),
        ([R] * 5 + [G] * 5, [(0, None, RED), (5 + K - 1, RED, GREEN)]),
        ([G] * 5 + [R] * (K - 1) + [G] * 5, [(0, None, GREEN)]),
        ([G] * 5 + [Y] * (K - 1) + [G] * 5, [(0, None, GREEN)]),
        ([R] * 5 + [G] * (K - 1) + [R] * 5, [(0, None, RED)]),
        # RED alone from reading 10, the flicker's last.
        ([G] * 3 + [Y, R] * 4 + [R] * K,
         [(0, None, GREEN), (3 + K - 1, GREEN, YELLOW), (10 + K - 1, YELLOW, RED)]),
        ([R] * 3 + [Y, G] * 3, [(0, None, RED), (3 + K - 1, RED, YELLOW)]),
        # A drop that turns back before K polls starts the count again.
        ([G] * 3 + [R, R, G, R, R, R], [(0, None, GREEN), (8, GREEN, RED)]),
    ], settles)


def test_the_thread_reports_a_change_in_time_drains_without_blocking_and_fails_loudly():
    """At 50 ms polls, a clean change is reported within K polls and one of
    when it happened, deadlines missed apart; drain never blocks, and gives
    each event once; a reader that raises stops the polling, and drain
    raises its error."""
    changed_at = time.monotonic_ns() + 200_000_000
    m = monitor.Monitor(reader=lambda: R if time.monotonic_ns() >= changed_at else G,
                        thresholds=T)
    assert m.drain() == []  # not started: nothing
    with m:
        while m.level != RED and time.monotonic_ns() < changed_at + 2e9:
            time.sleep(0.01)
    events = m.drain()
    assert [e.level for e in events] == [GREEN, RED] and m.drain() == [] and not m.running
    late_s = (events[1].t_mono_ns - changed_at) / 1e9
    assert late_s <= (K + 1 + m.missed) * monitor.POLL_S, late_s

    def broken():
        raise OSError("NVML went away")

    failing = monitor.Monitor(reader=broken, thresholds=T, poll_s=0.001)
    with failing:
        threading.Event().wait(0.05)
    with pytest.raises(RuntimeError, match="the pressure monitor stopped") as failed:
        failing.drain()
    assert isinstance(failed.value.__cause__, OSError)
