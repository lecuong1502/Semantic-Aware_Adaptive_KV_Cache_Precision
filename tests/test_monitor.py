"""The VRAM pressure monitor: polling, levels, hysteresis and events (#61).

Headroom is classified as GREEN, YELLOW or RED by thresholds in MiB, and a
new level is reported only once it has held for K polls. The hysteresis is
tested poll by poll on scripts of readings; the thread that polls, with an
injected reader, so that no GPU is needed.
"""

import itertools
import threading
import time

import numpy as np
import pytest

from conftest import each, require_model
from microinfer import Engine, monitor
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


def test_each_transition_says_how_much_of_the_change_was_the_engines_and_the_others():
    """Read only at a transition: the engine's own memory and the other
    processes', and how much each changed since the transition before, with
    the headroom's change; the first transition has nothing to change from."""
    readings = iter([G] * 3 + [R] * 20)
    held = iter([(1000 * MIB, 500 * MIB), (1200 * MIB, 1700 * MIB)])
    with monitor.Monitor(reader=lambda: next(readings, R), attribution=lambda: next(held),
                         thresholds=T, poll_s=0.001) as m:
        while m.level != RED:
            time.sleep(0.005)
    first, red = m.drain()
    assert (first.own_used_bytes, first.others_used_bytes) == (1000 * MIB, 500 * MIB)
    assert first.own_change_bytes is first.others_change_bytes is first.headroom_change_bytes \
        is None
    assert (red.own_change_bytes, red.others_change_bytes) == (200 * MIB, 1200 * MIB)
    assert red.headroom_change_bytes == R - G


def test_the_engine_records_events_between_steps_and_decodes_as_it_would_without():
    """Started by the engine, the monitor's events are drained between steps
    and recorded with the position the cache had reached and when they were
    drained; the tokens are the same as without it, for the engine does not
    react. A monitor that fails is recorded, and generation goes on."""
    engine = Engine(require_model("qwen2.5-0.5b-instruct"))
    engine.load_weights()
    prompt = engine.encode("The pressure monitor watches device memory while the engine")
    steps = 24
    plain = engine.generate(prompt, steps, stop_at_eos=False)

    polls = itertools.count()
    watching = monitor.Monitor(reader=lambda: G if next(polls) < 5 else R,
                               attribution=lambda: (0, 0), thresholds=T, poll_s=0.001)
    assert engine.start_monitor(watching) is watching
    with pytest.raises(RuntimeError, match="already"):
        engine.start_monitor()
    try:
        while watching.level != RED:
            time.sleep(0.005)
        watched = engine.generate(prompt, steps, stop_at_eos=False)
    finally:
        engine.stop_monitor()
    assert np.array_equal(plain, watched)
    records = engine.pressure_events
    assert [r.event.level for r in records] == [GREEN, RED]
    for r in records:
        assert 1 <= r.position <= len(prompt) + steps and r.drained_ns >= r.event.t_mono_ns

    def broken():
        raise OSError("NVML went away")

    engine.start_monitor(monitor.Monitor(reader=broken, thresholds=T, poll_s=0.001))
    try:
        time.sleep(0.02)
        assert np.array_equal(engine.generate(prompt, steps, stop_at_eos=False), plain)
    finally:
        engine.stop_monitor()
    assert isinstance(engine.monitor_error.__cause__, OSError)
