"""Synthetic contention: schedules for the simulator from patterns (#59).

A step, a sawtooth, and Poisson spikes whose amplitudes, rise times and
durations are drawn from given distributions: fixed, uniform, or RQ1's own
spikes, read back from the benchmark log. Tested for their shapes, their
determinism given a seed, and by running one through the simulator.
"""

import threading

import numpy as np
import pytest

from microinfer import benchlog, patterns, simulator

MiB = 2**20
GRANULE = simulator.GRANULE


def levels(schedule, times):
    return [schedule.at(t) for t in times]


def test_a_step_and_a_sawtooth_have_the_shapes_asked_for():
    """A step takes X for T seconds, after a lead, and gives it back. A
    sawtooth climbs from 0 to its peak over each period and drops at once,
    for as many cycles as asked, and ends at 0. Every level is whole
    granules."""
    step = patterns.step(300 * MiB, hold_s=2.0, lead_s=0.5)
    assert step.points == [(0.0, 0), (0.5, 300 * MiB), (2.5, 0)]

    saw = patterns.sawtooth(512 * MiB, period_s=1.0, cycles=3, resolution_s=0.1)
    for cycle in range(3):
        climb = levels(saw, [cycle + 0.05 + 0.1 * k for k in range(10)])
        assert climb == sorted(climb) and climb[0] < 64 * MiB, cycle
        assert abs(climb[-1] - 512 * MiB) <= 512 * MiB * 0.1 + GRANULE
    assert saw.at(3.0) == 0 and saw.duration_s == pytest.approx(3.0)
    assert all(b % GRANULE == 0 for _, b in step.points + saw.points)
    for bad in (lambda: patterns.step(-1, 1.0), lambda: patterns.sawtooth(1, 0, 1)):
        with pytest.raises(ValueError):
            bad()


def test_poisson_spikes_are_drawn_from_their_distributions_and_the_same_given_a_seed():
    """Arrivals at the rate asked; each spike a trapezoid whose amplitude,
    rise, plateau and fall come from their distributions; overlapping spikes
    add. The same seed gives the same schedule, another seed another."""
    params = patterns.SpikeParameters(
        rate_per_s=0.5, amplitude_bytes=patterns.Uniform(100 * MiB, 300 * MiB),
        rise_s=patterns.Fixed(0.1), duration_s=patterns.Uniform(0.2, 1.0),
        fall_s=patterns.Fixed(0.2))
    a = patterns.poisson_spikes(params, duration_s=120.0, seed=7)
    assert a.duration_s >= 120.0  # as long as asked, however early the last spike
    assert a == patterns.poisson_spikes(params, duration_s=120.0, seed=7)
    assert a != patterns.poisson_spikes(params, duration_s=120.0, seed=8)

    # Recover the spikes from the schedule: each rise from 0.
    rises = [t for (t0, b0), (t, b) in zip(a.points, a.points[1:]) if b0 == 0 and b > 0]
    assert 40 <= len(rises) <= 80  # about 60 at 0.5 per second over 120 s, overlaps merged
    peak = max(b for _, b in a.points)
    assert 300 * MiB < peak <= 4 * 300 * MiB + GRANULE  # some overlap, and add
    assert a.at(a.duration_s) == 0 and all(b % GRANULE == 0 for _, b in a.points)

    alone = patterns.poisson_spikes(
        patterns.SpikeParameters(rate_per_s=0.05, amplitude_bytes=patterns.Fixed(200 * MiB),
                                 rise_s=patterns.Fixed(0.0), duration_s=patterns.Fixed(0.5),
                                 fall_s=patterns.Fixed(0.0)), duration_s=600.0, seed=3)
    assert {b for _, b in alone.points} <= {0, 200 * MiB, 400 * MiB}


def test_a_generated_schedule_runs_on_the_simulator():
    schedule = patterns.step(64 * MiB, hold_s=0.3)
    stop = threading.Event()
    reservation = simulator.Reservation()
    threading.Timer(0.6, stop.set).start()
    try:
        events = simulator.run(schedule, reservation, stop)
    finally:
        reservation.release()
    assert [e["taken_bytes"] for e in events] == [64 * MiB, 0]


def test_parameters_come_from_rq1s_spikes_in_the_benchmark_log(tmp_path):
    """The spikes each contention-trace entry lists, of the issues and the
    action asked for, become empirical distributions. The rate is the spikes
    per second of recorded time, or, for one action, of the time the
    recordings spent on it: a minute each, as the protocol holds a span."""
    log = tmp_path / "log.jsonl"

    def recording(issue, spikes, hours):
        benchlog.append("contention-trace", model=None, context_length=None,
                        precision_tiers=None, config={"issue": issue, "trace": "t.csv.gz"},
                        results={"recorded_hours": hours, "spikes": spikes}, log=log)

    def spike(action, amplitude, rise, duration, recovery):
        return {"action": action, "amplitude_mib": amplitude, "rise_ms": rise,
                "duration_ms": duration, "recovery_ms": recovery, "ending": "recovered"}

    recording(55, [spike("vlc-2160p", 500, 100, 2000, 300),
                   spike("chrome-tabs-1", 150, 20, 400, None)], 0.25)
    recording(56, [spike("vlc-2160p", 540, 120, 1800, 400)], 0.25)
    recording(99, [spike("vlc-2160p", 9999, 1, 1, 1)], 1.0)  # another issue's

    vlc = patterns.from_rq1(log, issues=(55, 56), action="vlc-2160p")
    assert sorted(vlc.amplitude_bytes.values) == [500 * MiB, 540 * MiB]
    assert sorted(vlc.rise_s.values) == pytest.approx([0.125, 0.15])  # 10% to 90%: 0.8 of a rise
    assert sorted(vlc.duration_s.values) == [1.8, 2.0]
    assert sorted(vlc.fall_s.values) == pytest.approx([0.375, 0.5])  # 90% to 10%: 0.8 of a fall
    assert vlc.rate_per_s == pytest.approx(2 / (2 * 60))

    every = patterns.from_rq1(log, issues=(55, 56))
    assert len(every.amplitude_bytes.values) == 3
    assert every.rate_per_s == pytest.approx(3 / 1800)
    assert len(every.fall_s.values) == 2  # a spike with no recovery gives no fall

    # No spike of this action recovered: its fall is drawn as instant.
    recording(56, [spike("chrome-webgl", 100, 50, 900, None)], 0.25)
    assert patterns.from_rq1(log, issues=(56,), action="chrome-webgl").fall_s == patterns.Fixed(0.0)
    with pytest.raises(ValueError):
        patterns.Empirical(())

    rng = np.random.default_rng(0)
    assert all(vlc.amplitude_bytes.draw(rng) in (500 * MiB, 540 * MiB) for _ in range(20))
    with pytest.raises(ValueError, match="no spikes"):
        patterns.from_rq1(log, issues=(55,), action="nothing")
