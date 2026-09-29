"""Synthetic contention: schedules for the simulator from patterns (#59).

A step, a sawtooth, trapezoids of a known shape, and Poisson arrivals of
trapezoids drawn from distributions: fixed, uniform, or RQ1's own, read back
from the benchmark log and the labels beside the traces. Tested for their
shapes, their determinism given a seed, what RQ1's measures become, and by
running one through the simulator.
"""

import threading

import numpy as np
import pytest

from conftest import each
from microinfer import benchlog, patterns, recorder, simulator
from microinfer.patterns import Fixed, Trapezoid, Uniform

MiB = 2**20
GRANULE = simulator.GRANULE


def levels(schedule, times):
    return [schedule.at(t) for t in times]


def test_a_step_a_sawtooth_and_trapezoids_have_the_shapes_asked_for():
    """A step takes X for T seconds after a lead. A sawtooth climbs from 0 to
    its peak over each period and drops. A trapezoid rises, keeps its
    plateau and falls, and a train repeats it after a gap. Every level is
    whole granules, halves rounded up, and every schedule ends at 0."""
    assert patterns.step(300 * MiB, for_s=2.0, lead_s=0.5).points == [
        (0.0, 0), (0.5, 300 * MiB), (2.5, 0)]
    assert patterns.step(GRANULE // 2, for_s=1.0).points[0] == (0.0, GRANULE)

    saw = patterns.sawtooth(512 * MiB, period_s=1.0, cycles=3, resolution_s=0.1)
    for cycle in range(3):
        climb = levels(saw, [cycle + 0.05 + 0.1 * k for k in range(10)])
        assert climb[0] == 0 and climb == sorted(climb) and climb[-1] == 512 * MiB, cycle
    assert saw.at(3.0) == 0 and saw.duration_s == pytest.approx(3.0)

    shape = Trapezoid(amplitude_bytes=400 * MiB, rise_s=0.4, plateau_s=1.0, fall_s=0.2)
    one = patterns.trapezoids(shape, lead_s=1.0, resolution_s=0.01)
    assert levels(one, [0.9, 1.2, 1.41, 2.3, 2.61]) == [0, 200 * MiB, 400 * MiB, 400 * MiB, 0]
    train = patterns.trapezoids(shape, count=3, gap_s=0.5, resolution_s=0.01)
    starts = [t for (t0, b0), (t, b) in zip(train.points, train.points[1:]) if b0 == 0 and b]
    assert starts == pytest.approx([0.01, 2.11, 4.21], abs=0.011)
    assert all(b % GRANULE == 0 for s in (saw, one, train) for _, b in s.points)
    assert all(s.points[-1][1] == 0 for s in (saw, one, train))

    def refused(make):
        with pytest.raises(ValueError):
            make()

    each([(lambda: patterns.step(-1, 1.0),), (lambda: patterns.step(1, 0),),
          (lambda: patterns.sawtooth(1, 0.1, 1, 0.1),), (lambda: patterns.trapezoids(shape, 0),)],
         refused)


def test_poisson_arrivals_are_drawn_from_their_distributions_and_the_same_given_a_seed():
    """Arrivals at the rate asked, over the length asked; overlapping shapes
    add. The same seed gives the same schedule, another seed another."""
    params = patterns.ShapeParameters(
        rate_per_s=0.5, amplitude_bytes=Uniform(100 * MiB, 300 * MiB), rise_s=Fixed(0.1),
        plateau_s=Uniform(0.2, 1.0), fall_s=Fixed(0.2))
    a = patterns.poisson(params, length_s=120.0, seed=7)
    assert a == patterns.poisson(params, length_s=120.0, seed=7)
    assert a != patterns.poisson(params, length_s=120.0, seed=8)
    assert a.duration_s >= 120.0 and a.points[-1][1] == 0
    rises = [t for (t0, b0), (t, b) in zip(a.points, a.points[1:]) if b0 == 0 and b > 0]
    assert 40 <= len(rises) <= 80  # about 60 at 0.5 per second over 120 s, overlaps merged
    assert 300 * MiB < max(b for _, b in a.points) <= 4 * 300 * MiB  # some overlap, and add


def test_a_generated_schedule_runs_on_the_simulator():
    stop = threading.Event()
    reservation = simulator.Reservation()
    threading.Timer(0.6, stop.set).start()
    try:
        events = simulator.run(patterns.step(64 * MiB, for_s=0.3), reservation, stop)
    finally:
        reservation.release()
    assert [e["taken_bytes"] for e in events] == [64 * MiB, 0]


def test_rq1s_spikes_and_lasting_drops_become_their_own_parameters(tmp_path):
    """Spikes that recovered give amplitude, rise, plateau and fall; lasting
    drops give amplitude and rise, and are kept as long as the caller says.
    A measured rise or recovery is 0.8 of its ramp; a plateau is the time at
    or above the threshold less the ramps' share above it. Censored spikes
    count for neither. The rate for an action is per second of its spans,
    read from the labels beside the traces, in the recordings that have it."""
    log, traces = tmp_path / "log.jsonl", tmp_path / "traces"
    traces.mkdir()

    def recording(issue, name, spikes, spans, hours):
        benchlog.append("contention-trace", model=None, context_length=None,
                        precision_tiers=None,
                        config={"issue": issue, "trace": name, "threshold_mib": 64},
                        results={"recorded_hours": hours, "spikes": spikes}, log=log)
        writer = recorder._TraceWriter(recorder.labels_path(traces / name), recorder.LABEL_FORMAT,
                                       {}, recorder.LABEL_COLUMNS)
        for action, start_s, end_s in spans:
            writer.row(int(start_s * 1e9), 0.0, recorder.START, action)
            writer.row(int(end_s * 1e9), 0.0, recorder.END, action)
        writer.close({"labels": 2 * len(spans), "rejected": 0})

    def spike(action, ending, amplitude=128, rise=80, duration=1000, recovery=160):
        return {"action": action, "ending": ending, "amplitude_mib": amplitude,
                "rise_ms": rise, "duration_ms": duration,
                "recovery_ms": recovery if ending == "recovered" else None}

    recording(55, "a.csv.gz", [spike("vlc", "lasting", 500), spike("tabs", "recovered"),
                               spike("tabs", "censored")],
              [("vlc", 100, 160), ("tabs", 10, 70)], hours=0.25)
    recording(56, "b.csv.gz", [spike("vlc", "lasting", 540), spike("engine-prefill", "lasting")],
              [("engine-prefill", 0, 600), ("vlc", 700, 760)], hours=0.25)

    vlc = patterns.from_rq1("drops", "vlc", log, traces=traces, kept_s=Fixed(60.0))
    assert sorted(vlc.amplitude_bytes.values) == [500 * MiB, 540 * MiB]
    assert vlc.rate_per_s == pytest.approx(2 / 120)  # two in two minutes of vlc
    assert vlc.plateau_s == Fixed(60.0) and vlc.fall_s == Fixed(0.0)
    engine = patterns.from_rq1("drops", "engine-prefill", log, traces=traces, kept_s=Fixed(1))
    assert engine.rate_per_s == pytest.approx(1 / 600)  # a.csv.gz had no engine

    tabs = patterns.from_rq1("spikes", "tabs", log, traces=traces)
    assert tabs.rate_per_s == pytest.approx(1 / 60)  # the censored one is left out
    assert tabs.rise_s.values == pytest.approx((0.1,)) and tabs.fall_s.values == pytest.approx((0.2,))
    # 1.0 s at or above 64 MiB of a 128 MiB spike: half of each ramp is above.
    assert tabs.plateau_s.values == pytest.approx((1.0 - (0.1 + 0.2) * 0.5,))

    every = patterns.from_rq1("drops", log=log, traces=traces, kept_s=Fixed(1))
    assert every.rate_per_s == pytest.approx(3 / 1800)

    def refused(call, error):
        with pytest.raises(error):
            call()

    (traces / "b.labels.csv.gz").unlink()
    each([(lambda: patterns.from_rq1("drops", "vlc", log, traces=traces), ValueError),
          (lambda: patterns.from_rq1("spikes", "nothing", log, issues=(55,), traces=traces),
           ValueError),
          (lambda: patterns.from_rq1("drops", "vlc", log, traces=traces, kept_s=Fixed(1)),
           FileNotFoundError),
          (lambda: patterns.Empirical(()), ValueError)], refused)
    rng = np.random.default_rng(0)
    assert all(vlc.amplitude_bytes.draw(rng) in (500 * MiB, 540 * MiB) for _ in range(20))
