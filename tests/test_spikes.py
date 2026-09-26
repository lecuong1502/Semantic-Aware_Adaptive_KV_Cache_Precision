"""Spike detection and per-spike measures (#49).

The analysis RQ1's statistics rest on, as the Milestone 1 spec (#45) defines
it: the baseline is the rolling median of free memory over the preceding 5 s,
and a spike is free memory at least 64 MiB below it for at least 100 ms.

It is tested on synthetic traces whose spikes are planted with known shapes:
a trapezoid of deficit that rises linearly over R seconds, holds for H and
falls over F, so that every measure has a value known in advance. Around
them is noise of the size measured on this machine: NVML's free reading on
an idle desktop moves by up to three 2 MiB granules (ADR-0007, note from #10).
"""

import numpy as np
import pytest

from microinfer import spikes

MiB = 2**20
RATE = 50
PERIOD = 1 / RATE
BASE = 3000 * MiB


def quiet(seconds, seed=0):
    """A 50 Hz trace of free memory with this machine's idle noise: every
    sample off the baseline by a whole number of granules, up to three."""
    rng = np.random.default_rng(seed)
    n = int(seconds * RATE)
    t = (np.arange(n) * PERIOD * 1e9).astype(np.int64) + 10**12
    free = BASE + rng.integers(-3, 4, n) * 2 * MiB
    return t, free.astype(np.int64)


def plant(t, free, at, amplitude_mib, rise=0.0, hold=0.0, fall=0.0):
    """Take a trapezoid of memory from `free`, `at` seconds into the trace.
    fall=None holds it to the end of the trace: a lasting step."""
    s = (t - t[0]) / 1e9 - at
    a = amplitude_mib * MiB
    up = np.clip(s / rise, 0, 1) if rise else (s >= 0).astype(float)
    if fall is None:
        down = np.ones_like(s)
    else:
        after = s - rise - hold
        down = 1 - (np.clip(after / fall, 0, 1) if fall else (after >= 0).astype(float))
    free -= (a * np.where(s < 0, 0, np.minimum(up, down))).astype(np.int64)


def expected(amplitude_mib, rise, hold, fall, threshold_mib=64):
    """The measures of a planted trapezoid, noise aside."""
    below = 1 - threshold_mib / amplitude_mib  # the share of a ramp past the threshold
    return {"amplitude": amplitude_mib * MiB, "rise": 0.8 * rise,
            "duration": (rise + fall) * below + hold, "recovery": 0.8 * fall}


def test_every_planted_spike_is_found_with_its_measures_and_nothing_else():
    """Spikes of every shape are found with the measures they were planted
    with. A dip too shallow, and one too brief, are not spikes. A step that
    does not come back is one spike that never recovers, after which the
    lower level is the baseline and a spike on top of it is measured from
    there."""
    t, free = quiet(110)
    planted = {10: (200, 0.0, 0.5, 0.0),  # a square pulse
               20: (500, 0.5, 1.0, 2.0),
               30: (150, 1.0, 0.2, 0.4),
               40: (800, 0.1, 3.0, 0.1),
               90: (250, 0.3, 0.5, 0.5)}  # on the baseline the step left
    for at, shape in planted.items():
        plant(t, free, at, *shape)
    plant(t, free, 55, 50, hold=1.0)  # too shallow
    plant(t, free, 62, 300, hold=0.06)  # too brief
    plant(t, free, 75, 300, rise=0.2, fall=None)  # a lasting step

    found = spikes.find_spikes(t, free)
    starts = [(s.start_ns - t[0]) / 1e9 for s in found]
    assert len(found) == 6, starts
    step = found[4]
    del found[4]

    for spike, (at, shape) in zip(found, planted.items()):
        want = expected(*shape)
        ramp = max(shape[1], shape[3])
        tol = 1.5 * PERIOD + 0.05 * ramp  # a sample either way, and noise on a slope
        assert abs(spike.amplitude_bytes - want["amplitude"]) <= 8 * MiB, at
        assert spike.rise_s == pytest.approx(want["rise"], abs=tol), at
        assert spike.duration_s == pytest.approx(want["duration"], abs=tol), at
        assert spike.recovery_s == pytest.approx(want["recovery"], abs=tol), at
        baseline = BASE - (300 * MiB if at > 75 else 0)  # after the step, its level
        assert spike.recovered and abs(spike.baseline_bytes - baseline) <= 2 * MiB, at
        # A crossing is interpolated, so a sheer edge reads up to a period early.
        assert at - PERIOD <= (spike.start_ns - t[0]) / 1e9 <= at + shape[1]
        assert spike.start_ns <= spike.peak_ns <= spike.end_ns

    assert not step.recovered and step.end_ns is None and step.recovery_s is None
    assert abs(step.amplitude_bytes - 300 * MiB) <= 8 * MiB
    assert step.rise_s == pytest.approx(0.16, abs=1.5 * PERIOD + 0.01)
    assert step.duration_s == pytest.approx(5.0, abs=2 * PERIOD)  # held for the window


def test_the_noise_measured_on_this_machine_makes_no_spikes():
    t, free = quiet(30 * 60, seed=1)
    assert spikes.find_spikes(t, free) == []


def test_the_window_threshold_and_minimum_duration_are_parameters():
    assert spikes.DEFAULT_WINDOW_S == 5.0
    assert spikes.DEFAULT_THRESHOLD_BYTES == 64 * MiB
    assert spikes.DEFAULT_MIN_DURATION_S == 0.1

    t, free = quiet(40)
    plant(t, free, 10, 50, hold=0.5)  # 50 MiB: a spike only at a 32 MiB threshold
    plant(t, free, 20, 200, hold=0.06)  # 60 ms: a spike only at a 50 ms minimum
    plant(t, free, 30, 200, fall=None)  # a step: held for the window, whatever it is
    assert len(spikes.find_spikes(t, free)) == 1
    assert len(spikes.find_spikes(t, free, threshold_bytes=32 * MiB)) == 2
    assert len(spikes.find_spikes(t, free, min_duration_s=0.05)) == 2
    step = spikes.find_spikes(t, free, window_s=2.0)[-1]
    assert not step.recovered and step.duration_s == pytest.approx(2.0, abs=2 * PERIOD)


def test_a_trace_out_of_order_is_refused():
    t, free = quiet(10)
    with pytest.raises(ValueError, match="increasing"):
        spikes.find_spikes(t[::-1], free)
