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

from typing import NamedTuple

import numpy as np
import pytest

from microinfer import spikes

MiB = 2**20
RATE = 50
PERIOD = 1 / RATE
BASE = 3000 * MiB


def quiet(seconds, seed=0, noise=True):
    """A 50 Hz trace of free memory with this machine's idle noise: every
    sample off the baseline by a whole number of granules, up to three."""
    rng = np.random.default_rng(seed)
    n = int(seconds * RATE)
    t = (np.arange(n) * PERIOD * 1e9).astype(np.int64) + 10**12
    free = np.full(n, BASE, dtype=np.int64)
    if noise:
        free += rng.integers(-3, 4, n) * 2 * MiB
    return t, free


class Shape(NamedTuple):
    """A trapezoid of deficit: up linearly over `rise` seconds, level for
    `hold`, down over `fall`. fall=None holds it to the end: a lasting drop."""

    amplitude_mib: float
    rise: float = 0.0
    hold: float = 0.0
    fall: float | None = 0.0


def plant(t, free, at, shape):
    """Take `shape` from `free`, `at` seconds into the trace."""
    s = np.round((t - t[0]) / 1e9 - at, 9)
    up = np.clip(s / shape.rise, 0, 1) if shape.rise else (s >= 0).astype(float)
    if shape.fall is None:
        down = np.ones_like(s)
    else:
        after = s - shape.rise - shape.hold
        down = 1 - (np.clip(after / shape.fall, 0, 1) if shape.fall
                    else (after >= 0).astype(float))
    level = np.where(s < 0, 0, np.minimum(up, down))
    free -= (shape.amplitude_mib * MiB * level).astype(np.int64)


def expected(shape, threshold_mib=64):
    """The measures of a planted trapezoid, noise aside, and how far noise
    of 8 MiB can move a crossing on its slowest slope, plus a sample."""
    below = 1 - threshold_mib / shape.amplitude_mib  # the share of a ramp past the threshold
    ramp = max(shape.rise, shape.fall)
    return {"amplitude": shape.amplitude_mib * MiB, "rise": 0.8 * shape.rise,
            "duration": (shape.rise + shape.fall) * below + shape.hold,
            "recovery": 0.8 * shape.fall,
            "tol": 1.5 * PERIOD + 8 / shape.amplitude_mib * ramp}


def test_every_planted_spike_is_found_with_its_measures_and_nothing_else():
    """Spikes of every shape are found with the measures they were planted
    with; one whose deficit sits in the noise just above the threshold is
    found once, not in pieces. A dip too shallow, and one too brief, are not
    spikes. A drop that does not come back is one spike, a lasting drop,
    after which its level is the baseline and a spike on it is measured from
    there."""
    t, free = quiet(125)
    planted = {10: Shape(200, hold=0.5),  # a square pulse
               20: Shape(500, 0.5, 1.0, 2.0),
               30: Shape(150, 1.0, 0.2, 0.4),
               40: Shape(800, 0.1, 3.0, 0.1),
               105: Shape(250, 0.3, 0.5, 0.5)}  # on the level the drop left
    near_threshold = Shape(70, 1.0, 0.5, 1.0)
    for at, shape in planted.items():
        plant(t, free, at, shape)
    plant(t, free, 50, near_threshold)
    plant(t, free, 70, Shape(50, hold=1.0))  # too shallow
    plant(t, free, 77, Shape(300, hold=0.06))  # too brief
    plant(t, free, 90, Shape(300, 0.2, fall=None))  # a lasting drop

    found = spikes.find_spikes(t, free)
    assert [s.ending for s in found] == ["recovered"] * 5 + ["lasting", "recovered"], \
        [(s.start_ns - t[0]) / 1e9 for s in found]
    near, drop = found[4], found[5]

    for spike, (at, shape) in zip(found[:4] + found[6:], planted.items()):
        want = expected(shape)
        baseline = BASE - (300 * MiB if at > 90 else 0)  # after the drop, its level
        assert abs(spike.baseline_bytes - baseline) <= 2 * MiB, at
        assert abs(spike.amplitude_bytes - want["amplitude"]) <= 8 * MiB, at
        assert spike.rise_s == pytest.approx(want["rise"], abs=want["tol"]), at
        assert spike.duration_s == pytest.approx(want["duration"], abs=want["tol"]), at
        assert spike.recovery_s == pytest.approx(want["recovery"], abs=want["tol"]), at
        # A crossing is interpolated, so a sheer edge reads up to a period early.
        assert at - PERIOD <= (spike.start_ns - t[0]) / 1e9 <= at + shape.rise, at
        assert spike.start_ns <= spike.peak_start_ns <= spike.peak_end_ns <= spike.end_ns

    want = expected(near_threshold)
    assert near.duration_s == pytest.approx(want["duration"], abs=want["tol"])

    assert drop.end_ns is None and drop.recovery_s is None
    assert abs(drop.amplitude_bytes - 300 * MiB) <= 8 * MiB
    assert drop.rise_s == pytest.approx(0.16, abs=1.5 * PERIOD + 0.01)
    assert drop.duration_s == pytest.approx(5.0, abs=2 * PERIOD)  # held for the window


def test_the_noise_measured_on_this_machine_makes_no_spikes():
    """Half an hour of the idle noise, on a baseline that drifts by 40 MiB
    over minutes, as the desktop's own memory does."""
    t, free = quiet(30 * 60, seed=1)
    minutes = (t - t[0]) / 60e9
    free += (20 * MiB * np.sin(2 * np.pi * minutes / 5)).astype(np.int64)
    assert spikes.find_spikes(t, free) == []


def test_the_window_threshold_and_minimum_duration_are_parameters():
    """At their defaults of 5 s, 64 MiB and 100 ms, only the drop below is a
    spike; each parameter changed lets one more through."""
    t, free = quiet(40)
    plant(t, free, 10, Shape(50, hold=0.5))  # 50 MiB: a spike only at a 32 MiB threshold
    plant(t, free, 20, Shape(200, hold=0.06))  # 60 ms: a spike only at a 50 ms minimum
    plant(t, free, 30, Shape(200, fall=None))  # a drop: held for the window, whatever it is
    assert len(spikes.find_spikes(t, free)) == 1
    assert len(spikes.find_spikes(t, free, threshold_bytes=32 * MiB)) == 2
    assert len(spikes.find_spikes(t, free, min_duration_s=0.05)) == 2
    drop = spikes.find_spikes(t, free, window_s=2.0)[-1]
    assert drop.ending == "lasting" and drop.duration_s == pytest.approx(2.0, abs=2 * PERIOD)


def test_what_a_trace_does_not_show_is_not_measured():
    """A measure stops at the next spike and at a gap, and says None rather
    than run into either: two spikes with too little between them for either
    to have recovered to 10% get neither that recovery nor the next rise. A
    gap, or the end of the trace, cuts a spike short, and says so."""
    t, free = quiet(30, noise=False)
    plant(t, free, 10, Shape(300, 0.2, 0.5, 0.2))
    plant(t, free, 11.1, Shape(300, 0.2, 0.5, 0.2))
    between = (t - t[0] >= 10.7e9) & (t - t[0] < 11.3e9)
    free[between] = np.minimum(free[between], BASE - 31 * MiB)  # under release, over 10%
    plant(t, free, 20, Shape(300, hold=2.0))
    keep = ~((t - t[0] >= 21e9) & (t - t[0] < 22e9))  # a second of missed deadlines
    plant(t, free, 29.5, Shape(300, hold=5))  # still low at the end
    t, free = t[keep], free[keep]

    first, second, gapped, ended = spikes.find_spikes(t, free)
    assert first.recovered and first.recovery_s is None and first.rise_s is not None
    assert second.recovered and second.rise_s is None and second.recovery_s is not None
    assert gapped.ending == "censored" and gapped.duration_s == pytest.approx(1.0, abs=PERIOD)
    assert ended.ending == "censored" and ended.recovery_s is None

    with pytest.raises(ValueError, match="increasing"):
        spikes.find_spikes(t[::-1], free)
    with pytest.raises(ValueError, match="positive"):
        spikes.find_spikes(t, free, window_s=0)
