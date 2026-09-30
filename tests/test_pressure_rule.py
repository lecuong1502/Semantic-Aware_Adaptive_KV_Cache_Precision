"""The rule that sets the pressure thresholds from RQ1 (#63, ADR-0013).

Tested on measures whose every step is known, and against the log itself:
the monitor's defaults are what the rule gives for RQ1's entries.
"""

import pytest

from conftest import each
from microinfer import monitor, pressure_rule
from microinfer.footprint import MIB
from microinfer.pressure_rule import Measures


def test_the_rule_gives_red_k_and_yellow_from_amplitude_rise_and_bandwidth():
    """RED is the P90 amplitude rounded up to 64 MiB; K the most polls whose
    detection fits in the P10 rise; YELLOW, RED plus what a fast spike takes
    while detected and acted on, rounded up, at most its amplitude."""
    def gives(amplitude, rise_s, bandwidth, red, yellow, k):
        d = pressure_rule.derive(Measures([amplitude] * 10, [rise_s] * 10, bandwidth, 8.0))
        got = (d.thresholds.red_below_bytes, d.thresholds.yellow_below_bytes,
               d.thresholds.persist_polls)
        assert got == (red, yellow, k), (got, d.workings)

    each([
        # A 0.5 s rise: K = 9, 0.5 s to detect, of a 0.625 s ramp: 160 MiB taken.
        (200 * MIB, 0.5, 1e15, 256 * MIB, 448 * MIB, 9),
        # A slow plan: 200 MiB of FP16 read and 100 written at 450 MiB/s is 0.67 s
        # more, and the whole amplitude is taken before it is done.
        (100 * MIB, 0.5, 450 * MIB, 128 * MIB, 256 * MIB, 9),
        # A 0.15 s rise leaves K = 2: 0.15 s to detect, of a 0.1875 s ramp.
        (64 * MIB, 0.15, 1e15, 64 * MIB, 128 * MIB, 2),
    ], gives)

    def refused(measures):
        with pytest.raises(ValueError):
            pressure_rule.derive(measures)

    each([Measures([], [0.5], 1e9, 8.0), Measures([MIB], [0.08], 1e9, 8.0)], refused)


def test_the_monitors_defaults_are_the_rule_applied_to_rq1s_log():
    """ADR-0013's values: RQ1's 48 spikes, the 1.5B model's decode bandwidth
    at 32K positions and INT8's effective bits, as the log holds them. If an entry is logged
    again, this fails until the ADR and the defaults are brought up to date."""
    measures = pressure_rule.measures_from_log()
    assert len(measures.amplitudes_bytes) == 48
    assert pressure_rule.derive(measures).thresholds == monitor.DEFAULT
    assert (monitor.DEFAULT.red_below_bytes, monitor.DEFAULT.yellow_below_bytes,
            monitor.DEFAULT.persist_polls) == (512 * MIB, 1024 * MIB, 3)
