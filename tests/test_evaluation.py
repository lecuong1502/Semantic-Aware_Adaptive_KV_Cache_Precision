"""Evaluating the pressure monitor on synthetic contention (#64).

Ground truth is the recorder's trace with the monitor's thresholds applied,
never the monitor's own events. Tested on traces and events written here,
whose every episode is known, and on the grid's schedule.
"""

import numpy as np
import pytest

from conftest import each
from microinfer import evaluation, patterns
from microinfer.footprint import MIB
from microinfer.monitor import GREEN, RED, YELLOW, PressureEvent, Thresholds

T = Thresholds(red_below_bytes=512 * MIB, yellow_below_bytes=1024 * MIB, persist_polls=3)
MS = 1_000_000


def trace(levels_ms):
    """A 50 Hz trace of headroom from (from_ms, headroom) steps, to 10 s."""
    t = np.arange(0, 10_000, 20) * MS
    free = np.zeros(len(t), dtype=np.int64)
    for start, headroom in levels_ms:
        free[t >= start * MS] = headroom
    return t, free


def red(at_ms, previous=YELLOW):
    return PressureEvent(at_ms * MS, 0, previous, RED, 100 * MIB)


def test_true_red_episodes_come_from_the_trace_by_the_thresholds():
    """Runs of samples below RED, from the first below to the first back at
    or above; one the trace ends in runs to its end."""
    t, free = trace([(0, 2000 * MIB), (1000, 100 * MIB), (1300, 800 * MIB), (5000, 511 * MIB),
                     (5040, 2000 * MIB), (9000, 0)])
    assert evaluation.true_red(t, free, T) == [(1000 * MS, 1300 * MS), (5000 * MS, 5040 * MS),
                                               (9000 * MS, 9980 * MS)]


def test_latency_false_negatives_and_false_positives_are_counted_against_the_truth():
    """An episode of 100 ms or more is detected by the first RED event from
    its start to its end, and one poll after; its latency is that event's
    time less the start. One with none is a false negative; one shorter
    than 100 ms is neither. A RED event with no true RED over the K polls
    that settled it, and one poll either side, is a false positive."""
    episodes = [(1000 * MS, 1300 * MS), (3000 * MS, 3500 * MS), (5000 * MS, 5040 * MS),
                (7000 * MS, 7200 * MS)]

    def scores(events, latencies_ms, false_negatives, false_positives):
        s = evaluation.score(episodes, events, T)
        assert s.latencies_ms == pytest.approx(latencies_ms)
        assert (s.false_negatives, s.false_positives) == (false_negatives, false_positives)
        assert s.episodes == 3  # the 40 ms one is too short to count

    each([
        ([red(1150), red(3160), red(7240)], [150, 160, 240], 0, 0),
        # The second episode missed; an event at 6 s, far from any, is false.
        ([red(1150), red(6000), red(7150)], [150, 150], 1, 1),
        # Only RED events count, and only the first in an episode.
        ([PressureEvent(1150 * MS, 0, GREEN, YELLOW, 0), red(1160), red(1250), red(3100),
          red(7100)], [160, 100, 100], 0, 0),
        # An event just after a 40 ms episode is not false: RED was there.
        ([red(1150), red(3150), red(5100), red(7150)], [150, 150, 150], 0, 0),
    ], scores)


def test_the_grid_schedule_takes_a_base_then_each_cell_in_turn():
    """The base is taken first and kept, so that headroom starts in GREEN;
    each cell's spikes follow in turn, a gap apart; and the schedule gives
    everything back at its end. Each spike's start is known, for grouping."""
    cells = [evaluation.Cell(256 * MIB, rise_s=0.0, plateau_s=0.5),
             evaluation.Cell(512 * MIB, rise_s=0.2, plateau_s=0.1)]
    grid = evaluation.grid_schedule(cells, base_bytes=1024 * MIB, repeats=2, gap_s=1.0,
                                    lead_s=2.0, resolution_s=0.02)
    s = grid.schedule
    assert s.at(0.0) == 1024 * MIB and s.at(1.9) == 1024 * MIB
    assert [c for c, _ in grid.spikes] == [0, 0, 1, 1]
    starts = [t for _, t in grid.spikes]
    assert starts == pytest.approx([2.0, 3.5, 5.0, 6.5])
    assert s.at(2.25) == 1280 * MIB and s.at(3.0) == 1024 * MIB
    assert s.at(5.0 + 0.2 + 0.05) == 1536 * MIB
    assert s.points[-1][1] == 0 and s.duration_s > starts[-1] + 0.6
    assert all(b % patterns.granule_bytes() == 0 for _, b in s.points)
    # Scored per cell: each spike's episodes and events, to the next spike's start.
    started = 10**12
    at = [started + int(t * 1e9) for _, t in grid.spikes]
    episodes = [(at[0] + 10 * MS, at[0] + 400 * MS), (at[2] + 50 * MS, at[2] + 300 * MS)]
    events = [red((at[0] + 160 * MS) // MS), red((at[1] + 100 * MS) // MS)]
    first, second = evaluation.score_cells(grid, started, episodes, events, T)
    assert (first.episodes, first.latencies_ms, first.false_positives) == (1, [150.0], 1)
    assert (second.episodes, second.false_negatives, second.red_events) == (1, 1, 0)
