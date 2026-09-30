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


def back(at_ms):
    """The monitor leaving RED."""
    return PressureEvent(at_ms * MS, 0, RED, YELLOW, 800 * MIB)


def episode(start_ms, end_ms):
    return evaluation.Episode(start_ms * MS, end_ms * MS)


def test_true_red_episodes_come_from_the_trace_by_the_thresholds():
    """Runs of samples below RED, from the first below to the first back at
    or above; one the trace ends in runs to its end."""
    t, free = trace([(0, 2000 * MIB), (1000, 100 * MIB), (1300, 800 * MIB), (5000, 511 * MIB),
                     (5040, 2000 * MIB), (9000, 0)])
    assert evaluation.true_red(t, free, T) == [episode(1000, 1300), episode(5000, 5040),
                                               episode(9000, 9980)]


def test_latency_false_negatives_and_false_positives_are_counted_against_the_truth():
    """An episode of 100 ms or more is detected by the first unused RED event
    from its start to one poll after its end: its latency, in ms and polls.
    One with none is a false negative, counted apart if shorter than K + 1
    polls, as the monitor cannot catch it; one that begins with the monitor
    still RED is counted apart too; one shorter than 100 ms is none of them.
    A RED event with no true RED over the K polls that settled it, one poll
    either side, is a false positive. `within` keeps what begins in it."""
    episodes = [episode(1000, 1300), episode(3000, 3500), episode(5000, 5040),
                episode(7000, 7120)]
    recovered = [back(1400), back(3600), back(7300)]

    def scores(events, latencies_ms, missed, below_k, already, false_positives,
               within=None):
        s = evaluation.score(episodes, events, T, within=within)
        assert s.latencies_ms == pytest.approx(latencies_ms)
        assert s.latencies_polls == pytest.approx([ms / 50 for ms in latencies_ms])
        assert (s.missed_detectable, s.missed_below_k, s.already_red, s.false_positives) == (
            missed, below_k, already, false_positives)

    each([
        # The 120 ms episode ends before K polls could report it; the late
        # event after it is not false, for RED was there.
        ([red(1150), red(3160), red(7240)] + recovered, [150, 160], 0, 1, 0, 0),
        # The second episode missed; an event at 6 s, far from any, is false,
        # and the monitor is still RED when the last episode begins.
        ([red(1150), red(6000)] + recovered, [150], 1, 0, 1, 1),
        # Only RED events count, and each detects one episode at most.
        ([PressureEvent(1150 * MS, 0, GREEN, YELLOW, 0), red(1160), red(1250), red(3100),
          red(7100)] + recovered, [160, 100, 100], 0, 0, 0, 0),
        # Never back from RED: the later episodes begin already RED.
        ([red(1150)], [150], 0, 0, 2, 0),
        ([red(1150), red(3160)] + recovered, [160], 0, 0, 0, 0, (2000 * MS, 4000 * MS)),
    ], scores)


def test_the_grid_schedule_takes_a_base_then_each_cell_in_turn_and_is_scored_per_cell():
    """The base is taken first and kept, so that headroom starts in GREEN;
    each cell's pulses follow in turn, a gap apart, seeded gaps a fraction
    of a poll longer; and the schedule gives everything back at its end.
    Each pulse's start is when the simulator applied its first change, and
    a cell is scored on what begins from its pulses' starts to the next."""
    cells = [evaluation.Cell(256 * MIB, ramp_s=0.0, plateau_s=0.5),
             evaluation.Cell(512 * MIB, ramp_s=0.2, plateau_s=0.1)]
    grid = evaluation.grid_schedule(cells, base_bytes=1024 * MIB, repeats=2, gap_s=1.0,
                                    lead_s=2.0, resolution_s=0.02)
    s = grid.schedule
    assert s.at(0.0) == 1024 * MIB and s.at(1.9) == 1024 * MIB
    assert [p.cell for p in grid.pulses] == [0, 0, 1, 1]
    starts = [p.start_s for p in grid.pulses]
    assert starts == pytest.approx([2.0, 3.5, 5.0, 6.5])
    assert s.at(2.25) == 1280 * MIB and s.at(3.0) == 1024 * MIB
    assert s.at(5.0 + 0.2 + 0.05) == 1536 * MIB
    assert s.points[-1][1] == 0 and s.duration_s > starts[-1] + 0.6
    assert all(b % patterns.granule_bytes() == 0 for _, b in s.points)
    jittered = [p.start_s for p in evaluation.grid_schedule(
        cells, 1024 * MIB, repeats=2, gap_s=1.0, lead_s=2.0, seed=7).pulses]
    extra = np.diff(jittered) - np.diff(starts)
    assert all(0 <= x < 0.05 for x in extra) and len(set(np.round(extra, 9))) > 1

    started = 10**12
    # The simulator's changes: every point of the schedule, applied 3 ms late.
    changes = [{"scheduled_ns": started + int(t * 1e9), "applied_ns": started + int(t * 1e9)
                + 3 * MS} for t, _ in s.points]
    at = evaluation.applied_starts(grid, started, changes)
    # A ramp's first change is one step of the schedule's resolution in.
    assert all(3 * MS <= a - started - int(t * 1e9) <= 23 * MS for a, t in zip(at, starts))
    ms = [a // MS for a in at]
    episodes = [episode(ms[0] + 10, ms[0] + 400), episode(ms[2] + 50, ms[2] + 300)]
    events = [red(ms[0] + 160), back(ms[0] + 500), red(ms[1] + 100), back(ms[1] + 400)]
    first, second = evaluation.score_cells(grid, at, started + int(s.duration_s * 1e9),
                                           episodes, events, T)
    assert (first.episodes, first.latencies_ms, first.false_positives) == (1, [150.0], 1)
    assert (second.episodes, second.missed_detectable, second.red_events) == (1, 1, 0)
