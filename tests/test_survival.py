"""The survival experiment (#108): a generation the simulator takes memory
from part way through, run on a static engine and on an adaptive one.

Three seams. Engine.generate reports its progress, as Engine.hold does, so
that contention can arrive at a position the experiment chooses (Seam A).
microinfer.survival holds the experiment's two pieces of pure logic: how
much the simulator takes, and what a run's plans and ending come to. And the
run itself, at a small scale: Qwen2.5-0.5B beside the real simulator.
"""

import gc

import numpy as np
import pytest

from conftest import require_model
from microinfer import Engine, ModelConfig, monitor, replay, survival
from microinfer.engine import (COMPLETE, DECODING, EXHAUSTED, PREFILLING, AppliedBatch, Ending,
                               PlanRecord, PressureRecord)
from microinfer.footprint import MIB

NAME = "qwen2.5-0.5b-instruct"


def prompt(n, seed):
    return np.random.default_rng(seed).integers(1000, 100_000, n).astype(np.int32)


@pytest.fixture(scope="module")
def engine() -> Engine:
    e = Engine(require_model(NAME), prefill_chunk=256)
    e.load_weights()
    return e


def test_a_generation_reports_each_prefill_chunk_and_each_decoded_token(engine):
    """As a hold does: (PREFILLING, positions so far, None) after each chunk,
    then (DECODING, the position after it, the token) after each decoded
    token. The first new token comes from the prefill, so 8 tokens are 7
    decoding steps; the tokens reported are the tokens returned."""
    reports = []
    out = engine.generate(prompt(600, 108), 8, stop_at_eos=False,
                          report=lambda *r: reports.append(r))
    assert reports[:3] == [(PREFILLING, 256, None), (PREFILLING, 512, None),
                           (PREFILLING, 600, None)]
    decoding = reports[3:]
    assert [(s, p) for s, p, _ in decoding] == [(DECODING, 600 + k) for k in range(1, 8)]
    assert [t for _, _, t in decoding] == list(out[1:])
    assert engine.ending.state == COMPLETE


# -- how much the simulator takes ---------------------------------------------------------

def test_the_simulator_leaves_the_engine_its_shortfall_less_than_it_needs():
    """Qwen2.5-0.5B's FP16 cache is 192 MiB at 16K positions and 384 MiB at
    32K (ADR-0003), so at 16K it has 192 MiB still to grow. With 1000 MiB
    free and the simulator's context, 82 MiB, still to come, the simulator
    takes 1000 - 82 - 192 + 256 = 982 MiB for a shortfall of 256: a static
    engine would end 256 MiB short."""
    cfg = ModelConfig.from_card(NAME)
    taken = survival.contention_bytes(cfg, headroom_bytes=1000 * MIB, positions=16384,
                                      context=32768, simulator_context_bytes=82 * MIB,
                                      shortfall_bytes=256 * MIB)
    assert taken == 982 * MIB


def test_a_take_is_rounded_up_to_a_granule_so_the_shortfall_is_never_less():
    cfg = ModelConfig.from_card(NAME)
    taken = survival.contention_bytes(cfg, headroom_bytes=1000 * MIB + 1, positions=16384,
                                      context=32768, simulator_context_bytes=82 * MIB,
                                      shortfall_bytes=256 * MIB)
    assert taken == 984 * MIB


def test_an_engine_already_short_of_memory_is_refused():
    """With 100 MiB free and 192 MiB still to grow, the engine runs out with
    nothing taken: there is no experiment to run."""
    cfg = ModelConfig.from_card(NAME)
    with pytest.raises(ValueError, match="already"):
        survival.contention_bytes(cfg, headroom_bytes=100 * MIB, positions=16384,
                                  context=32768, simulator_context_bytes=82 * MIB,
                                  shortfall_bytes=0)


# -- what a run comes to ------------------------------------------------------------------

class Moves:
    """A plan as a summary reads it: its level, byte target, whether it fell
    short, and how many moves are in it."""

    def __init__(self, level, target_bytes, moves, short=False):
        self.level, self.target_bytes, self.short, self._moves = level, target_bytes, short, moves

    def __len__(self):
        return self._moves


def red_at_ten_seconds():
    """A RED event at 10 s on the monitor's clock, drained 30 ms later; its
    plan made in 4 ms and applied in one batch that ended at 10.080 s,
    freeing 400 MiB of cache, which the driver saw as 400 MiB less held."""
    event = monitor.PressureEvent(t_mono_ns=10_000_000_000, poll=200, previous=monitor.GREEN,
                                  level=monitor.RED, headroom_bytes=200 * MIB)
    pressure = PressureRecord(event, positions_held=16384, drained_ns=10_030_000_000)
    batch = AppliedBatch(0, 300, 0.050, 900 * MIB, 500 * MIB, 2000 * MIB, 1600 * MIB,
                         at_seconds=10.080)
    return PlanRecord(pressure, Moves(monitor.RED, 888 * MIB, 300, short=True), 0.004, [batch],
                      ended="applied", headroom_bytes=200 * MIB)


def test_a_run_that_completes_survived_and_its_plans_say_what_they_returned():
    later = AppliedBatch(0, 40, 0.010, 520 * MIB, 510 * MIB, 1620 * MIB, 1610 * MIB,
                         at_seconds=31.0)
    persisting = PlanRecord(red_at_ten_seconds().pressure, Moves(monitor.RED, 30 * MIB, 40),
                            0.001, [later], ended="applied", headroom_bytes=300 * MIB,
                            persisting=True)
    ending = Ending(COMPLETE, 32767, np.arange(64, dtype=np.int32))
    s = survival.summarise([red_at_ten_seconds(), persisting], ending=ending)
    assert s["survived"] is True
    assert s["ending"] == {"state": COMPLETE, "positions": 32767, "tokens": 64, "reason": ""}
    first = s["plans"][0]
    assert first == {"level": "RED", "emergency": None, "persisting": False, "upgrades": False,
                     "ended": "applied", "headroom_mib": 200.0, "target_mib": 888.0,
                     "short": True, "moves": 300, "applied": 300, "skipped": 0, "batches": 1,
                     "planning_ms": 4.0, "applying_ms": 50.0, "returned_mib": 400.0,
                     "seen_mib": 400.0, "seen_within_granule": True,
                     "event_to_applied_ms": 80.0}
    assert s["plans"][1]["persisting"] is True
    assert s["plans"][1]["event_to_applied_ms"] is None  # not made on its event
    assert s["totals"] == {"plans": 2, "emergency_plans": 0, "downgrades_applied": 340,
                           "upgrades_applied": 0, "returned_mib": 410.0, "seen_mib": 410.0,
                           "every_plan_seen_within_granule": True}


def test_a_plan_the_driver_saw_return_more_than_a_granule_less_is_flagged():
    """#88's gate: after every plan, the driver sees what the cache returned,
    to a granule. 400 MiB returned and 396 seen is two granules off."""
    record = red_at_ten_seconds()
    record.batches[0].own_bytes_after += 4 * MIB
    s = survival.summarise([record], ending=Ending(COMPLETE, 1, np.arange(1, dtype=np.int32)))
    assert s["plans"][0]["seen_mib"] == 396.0
    assert s["plans"][0]["seen_within_granule"] is False
    assert s["totals"]["every_plan_seen_within_granule"] is False


def test_a_static_run_that_ran_out_did_not_survive_and_says_where():
    s = survival.summarise([], error="out of memory: cuMemCreate", positions_reached=20480)
    assert s["survived"] is False
    assert s["ending"] == {"state": "out of memory", "positions": 20480,
                           "error": "out of memory: cuMemCreate"}
    assert s["plans"] == [] and s["totals"]["plans"] == 0


def test_an_exhausted_run_did_not_survive():
    ending = Ending(EXHAUSTED, 30000, np.arange(3, dtype=np.int32), "nothing to downgrade")
    s = survival.summarise([red_at_ten_seconds()], ending=ending)
    assert s["survived"] is False
    assert s["ending"]["state"] == EXHAUSTED and s["ending"]["reason"] == "nothing to downgrade"


def record(level, previous, at_s, wait_ms, positions):
    event = monitor.PressureEvent(t_mono_ns=int(at_s * 1e9), poll=0, previous=previous,
                                  level=level, headroom_bytes=600 * MIB)
    return PressureRecord(event, positions, int(at_s * 1e9 + wait_ms * 1e6))


def test_each_event_says_how_long_it_waited_for_a_step_boundary():
    """#135: the time from an event to the step boundary at which the engine
    drained it, for each event, and over the YELLOW and RED ones, the ones
    a plan answers. The first GREEN, when the monitor starts, answers
    nothing and is left out of the spread."""
    events = [record(monitor.GREEN, None, 1.0, 479.0, 512),
              record(monitor.RED, monitor.GREEN, 20.0, 30.0, 16896),
              record(monitor.YELLOW, monitor.RED, 21.0, 900.0, 16928)]
    w = survival.waits(events)
    assert w["events"][1] == {"level": "RED", "previous": "GREEN", "headroom_mib": 600.0,
                              "positions": 16896, "wait_ms": 30.0}
    assert w["events"][0]["previous"] is None
    assert w["pressure_wait_ms"] == {"median": 465.0, "p90": 813.0, "max": 900.0}


def test_no_pressure_has_no_spread():
    assert survival.waits([record(monitor.GREEN, None, 1.0, 5.0, 0)])["pressure_wait_ms"] is None


# -- the run, small -----------------------------------------------------------------------

@pytest.mark.slow
def test_the_static_engine_runs_out_where_the_adaptive_one_survives(tmp_path):
    """The experiment at a small scale: Qwen2.5-0.5B prefills 8192 positions
    and decodes 32 tokens; at 4096 positions the simulator, a process of its
    own, takes what leaves the engine 48 MiB short of its full cache, and
    keeps it. The static engine runs out before its cache is full. The
    adaptive one, facing the same rule, downgrades, the driver sees what
    its process holds fall, and it finishes with every token."""
    ids = prompt(8192, 1081)
    context_bytes = replay.simulator_context_bytes(tmp_path)

    static = Engine(require_model(NAME))
    static.load_weights()
    ran_out = survival.run(static, ids, 32, contention_at=4096, shortfall_bytes=48 * MIB,
                           out=tmp_path / "static.csv.gz",
                           simulator_context_bytes=context_bytes)
    del static
    gc.collect()
    assert ran_out.summary["survived"] is False
    assert ran_out.summary["ending"]["state"] == survival.OUT_OF_MEMORY
    assert 4096 <= ran_out.summary["ending"]["positions"] < 8192 + 31
    assert ran_out.contention_position >= 4096 and ran_out.taken_bytes > 0

    adaptive = Engine(require_model(NAME), kv_adaptive=True)
    adaptive.load_weights()
    survived = survival.run(adaptive, ids, 32, contention_at=4096, shortfall_bytes=48 * MIB,
                            out=tmp_path / "adaptive.csv.gz",
                            simulator_context_bytes=context_bytes)
    del adaptive
    gc.collect()
    s = survived.summary
    print(f"\nstatic ran out at {ran_out.summary['ending']['positions']}; adaptive: "
          f"{s['totals']}")
    assert s["survived"] is True
    assert s["ending"] == {"state": COMPLETE, "positions": 8192 + 31, "tokens": 32, "reason": ""}
    assert s["totals"]["downgrades_applied"] > 0
    assert s["totals"]["returned_mib"] > 0
    assert s["totals"]["every_plan_seen_within_granule"] is True
    assert (tmp_path / "adaptive.csv.gz").exists()
