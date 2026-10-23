"""The survival experiment (#108): a generation the simulator takes memory
from part way through, run on a static engine and on an adaptive one.

Three seams. Engine.generate reports its progress, as Engine.hold does, so
that contention can arrive at a position the experiment chooses (Seam A).
microinfer.survival holds the experiment's two pieces of pure logic: how
much the simulator takes, and what a run's plans and ending come to. And the
run itself, at a small scale: Qwen2.5-0.5B beside the real simulator.
"""

import gc
import importlib.util
from pathlib import Path

import numpy as np
import pytest

from conftest import require_model
from microinfer import Engine, ModelConfig, controller, monitor, replay, survival
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
    w = survival.waits(events, [])
    assert w["events"][1] == {"level": "RED", "previous": "GREEN", "headroom_mib": 600.0,
                              "positions": 16896, "wait_ms": 30.0, "to_plan_ms": None}
    assert w["events"][0]["previous"] is None
    assert w["pressure_wait_ms"] == {"median": 465.0, "p90": 813.0, "max": 900.0}


def test_an_event_s_time_to_its_plan_is_its_wait_and_the_plan_s_making():
    """The plan made on the RED event took 4 ms to make, after the event's
    30 ms wait: 34 ms from the event to its plan. A persisting plan is made
    on no event, and counts for none."""
    plan = red_at_ten_seconds()
    later = PlanRecord(plan.pressure, Moves(monitor.RED, 1, 1), 0.002, [], persisting=True)
    w = survival.waits([plan.pressure], [plan, later])
    assert w["events"][0]["to_plan_ms"] == 34.0
    assert w["to_plan_ms"] == {"median": 34.0, "p90": 34.0, "max": 34.0}


def test_no_pressure_has_no_spread():
    w = survival.waits([record(monitor.GREEN, None, 1.0, 5.0, 0)], [])
    assert w["pressure_wait_ms"] is None and w["to_plan_ms"] is None


# -- what a pulse and release comes to (#155) ---------------------------------------------

def upgrades(n):
    """An upgrade plan as recovery reads it, n upgrades."""
    z = np.zeros(n, dtype=np.int64)
    return controller.UpgradePlan(monitor.GREEN, 0, "", z, z, z, z, z, np.zeros(n),
                                  np.zeros(n), np.zeros(n), 0.05)


def applied(plan, *at, ended="applied"):
    return PlanRecord(None, plan, 0.001, [AppliedBatch(0, 1, 0.01, 0, 0, at_seconds=t)
                                          for t in at], ended=ended)


def step(t, bits, below):
    return {"t": t, "state": "decoding", "position": 0, "effective_bits": bits,
            "pages_below_fp16": below}


def pulse_restored():
    """The take at 100 s; a RED plan applied at 100.1 s and 100.3 s; the
    pulse ends at 130 s; upgrades from 135.5 s to 137 s; the cache back at
    FP16 at the step of 137.2 s."""
    plans = [applied(Moves(monitor.RED, 1, 2), 100.1, 100.3),
             applied(upgrades(3), 135.5, 136.0, 137.0)]
    timeline = [step(99.0, 16.0, 0), step(100.5, 4.0, 500), step(130.0, 4.0, 500),
                step(135.5, 6.0, 300), step(137.2, 16.0, 0), step(140.0, 16.0, 0)]
    return timeline, plans


def test_a_pulse_restored_says_when_and_how_long_after_pressure_ended():
    timeline, plans = pulse_restored()
    r = survival.recovery(timeline, plans, took_at_seconds=100.0, pressure_ended_seconds=130.0,
                          cooldown_seconds=5.0)
    assert r["pressure_ended_s"] == 30.0
    assert r["last_downgrade_s"] == pytest.approx(0.3)
    assert r["first_upgrade_s"] == 35.5 and r["last_upgrade_s"] == 37.0
    assert r["first_upgrade_after_cooldown"] is True
    assert r["restored"] is True and r["time_to_restore_s"] == pytest.approx(7.2)
    assert r["thrash"] == {"switches": 1, "downgrade_plans_after_first_upgrade": 0,
                           "upgrade_plans_cancelled_by_pressure": 0}


def test_the_effective_bits_are_averaged_over_time_each_step_s_held_until_the_next():
    """From the take: 4 bits for 35 s, 6 for 1.7 s, 16 for 2.8 s; under
    pressure, 4; after it, 4 for 5.5 s, 6 for 1.7 s, 16 for 2.8 s."""
    timeline, plans = pulse_restored()
    bits = survival.recovery(timeline, plans, took_at_seconds=100.0,
                             pressure_ended_seconds=130.0, cooldown_seconds=5.0)["effective_bits"]
    assert bits["mean"] == pytest.approx((4 * 35 + 6 * 1.7 + 16 * 2.8) / 39.5)
    assert bits["mean_under_pressure"] == pytest.approx(4.0)
    assert bits["mean_after_pressure"] == pytest.approx((4 * 5.5 + 6 * 1.7 + 16 * 2.8) / 10)
    assert bits["lowest"] == 4.0


def test_downgrades_and_upgrades_that_alternate_are_thrash():
    """Upgrades that pressure cancels, then a downgrade, then upgrades
    again: three turns, one downgrade plan after the first upgrade, and the
    first upgrades too early for the cooldown after the downgrade before."""
    plans = [applied(Moves(monitor.RED, 1, 1), 100.1),
             applied(upgrades(2), 102.0, ended="cancelled by pressure"),
             applied(Moves(monitor.YELLOW, 1, 1), 103.0), applied(upgrades(2), 110.0)]
    r = survival.recovery([step(111.0, 16.0, 0)], plans, took_at_seconds=100.0,
                          pressure_ended_seconds=104.0, cooldown_seconds=5.0)
    assert r["thrash"] == {"switches": 3, "downgrade_plans_after_first_upgrade": 1,
                           "upgrade_plans_cancelled_by_pressure": 1}
    assert r["first_upgrade_after_cooldown"] is False
    assert r["restored"] and r["time_to_restore_s"] == 7.0


def test_a_pulse_never_ended_is_never_restored():
    timeline, plans = pulse_restored()
    r = survival.recovery(timeline, plans[:1], took_at_seconds=100.0,
                          pressure_ended_seconds=None, cooldown_seconds=5.0)
    assert r["restored"] is False and r["time_to_restore_s"] is None
    assert r["first_upgrade_s"] is None and r["first_upgrade_after_cooldown"] is None
    assert r["effective_bits"]["mean_after_pressure"] is None


def test_the_tool_releases_a_pulse_for_an_adaptive_engine_only(tmp_path):
    path = Path(__file__).resolve().parent.parent / "tools" / "survival_experiment.py"
    spec = importlib.util.spec_from_file_location("survival_tool", path)
    tool = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tool)
    with pytest.raises(SystemExit):
        tool.main(["static", "--out", str(tmp_path / "s.csv.gz"), "--issue", "155",
                   "--pulse", "30"])


# -- the run, small -----------------------------------------------------------------------

@pytest.mark.slowest
def test_the_static_engine_runs_out_where_the_adaptive_one_survives(tmp_path):
    """The experiment at a small scale: Qwen2.5-0.5B prefills 4096 positions
    and decodes 16 tokens; at 2048 positions the simulator, a process of its
    own, takes what leaves the engine 24 MiB short of its full cache, and
    keeps it. The static engine runs out before its cache is full. The
    adaptive one, facing the same rule, downgrades, the driver sees what
    its process holds fall, and it finishes with every token."""
    ids = prompt(4096, 1081)
    context_bytes = replay.simulator_context_bytes(tmp_path)

    static = Engine(require_model(NAME))
    static.load_weights()
    ran_out = survival.run(static, ids, 16, contention_at=2048, shortfall_bytes=24 * MIB,
                           out=tmp_path / "static.csv.gz",
                           simulator_context_bytes=context_bytes)
    del static
    gc.collect()
    assert ran_out.summary["survived"] is False
    assert ran_out.summary["ending"]["state"] == survival.OUT_OF_MEMORY
    assert 2048 <= ran_out.summary["ending"]["positions"] < 4096 + 15
    assert ran_out.contention_position >= 2048 and ran_out.taken_bytes > 0

    adaptive = Engine(require_model(NAME), kv_adaptive=True)
    adaptive.load_weights()
    survived = survival.run(adaptive, ids, 16, contention_at=2048, shortfall_bytes=24 * MIB,
                            out=tmp_path / "adaptive.csv.gz",
                            simulator_context_bytes=context_bytes)
    del adaptive
    gc.collect()
    s = survived.summary
    print(f"\nstatic ran out at {ran_out.summary['ending']['positions']}; adaptive: "
          f"{s['totals']}")
    assert s["survived"] is True
    assert s["ending"] == {"state": COMPLETE, "positions": 4096 + 15, "tokens": 16, "reason": ""}
    assert s["totals"]["downgrades_applied"] > 0
    assert s["totals"]["returned_mib"] > 0
    assert s["totals"]["every_plan_seen_within_granule"] is True
    assert (tmp_path / "adaptive.csv.gz").exists()


@pytest.mark.slowest
def test_a_pulse_released_is_restored_after_the_cooldown(tmp_path):
    """The pulse at a small scale: at 2048 positions the simulator takes
    what leaves Qwen2.5-0.5B 24 MiB short and gives it back 2 s later. The
    engine downgrades, and once the pulse has passed and the cooldown run,
    its upgrades bring every page back to FP16 before the generation ends,
    with no thrash."""
    ids = prompt(4096, 1551)
    context_bytes = replay.simulator_context_bytes(tmp_path)
    adaptive = Engine(require_model(NAME), kv_adaptive=True, kv_upgrade_cooldown_seconds=0.5)
    adaptive.load_weights()
    result = survival.run(adaptive, ids, 600, contention_at=2048, shortfall_bytes=24 * MIB,
                          out=tmp_path / "pulse.csv.gz", simulator_context_bytes=context_bytes,
                          pulse_s=2.0)
    r = survival.recovery(result.timeline, adaptive.plans,
                          took_at_seconds=result.took_at_seconds,
                          pressure_ended_seconds=result.pressure_ended_seconds,
                          cooldown_seconds=adaptive.kv_upgrade_cooldown_seconds)
    del adaptive
    gc.collect()
    print(f"\n{r}")
    assert result.summary["survived"] is True
    assert result.summary["totals"]["downgrades_applied"] > 0
    assert result.summary["totals"]["upgrades_applied"] > 0
    assert r["pressure_ended_s"] >= 2.0
    assert r["first_upgrade_after_cooldown"] is True
    assert r["restored"] and r["time_to_restore_s"] >= 0.5
    assert r["thrash"]["switches"] == 1
    assert r["effective_bits"]["lowest"] < 16 and result.timeline[-1]["effective_bits"] == 16.0
