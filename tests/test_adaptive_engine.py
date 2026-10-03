"""Pressure events make the engine plan and downgrade between steps (#105,
Seam A).

With kv_adaptive, the engine reacts to the monitor's events: on YELLOW or
RED it makes a plan from the scores (#101, #104) and applies it between
decoding steps, synchronously (#88): at RED all at once, at YELLOW a step
at a time, each step within the budget, planning included. Every plan is
recorded with the event that caused it.

The pressure here is simulated: a monitor reads a headroom the test
chooses. What the engine does about it is real: pages are downgraded, and
what this process holds by the driver's own account falls.
"""

import tempfile
from pathlib import Path

import numpy as np
import pytest
from test_simulator import simulating

from conftest import require_model
from microinfer import Engine, _microinfer, monitor
from microinfer.golden import GoldenError, GoldenSet
from test_static_tiers import REPO

MIB = 2**20


def simulated(headroom_mib):
    """A monitor that reads `headroom_mib` MiB of headroom at every poll."""
    return monitor.Monitor(reader=lambda: headroom_mib * MIB, poll_s=0.002)


def golden_ids(name, prompt_id):
    try:
        return GoldenSet(REPO / "tests" / "golden" / name)[prompt_id].token_ids
    except GoldenError as exc:
        pytest.skip(str(exc))


@pytest.mark.slow
def test_a_simulated_red_makes_the_engine_return_memory_the_driver_sees():
    """RED: the plan is applied at once, in the step after the event, every
    downgrade of it; the cache's allocator gives memory back, and what this
    process holds by the driver's account falls by as much, to a granule.
    The plan is recorded with the RED event that caused it. A cache of
    2048 positions, so that what the downgrades free is granules: on a few
    hundred positions the FP16 range's last granule freed is the INT2
    range's first one mapped."""
    name = "qwen2.5-0.5b-instruct"
    engine = Engine(require_model(name), kv_adaptive=True, prefill_chunk=None)
    engine.load_weights()
    engine.measure_plans = True
    ids = np.random.default_rng(106).integers(1000, 100_000, 2048).astype(np.int32)
    engine.start_monitor(simulated(100))
    try:
        engine.generate(ids, 16, stop_at_eos=False)
    finally:
        engine.stop_monitor()
    red = [p for p in engine.plans if p.plan.level == monitor.RED]
    assert red, [p.pressure.event.level for p in engine.plans]
    record = red[0]
    assert record.pressure.event.level == monitor.RED
    assert record.pressure in engine.pressure_events
    assert record.ended == "applied" and len(record.batches) == 1
    assert record.applied == len(record.plan) > 0
    batch = record.batches[0]
    returned = batch.cache_bytes_before - batch.cache_bytes_after
    seen = batch.own_bytes_before - batch.own_bytes_after
    print(f"\nRED: {len(record.plan)} downgrades, the allocator returned {returned / MIB:.1f} MiB, "
          f"the driver saw {seen / MIB:.1f} MiB")
    assert returned > 0
    assert abs(seen - returned) <= _microinfer.granule_bytes()


@pytest.mark.slow
def test_at_yellow_no_step_is_delayed_beyond_the_budget():
    """YELLOW: the plan is applied over several steps, each step's planning
    and downgrades within the budget, the batches one after another, until
    every downgrade is applied. Qwen2.5-1.5B, whose downgrades #97 timed,
    with the whole prompt prefilled before the first drain."""
    name = "qwen2.5-1.5b-instruct"
    engine = Engine(require_model(name), kv_adaptive=True, prefill_chunk=None)
    engine.load_weights()
    ids = np.random.default_rng(105).integers(1000, 100_000, 2048).astype(np.int32)
    engine.start_monitor(simulated(1000))
    try:
        engine.generate(ids, 40, stop_at_eos=False)
    finally:
        engine.stop_monitor()
    yellow = [p for p in engine.plans if p.plan.level == monitor.YELLOW]
    assert yellow
    record = yellow[0]
    budget = engine.kv_plan_budget_seconds
    seconds = [b.seconds for b in record.batches]
    print(f"\nYELLOW: {len(record.plan)} downgrades over {len(record.batches)} steps, "
          f"planning {record.planning_seconds * 1e3:.1f} ms, the longest step "
          f"{max(seconds) * 1e3:.1f} ms of a {budget * 1e3:.0f} ms budget")
    assert len(record.batches) > 1
    assert max(seconds) <= budget
    assert [b.start for b in record.batches[1:]] == [b.end for b in record.batches[:-1]]
    assert record.ended == "applied" and record.applied + record.skipped == len(record.plan)


def test_an_engine_that_does_not_adapt_only_records():
    """Without kv_adaptive the engine records the events, as it did, and
    makes no plan."""
    name = "qwen2.5-0.5b-instruct"
    engine = Engine(require_model(name))
    engine.load_weights()
    engine.start_monitor(simulated(100))
    try:
        engine.generate(golden_ids(name, "medium-01"), 8, stop_at_eos=False)
    finally:
        engine.stop_monitor()
    assert engine.pressure_events and engine.plans == []
    with pytest.raises(ValueError, match="adaptive"):
        Engine(require_model(name), kv_adaptive=True, kv_cache="contiguous")
    with pytest.raises(ValueError, match="kv_score_source"):
        Engine(require_model(name), kv_score_source="entropy")


def test_a_red_mid_decode_plans_from_the_scorers_scores():
    """A RED that comes while decoding, once the scorer has observed pages,
    is planned from its semantic scores, not from neutral ones: the plan's
    scores differ page to page, and it is applied all at once."""
    name = "qwen2.5-0.5b-instruct"
    engine = Engine(require_model(name), kv_adaptive=True, prefill_chunk=None)
    engine.load_weights()
    ids = np.random.default_rng(107).integers(1000, 100_000, 1024).astype(np.int32)

    def headroom():  # GREEN until decoding is well under way, then RED
        cache = engine._cache
        decoding = cache is not None and cache.length > len(ids) + 12
        return (100 if decoding else 2000) * MIB

    engine.start_monitor(monitor.Monitor(reader=headroom, poll_s=0.002))
    try:
        engine.generate(ids, 40, stop_at_eos=False)
    finally:
        engine.stop_monitor()
    red = [p for p in engine.plans if p.plan.level == monitor.RED]
    assert red and red[0].pressure.positions_held > len(ids) + 12
    scores = red[0].plan.scores
    assert np.isfinite(scores).all() and np.unique(scores).size > 10
    assert red[0].ended == "applied" and len(red[0].batches) == 1


@pytest.mark.slow
def test_pressure_that_persists_is_planned_for_again_as_pages_are_sealed():
    """A RED that holds: the first plan takes every page outside the recency
    floor, short of its target; as decoding seals more pages and they leave
    the floor, the engine plans again, for the headroom it reads now, and
    downgrades them too. Such a plan is marked as answering pressure that
    persisted, with the event that began it."""
    name = "qwen2.5-0.5b-instruct"
    engine = Engine(require_model(name), kv_adaptive=True, prefill_chunk=None)
    engine.load_weights()
    ids = np.random.default_rng(108).integers(1000, 100_000, 1024).astype(np.int32)
    engine.start_monitor(simulated(100))
    try:
        engine.generate(ids, 3 * _microinfer.device.page_tokens + 4, stop_at_eos=False)
    finally:
        engine.stop_monitor()
    first, *later = engine.plans
    assert first.plan.level == monitor.RED and first.plan.short and not first.persisting
    assert later and all(p.persisting and p.pressure is first.pressure for p in later)
    assert all(p.headroom_bytes == 100 * MIB and p.ended == "applied" for p in later)
    newest = int(first.plan.pages.max())
    assert all(int(p.plan.pages.min()) > newest for p in later)


@pytest.mark.slow
def test_real_contention_makes_the_engine_downgrade():
    """Not a monitor told what to read: the contention simulator, a process
    of its own, takes device memory mid-decode until headroom by NVML is
    RED, and the engine, its monitor reading NVML, answers with a plan that
    returns memory. Qwen2.5-0.5B on 2048 positions, the simulator taking all
    but 380 MiB of what the engine left free at its peak a run before."""
    name = "qwen2.5-0.5b-instruct"
    engine = Engine(require_model(name), kv_adaptive=True, prefill_chunk=None)
    engine.load_weights()
    ids = np.random.default_rng(109).integers(1000, 100_000, 2048).astype(np.int32)
    engine.generate(ids, 96, stop_at_eos=False)  # warm, and the free memory at its peak
    left = engine.peak_footprint().device_free
    take = left - 380 * MIB
    assert take > 0
    with tempfile.TemporaryDirectory() as tmp, simulating(Path(tmp), [(0.0, 0), (1.0, take),
                                                                        (60.0, take)]):
        engine.start_monitor()
        try:
            engine.generate(ids, 96, stop_at_eos=False)
        finally:
            engine.stop_monitor()
    red = [p for p in engine.plans if p.plan.level == monitor.RED]
    assert red, [r.event.level for r in engine.pressure_events]
    record = red[0]
    assert record.applied > 0
    assert record.batches[0].cache_bytes_before > record.batches[0].cache_bytes_after
