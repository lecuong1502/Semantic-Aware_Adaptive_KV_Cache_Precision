"""Upgrades at GREEN, in the decode loop (#106, Seam A).

When the monitor reads GREEN, with headroom above T_high plus a P90 spike,
and the cooldown since the last downgrade has passed, the engine restores
pages from their FP16 shadows between steps, in the upgrade policy's order
(#103), within the same per-step budget as a YELLOW plan. A pressure event
cancels upgrades not yet made. Under a pulse train of pressure the cooldown
holds every upgrade back until the pulses stop: no downgrade-upgrade thrash.

As in test_adaptive_engine.py, the pressure is simulated, a monitor reading
a headroom the test chooses, and the moves are real.
"""

import time

import numpy as np
from test_controller import upgrade_greedy_reference

from conftest import require_model
from microinfer import Engine, _microinfer, controller, monitor

MIB = 2**20
NAME = "qwen2.5-0.5b-instruct"
#: Short, so that a test waits on it for a fraction of a second.
COOLDOWN = 0.3
#: GREEN, above T_high + 512 MiB by far: every page fits back at FP16.
AMPLE_MIB = 8000


def adaptive_engine():
    engine = Engine(require_model(NAME), kv_adaptive=True, prefill_chunk=None,
                    kv_upgrade_cooldown_seconds=COOLDOWN)
    engine.load_weights()
    return engine


def run(engine, ids, new, headroom):
    """Generate under a monitor reading `headroom(cache)`, the cache the
    engine decodes into, or None before it has one; returns the cache."""
    caches = []

    def reader():
        if engine._cache is not None and not caches:
            caches.append(engine._cache)
        return headroom(caches[0] if caches else None)

    engine.start_monitor(monitor.Monitor(reader=reader, poll_s=0.002))
    try:
        engine.generate(ids, new, stop_at_eos=False)
    finally:
        engine.stop_monitor()
    return caches[0]


def assert_restored(cache):
    """Every page of the cache that holds a shadow is back at FP16."""
    pages = cache.pages
    tiers, shadowed = pages.page_tiers(), pages.shadowed()
    fp16 = int(_microinfer.Tier.FP16.value)
    assert shadowed.any()
    assert (tiers[shadowed] == fp16).all(), np.unique(tiers[shadowed], return_counts=True)


def test_after_contention_passes_pages_return_to_fp16_in_the_policy_s_order():
    """A RED mid-decode downgrades pages; once headroom is ample again and
    the cooldown has passed, an upgrade plan restores every page with a
    shadow to FP16, each step within the budget. Its upgrades are applied
    in its order, which is the policy's: the greedy of #103 on the tiers,
    shadows and scores the engine planned from. None comes sooner than the
    cooldown after the last downgrade."""
    engine = adaptive_engine()
    planned_from = []
    real = controller.upgrade_plan_arrays

    def recording(headroom, thresholds, scores, tiers, shadowed, *args, **kwargs):
        inputs = (headroom, thresholds, scores.copy(), tiers.copy(), shadowed.copy())
        plan = real(headroom, thresholds, scores, tiers, shadowed, *args, **kwargs)
        planned_from.append((plan, inputs))
        return plan

    controller.upgrade_plan_arrays = recording
    ids = np.random.default_rng(206).integers(1000, 100_000, 2048).astype(np.int32)

    def headroom(cache):  # RED for a few decode steps, then ample
        length = cache.length if cache is not None else 0
        if len(ids) + 4 < length <= len(ids) + 8:
            return 100 * MIB
        return (AMPLE_MIB if length > len(ids) else 2000) * MIB

    try:
        cache = run(engine, ids, 160, headroom)
    finally:
        controller.upgrade_plan_arrays = real
    downgrades = [p for p in engine.plans if not p.upgrades]
    upgrades = [p for p in engine.plans if p.upgrades]
    assert downgrades and downgrades[0].plan.level == monitor.RED
    assert upgrades, "no upgrade plan was made"
    record = upgrades[0]
    plan = record.plan
    print(f"\n{len(plan)} upgrades over {len(record.batches)} steps, the longest "
          f"{max(b.seconds for b in record.batches) * 1e3:.1f} ms; "
          f"{record.batches[0].at_seconds - downgrades[-1].batches[-1].at_seconds:.2f} s "
          f"after the last downgrade")
    assert record.ended == "applied" and record.applied + record.skipped == len(plan)
    assert [b.start for b in record.batches] == [0] + [b.end for b in record.batches[:-1]]
    assert all(b.seconds <= engine.kv_plan_budget_seconds for b in record.batches)
    headroom_bytes, thresholds, scores, tiers, shadowed = next(
        inputs for made, inputs in planned_from if made is plan)
    keys = list(zip(*np.nonzero(tiers >= 0)))
    available = headroom_bytes - thresholds.yellow_below_bytes - controller.DEFAULT_UPGRADE_SPIKE_BYTES
    assert [(u.layer, u.page, u.current_tier, u.target_tier) for u in plan.upgrades] == \
        upgrade_greedy_reference(
            available, {(int(l), int(p)): float(scores[l, p]) for l, p in keys},
            {(int(l), int(p)): controller.TIERS[tiers[l, p]] for l, p in keys},
            {(int(l), int(p)) for l, p in keys if shadowed[l, p]},
            page_bytes=engine._plan_inputs[0], errors=engine._plan_inputs[1])
    assert record.batches[0].at_seconds - downgrades[-1].batches[-1].at_seconds >= COOLDOWN
    assert record.batches[-1].cache_bytes_after > downgrades[0].batches[0].cache_bytes_after
    assert_restored(cache)


def test_a_pulse_train_makes_no_downgrade_upgrade_thrash():
    """RED pulses of 40 ms every 150 ms, for a second, closer together than
    the cooldown: the engine downgrades on the first and makes no upgrade
    until the train has stopped and the cooldown passed, though the later
    pulses find nothing left to downgrade; then it restores the pages.
    Downgrades and upgrades do not alternate: every upgrade plan comes
    after every downgrade plan."""
    engine = adaptive_engine()
    ids = np.random.default_rng(207).integers(1000, 100_000, 1024).astype(np.int32)
    train = []

    def headroom(cache):
        if cache is None or cache.length <= len(ids):
            return 2000 * MIB
        now = time.monotonic()
        if not train:
            train.append(now)
        since = now - train[0]
        pulsing = since < 1.0 and since % 0.15 < 0.04
        return (100 if pulsing else AMPLE_MIB) * MIB

    cache = run(engine, ids, 400, headroom)
    kinds = ["up" if p.upgrades else "down" for p in engine.plans]
    print(f"\nplans: {kinds}")
    assert "down" in kinds and "up" in kinds
    assert kinds == sorted(kinds, key=lambda k: k == "up"), "downgrades and upgrades alternated"
    last_down = max(b.at_seconds for p in engine.plans if not p.upgrades for b in p.batches)
    first_up = min(b.at_seconds for p in engine.plans if p.upgrades for b in p.batches)
    assert last_down >= train[0]  # the pulses kept downgrading, or held the cache
    assert first_up - last_down >= COOLDOWN
    last_pulse_ended = train[0] + 0.9 + 0.04
    assert first_up - last_pulse_ended >= COOLDOWN
    assert_restored(cache)


def test_a_blip_of_pressure_while_upgrading_cancels_the_upgrades_left():
    """A RED that rises and falls within a step or two, while upgrades are
    under way, cancels those not yet made, though the latest event the step
    drains is GREEN: the cooldown runs again from the blip's end, and only
    then are the pages restored."""
    engine = adaptive_engine()
    ids = np.random.default_rng(208).integers(1000, 100_000, 2048).astype(np.int32)
    blip = []

    def headroom(cache):
        length = cache.length if cache is not None else 0
        if len(ids) + 4 < length <= len(ids) + 8:
            return 100 * MIB
        pending = engine._pending
        upgrading = pending is not None and pending.upgrades and pending.batches
        now = time.monotonic()
        if upgrading and not blip:
            blip.append(now)
        if blip and now - blip[0] < 0.012:
            return 100 * MIB
        return (AMPLE_MIB if length > len(ids) else 2000) * MIB

    cache = run(engine, ids, 200, headroom)
    upgrades = [p for p in engine.plans if p.upgrades]
    print(f"\nupgrade plans: {[(len(p.plan), p.applied, p.ended) for p in upgrades]}")
    assert blip and len(upgrades) >= 2
    first, *later = upgrades
    assert first.ended == "cancelled by pressure" and 0 < first.applied < len(first.plan)
    assert engine.pressure_ended_seconds > blip[0]
    assert later[0].batches[0].at_seconds - engine.pressure_ended_seconds >= COOLDOWN
    assert later[-1].ended == "applied"
    assert_restored(cache)
