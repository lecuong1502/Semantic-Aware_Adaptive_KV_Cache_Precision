"""A controlled plan for a byte target and a score source (#142).

RQ3 compresses a cache between two turns by a byte target the experiment
chooses, not by a pressure event's headroom (ADR-0017): a share of the FP16
bytes of the pages outside the recency floor. The planner is ADR-0016's,
so every source meets the same target, to within one page's move.

Two seams: the controller's pure planning for a target, and the engine,
through a conversation (Seam A).
"""

import numpy as np
import pytest

from conftest import require_model
from microinfer import Engine, controller, model, monitor

#: Page bytes and errors of a worked example: each tier down saves less and
#: costs more per byte than the last, as the logged tiers do.
PAGE_BYTES = {"FP16": 1000, "INT8": 550, "INT4": 300, "INT2": 170}
ERRORS = {"FP16": 0.0, "INT8": 0.001, "INT4": 0.01, "INT2": 0.1}
MOVES = {(a, b): 1e-5 for a in controller.TIERS for b in controller.TIERS if a != b}


def cache(layers=2, pages=10):
    """Every page at FP16, as tier indices; uniform scores."""
    return np.zeros((layers, pages), dtype=np.int8), np.ones((layers, pages))


def test_a_share_is_of_the_fp16_bytes_outside_the_recency_floor():
    """320 positions on pages of 32, the last 128 the floor: pages 0 to 5 of
    each layer are candidates, 12 pages, 12000 FP16 bytes. 50% of them is
    6000 to reclaim, 80% is 9600."""
    tiers, _ = cache()
    for share, want in ((0.5, 6000), (0.8, 9600)):
        assert controller.share_target_bytes(
            share, tiers, PAGE_BYTES, positions=320, page_tokens=32, recency_floor=128) == want


def test_what_a_page_already_gave_back_counts_toward_the_share():
    """One candidate already at INT8 holds 550, not 1000: 80% of 12000 leaves
    2400, so 11550 - 2400 = 9150 is left to reclaim."""
    tiers, _ = cache()
    tiers[0, 0] = controller.TIERS.index("INT8")
    assert controller.share_target_bytes(
        0.8, tiers, PAGE_BYTES, positions=320, page_tokens=32, recency_floor=128) == 9150


def test_a_share_is_a_fraction():
    tiers, _ = cache()
    with pytest.raises(ValueError):
        controller.share_target_bytes(1.2, tiers, PAGE_BYTES, positions=320, page_tokens=32,
                                      recency_floor=128)


def test_a_plan_for_a_target_meets_it_within_one_move():
    """Equal scores go breadth first: all 12 candidates to INT8 save 5400,
    short of 6000, and 3 to INT4 at 250 each make 6150, over by less than one
    move. Nothing within the floor is named."""
    tiers, scores = cache()
    plan = controller.target_plan_arrays(
        6000, monitor.RED, scores, tiers, PAGE_BYTES, ERRORS, move_seconds=MOVES,
        positions=320, page_tokens=32)
    assert plan.reclaimed_bytes == 6150 and not plan.short
    steps = [(controller.TIERS[c], controller.TIERS[t]) for c, t in zip(plan.current, plan.target)]
    assert steps.count(("FP16", "INT8")) == 12 and steps.count(("INT8", "INT4")) == 3
    assert plan.pages.max() <= 5
    assert plan.level == monitor.RED and plan.target_bytes == 6000


def test_the_bytes_between_two_tier_maps_are_what_the_pages_gave_back():
    """Two pages to INT8 and one to INT4 give back 2 x 450 + 700."""
    before, _ = cache()
    after = before.copy()
    after[0, 0] = after[1, 3] = controller.TIERS.index("INT8")
    after[0, 1] = controller.TIERS.index("INT4")
    assert controller.bytes_given_back(before, after, PAGE_BYTES) == 1600


def test_a_target_out_of_reach_is_short():
    """Every candidate at INT2 saves 12 x 830 = 9960, short of 10000."""
    tiers, scores = cache()
    plan = controller.target_plan_arrays(
        10000, monitor.RED, scores, tiers, PAGE_BYTES, ERRORS, move_seconds=MOVES,
        positions=320, page_tokens=32)
    assert plan.short and plan.reclaimed_bytes == 9960


def test_a_plan_for_its_headroom_is_unchanged():
    """The plan a pressure event makes is the plan for the target its
    headroom gives: T_high + margin - headroom."""
    tiers, scores = cache()
    thresholds = monitor.DEFAULT
    headroom = thresholds.red_below_bytes - 1
    target = thresholds.yellow_below_bytes + controller.DEFAULT_MARGIN_BYTES - headroom
    by_headroom = controller.plan_arrays(headroom, thresholds, scores, tiers, PAGE_BYTES, ERRORS,
                                         move_seconds=MOVES, positions=320, page_tokens=32)
    by_target = controller.target_plan_arrays(target, monitor.RED, scores, tiers, PAGE_BYTES,
                                              ERRORS, move_seconds=MOVES, positions=320,
                                              page_tokens=32)
    assert by_headroom.target_bytes == by_target.target_bytes
    np.testing.assert_array_equal(by_headroom.pages, by_target.pages)
    np.testing.assert_array_equal(by_headroom.lower, by_target.lower)


# -- through a conversation ---------------------------------------------------------------

NAME = "qwen2.5-0.5b-instruct"


def prompt(n, seed):
    return np.random.default_rng(seed).integers(1000, 100_000, n).astype(np.int32)


@pytest.fixture(scope="module")
def engine():
    e = Engine(require_model(NAME), kv_adaptive=True, kv_scoring=True, prefill_chunk=None)
    e.load_weights()
    return e


def planned(engine, share, source):
    """Turn 1 over 2048 positions, then a controlled plan: its record."""
    with engine.conversation() as chat:
        chat.turn(prompt(2048, 142), 16, stop_at_eos=False)
        record = chat.plan(share, source=source, seed=7)
        chat.turn(prompt(8, 1420), 2, stop_at_eos=False)  # the cache reads on after it
    return record


@pytest.mark.slow
@pytest.mark.parametrize("share", [0.7, 0.8])
def test_every_source_meets_a_share_within_one_move_and_the_same_bytes(engine, share):
    """At 70% and 80%, semantic, uniform and random each meet the target,
    over it by less than one page's move, the record saying so, and so
    reclaim the same bytes, to within one move."""
    page = dict(zip(controller.TIERS, model.tier_page_bytes(engine.config)))
    largest_move = max(page[a] - page[b] for a, b in zip(controller.TIERS, controller.TIERS[1:]))
    records = {s: planned(engine, share, s) for s in ("semantic", "uniform", "random")}
    reclaimed = []
    for source, record in records.items():
        c = record.controlled
        assert c["source"] == source and c["share"] == share
        assert c["target_met"] and not record.plan.short
        assert c["target_bytes"] <= c["reclaimed_bytes"] < c["target_bytes"] + largest_move
        assert record.ended == "applied" and record.pressure is None
        reclaimed.append(c["reclaimed_bytes"])
    assert max(reclaimed) - min(reclaimed) < largest_move


@pytest.mark.slow
def test_the_record_gives_the_effective_bits_the_plan_leaves(engine):
    """At 80% the pages outside the floor hold at most a fifth of their FP16
    bytes: 3.2 bits an element, metadata counted; the whole cache, with the
    floor's pages at FP16, a little more."""
    c = planned(engine, 0.8, "semantic").controlled
    assert c["candidate_effective_bits"] <= 16 * 0.2 + 1e-9
    assert c["candidate_effective_bits"] < c["effective_bits"] < 16


@pytest.mark.slow
def test_the_record_gives_the_bytes_the_pages_gave_back_not_those_planned(engine, monkeypatch):
    """A plan the allocator stops before its first move records what its
    pages gave back, nothing, and that the target was not met, though the
    plan was made to meet it."""
    def stopped(self, cache, started, planned, own_before):
        self._pending.ended, self._pending = "stopped: simulated", None

    monkeypatch.setattr(Engine, "_apply", stopped)
    record = planned(engine, 0.8, "uniform")
    c = record.controlled
    assert record.ended == "stopped: simulated" and not record.plan.short
    assert c["target_met"] is False and c["reclaimed_bytes"] == 0 < c["target_bytes"]


@pytest.mark.slow
def test_a_plan_is_refused_while_a_monitor_runs(engine):
    """A controlled plan involves no monitor: one running could replace it,
    add to it, or upgrade its pages back."""
    with engine.conversation() as chat:
        chat.turn(prompt(64, 144), 2, stop_at_eos=False)
        engine.start_monitor(monitor.Monitor(reader=lambda: 8000 * 2**20, poll_s=0.01))
        try:
            with pytest.raises(ValueError, match="monitor"):
                chat.plan(0.8, source="uniform")
        finally:
            engine.stop_monitor()


@pytest.mark.slow
def test_a_plan_needs_a_cache_that_can_downgrade():
    """A cache that does not seal holds its last page still being written:
    nothing can be downgraded, so a controlled plan is refused."""
    static = Engine(require_model(NAME), prefill_chunk=None)
    static.load_weights()
    with static.conversation() as chat:
        chat.turn(prompt(64, 143), 2, stop_at_eos=False)
        with pytest.raises(ValueError, match="adaptive"):
            chat.plan(0.8, source="uniform")
