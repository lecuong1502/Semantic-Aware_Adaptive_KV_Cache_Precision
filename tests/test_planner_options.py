"""Planner options for ADR-0018's ablations (#144).

Three options of the planner, each recorded in every plan: the lowest tier a
plan may use; a marginal cost of importance alone, the tier's error left
out; and a first page never downgraded. Their defaults are the policy of
ADR-0016, and leave every plan as it was.

Two seams: the controller's pure planning, and the engine, which passes its
options to every plan it makes (Seam A).
"""

import numpy as np
import pytest
from test_upgrades_at_green import AMPLE_MIB, COOLDOWN, MIB, run

from conftest import require_model
from microinfer import Engine, controller, monitor
from microinfer.controller import PlanOptions

PAGE_BYTES = {"FP16": 1000, "INT8": 550, "INT4": 300, "INT2": 170}
ERRORS = {"FP16": 0.0, "INT8": 0.001, "INT4": 0.01, "INT2": 0.1}
MOVES = {(a, b): 1e-5 for a in controller.TIERS for b in controller.TIERS if a != b}
INT2 = controller.TIERS.index("INT2")


def plan(target, scores=None, options=None, layers=2, pages=10):
    """A plan for `target` bytes over every page at FP16, 320 positions on
    pages of 32, the last 128 the floor: pages 0 to 5 are candidates."""
    tiers = np.zeros((layers, pages), dtype=np.int8)
    scores = np.ones((layers, pages)) if scores is None else scores
    extra = {} if options is None else {"options": options}
    return controller.target_plan_arrays(target, monitor.RED, scores, tiers, PAGE_BYTES, ERRORS,
                                         move_seconds=MOVES, positions=320, page_tokens=32,
                                         **extra)


def steps(p):
    return list(zip(p.layers.tolist(), p.pages.tolist(), p.current.tolist(), p.target.tolist()))


def test_the_defaults_are_adr_0016_s_policy():
    assert PlanOptions() == PlanOptions(lowest_tier="INT2", importance_only=False,
                                        protect_first_page=False)


#: The plans the planner made before #144, on main at 4febdc2, for the scores
#: of seed 144: (layer, page, step) in order, step 0 being FP16 to INT8.
BEFORE = {
    3000: [(0, 1, 0), (1, 3, 0), (0, 5, 0), (0, 3, 0), (1, 2, 0), (1, 5, 0), (1, 0, 0)],
    9000: [(0, 1, 0), (1, 3, 0), (0, 5, 0), (0, 3, 0), (1, 2, 0), (1, 5, 0), (1, 0, 0),
           (0, 4, 0), (0, 2, 0), (1, 1, 0), (0, 0, 0), (1, 4, 0),
           (0, 1, 1), (1, 3, 1), (0, 5, 1), (0, 3, 1), (1, 2, 1), (1, 5, 1), (1, 0, 1),
           (0, 4, 1), (0, 2, 1), (1, 1, 1), (0, 0, 1), (1, 4, 1),
           (0, 1, 2), (1, 3, 2), (0, 5, 2), (0, 3, 2), (1, 2, 2)],
}


def test_defaults_leave_a_plan_unchanged_bit_for_bit():
    """With no options, and with the defaults given, a plan is the one the
    planner made before #144, move for move, and records the defaults."""
    scores = np.random.default_rng(144).random((2, 10))
    for target, moves in BEFORE.items():
        for p in (plan(target, scores), plan(target, scores, PlanOptions())):
            assert list(zip(p.layers.tolist(), p.pages.tolist(), p.current.tolist())) == moves
            assert not p.short and p.options == PlanOptions()
    a, b = plan(9000, scores), plan(9000, scores, PlanOptions())
    np.testing.assert_array_equal(a.costs, b.costs)


def test_the_dict_planner_takes_the_options_too():
    """plan(), keyed by (layer, page), threads the options as plan_arrays does."""
    keys = [(layer, page) for layer in range(2) for page in range(10)]
    p = controller.plan(monitor.DEFAULT.red_below_bytes - 1, monitor.DEFAULT,
                        {k: 1.0 for k in keys}, {k: "FP16" for k in keys}, PAGE_BYTES, ERRORS,
                        move_seconds=MOVES, positions=320, page_tokens=32,
                        options=PlanOptions(protect_first_page=True))
    assert p.options.protect_first_page and 0 not in p.pages.tolist()


def test_with_int4_the_lowest_no_page_goes_to_int2():
    """Every candidate to INT4 saves 12 x 700 = 8400. A target of 8000 is
    met with INT4 alone; one of 9000, which would need INT2, is short."""
    low = PlanOptions(lowest_tier="INT4")
    met = plan(8000, options=low)
    assert not met.short and INT2 not in met.target.tolist()
    short = plan(9000, options=low)
    assert short.short and short.reclaimed_bytes == 8400
    assert INT2 not in short.target.tolist()
    assert short.options.lowest_tier == "INT4"


def test_importance_alone_takes_moves_in_score_order_regardless_of_tier_error():
    """With the tier's error left out, the least attended page goes all the
    way down before the next is touched: a page's three moves, then the
    next page's, in score order, where marginal cost would have taken every
    page to INT8 first."""
    scores = np.ones((1, 10))
    scores[0, :6] = [0.6, 0.1, 0.5, 0.2, 0.4, 0.3]
    p = plan(2490, scores, PlanOptions(importance_only=True), layers=1)
    order = [page for _, page, _, _ in steps(p)]
    assert order == [1, 1, 1, 3, 3, 3, 5, 5, 5]
    assert [s for _, _, s, _ in steps(p)][:3] == [0, 1, 2]
    by_cost = plan(2490, scores, layers=1)
    assert all(t == 1 for *_, t in steps(by_cost))  # breadth first: INT8 only


def test_a_protected_first_page_is_never_named():
    """However short of its target, a plan with the first page protected
    names no page 0 of any layer: the other 10 candidates, all to INT2,
    save 8300, short of 9000."""
    p = plan(9000, options=PlanOptions(protect_first_page=True))
    assert 0 not in p.pages.tolist()
    assert p.short and p.reclaimed_bytes == 8300


def test_an_upgrade_plan_records_the_options_too():
    """No option changes how pages move up, but a run's upgrade plans say
    which options it ran with, as its downgrade plans do: one made, and one
    the cooldown holds."""
    options = PlanOptions(lowest_tier="INT4", importance_only=True)
    tiers = np.full((2, 10), controller.TIERS.index("INT8"), dtype=np.int8)
    shadowed = np.ones((2, 10), dtype=bool)
    for last, held in ((0.0, False), (99.0, True)):
        up = controller.upgrade_plan_arrays(
            8000 * 2**20, monitor.DEFAULT, np.ones((2, 10)), tiers, shadowed, PAGE_BYTES,
            ERRORS, move_seconds=MOVES, last_downgrade_seconds=last, now_seconds=100.0,
            options=options)
        assert up.options == options and bool(up.held_by) == held
    assert controller.upgrade_plan_arrays(
        8000 * 2**20, monitor.DEFAULT, np.ones((2, 10)), tiers, shadowed, PAGE_BYTES, ERRORS,
        move_seconds=MOVES, last_downgrade_seconds=0.0, now_seconds=100.0).options == \
        PlanOptions()


def test_a_lowest_tier_is_a_quantised_tier():
    with pytest.raises(ValueError, match="lowest"):
        PlanOptions(lowest_tier="FP16")


# -- the engine ---------------------------------------------------------------------------

NAME = "qwen2.5-0.5b-instruct"


def test_the_engine_takes_plan_options_only():
    with pytest.raises(TypeError, match="PlanOptions"):
        Engine(require_model(NAME), kv_plan_options={"lowest_tier": "INT4"})


@pytest.mark.slow
def test_the_engine_s_options_reach_its_plans():
    """An engine with options passes them to every plan it makes: a
    controlled plan at 80% with INT4 the lowest tier is short, names no
    INT2, and records the options."""
    options = PlanOptions(lowest_tier="INT4", protect_first_page=True)
    engine = Engine(require_model(NAME), kv_adaptive=True, kv_scoring=True,
                    kv_plan_options=options, prefill_chunk=None)
    engine.load_weights()
    ids = np.random.default_rng(1441).integers(1000, 100_000, 1024).astype(np.int32)
    with engine.conversation() as chat:
        chat.turn(ids, 8, stop_at_eos=False)
        record = chat.plan(0.8, source="semantic")
    assert record.plan.options == options
    assert record.plan.short and record.controlled["target_met"] is False
    assert INT2 not in record.plan.target.tolist() and 0 not in record.plan.pages.tolist()


@pytest.mark.slowest
def test_the_engine_s_upgrade_plans_record_its_options():
    """RED for a few decode steps, then ample: the engine's upgrade plan,
    made at GREEN after the cooldown, records the options it runs with."""
    options = PlanOptions(protect_first_page=True)
    engine = Engine(require_model(NAME), kv_adaptive=True, prefill_chunk=None,
                    kv_upgrade_cooldown_seconds=COOLDOWN, kv_plan_options=options)
    engine.load_weights()
    ids = np.random.default_rng(1442).integers(1000, 100_000, 2048).astype(np.int32)

    def headroom(cache):
        length = cache.length if cache is not None else 0
        if len(ids) + 4 < length <= len(ids) + 8:
            return 100 * MIB
        return (AMPLE_MIB if length > len(ids) else 2000) * MIB

    run(engine, ids, 160, headroom)
    upgrades = [p for p in engine.plans if p.upgrades]
    assert upgrades and all(p.plan.options == options for p in upgrades)
