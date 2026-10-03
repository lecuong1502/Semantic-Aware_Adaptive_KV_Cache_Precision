"""The precision controller's plan: a byte target, met greedily by marginal
cost (#101). Pure logic, no device.

Given the headroom, ADR-0013's thresholds, each page's importance score and
current tier, the controller reclaims the bytes that bring headroom back
above YELLOW with a margin, taking again and again the (page, one tier down)
with the smallest score x added error / bytes saved. A tier's error is what
Milestone 0's roundtrips logged. At YELLOW the plan is split into batches
within a per-step time budget; at RED it is one batch.
"""

import time

import numpy as np
import pytest

from microinfer import controller, monitor
from microinfer.controller import Plan, plan
from microinfer.footprint import MIB

#: A page's bytes at each tier, and each tier's error, of Qwen2.5-0.5B's
#: shape and roundtrips, rounded: what the greedy decides by.
PAGE_BYTES = {"FP16": 16384, "INT8": 8960, "INT4": 4864, "INT2": 2816}
ERRORS = {"FP16": 0.0, "INT8": 2.2e-5, "INT4": 6.4e-3, "INT2": 0.161}
SECONDS = {(a, b): 1e-4 for a in PAGE_BYTES for b in PAGE_BYTES if a != b}
THRESHOLDS = monitor.Thresholds(red_below_bytes=512 * MIB, yellow_below_bytes=1024 * MIB,
                                persist_polls=3)


def cache(layers, pages, tier="FP16"):
    return {(layer, page): tier for layer in range(layers) for page in range(pages)}


#: Positions per page, as the default build's P.
P = 32


def make_plan(headroom, scores, tiers, margin=0, budget=0.020, positions=None, floor=None):
    """A plan over `tiers`. Unless `positions` says otherwise, the cache
    holds a recency floor's worth beyond every page given, so that no page
    is within it."""
    floor = controller.DEFAULT_RECENCY_FLOOR if floor is None else floor
    if positions is None:
        positions = (max(page for _, page in tiers) + 1) * P + floor
    return plan(headroom, THRESHOLDS, scores, tiers, PAGE_BYTES, ERRORS,
                margin_bytes=margin, move_seconds=SECONDS, budget_seconds=budget,
                positions=positions, page_tokens=P, recency_floor=floor)


def yellow_short_by(nbytes, margin=0):
    """A headroom that needs `nbytes` reclaimed to be back at YELLOW's line."""
    return THRESHOLDS.yellow_below_bytes + margin - nbytes


def test_the_plan_meets_the_target_and_overshoots_by_less_than_one_move():
    """However the scores fall, the moves reclaim at least the target, and
    take away the last move and they would not: no move beyond the one
    that meets it. With a margin, the target is the margin above YELLOW.
    At GREEN, even within the margin, there is nothing to reclaim: a plan
    answers YELLOW or RED."""
    rng = np.random.default_rng(101)
    tiers = cache(4, 400)  # more than any target here can take
    for trial in range(20):
        scores = {key: float(s) for key, s in zip(tiers, rng.random(len(tiers)))}
        margin = int(rng.integers(0, 3)) * MIB
        need = margin + int(rng.integers(1, 600_000))  # below YELLOW's line
        p = make_plan(yellow_short_by(need, margin), scores, tiers, margin=margin)
        assert p.target_bytes == need and not p.short
        assert p.reclaimed_bytes >= need, trial
        assert p.reclaimed_bytes - p.downgrades[-1].bytes_saved < need, trial
        assert p.reclaimed_bytes == sum(m.bytes_saved for m in p.downgrades)
    green = make_plan(THRESHOLDS.yellow_below_bytes, {k: 1.0 for k in tiers}, tiers, margin=MIB)
    assert green.level == monitor.GREEN and green.downgrades == () and green.target_bytes == 0


def test_a_move_is_one_tier_down_from_where_the_page_is():
    """Each move takes its page one tier down from where the plan left it,
    from the tier the cache holds it at; INT2 goes no lower."""
    tiers = {(0, 0): "INT8", (0, 1): "INT2", (0, 2): "FP16"}
    p = make_plan(yellow_short_by(10**9), {k: 1.0 for k in tiers}, tiers)
    at = dict(tiers)
    ladder = controller.TIERS
    for m in p.downgrades:
        assert m.current_tier == at[(m.layer, m.page)]
        assert ladder.index(m.target_tier) == ladder.index(m.current_tier) + 1
        assert m.bytes_saved == PAGE_BYTES[m.current_tier] - PAGE_BYTES[m.target_tier]
        at[(m.layer, m.page)] = m.target_tier
    assert all(t == "INT2" for t in at.values()) and p.short


def test_equal_scores_go_breadth_first():
    """With every score equal, every page goes to INT8 before any goes to
    INT4, and every page to INT4 before any to INT2."""
    tiers = cache(3, 10)
    p = make_plan(yellow_short_by(10**9), {k: 0.5 for k in tiers}, tiers)
    order = [m.target_tier for m in p.downgrades]
    assert order == ["INT8"] * 30 + ["INT4"] * 30 + ["INT2"] * 30


def test_one_near_zero_score_goes_deep_first():
    """A page the model hardly attends goes all the way down before any
    other page leaves INT8."""
    tiers = cache(2, 8)
    scores = {k: 0.5 for k in tiers}
    scores[(1, 5)] = 1e-9
    p = make_plan(yellow_short_by(10**9), scores, tiers)
    deep = [i for i, m in enumerate(p.downgrades) if (m.layer, m.page) == (1, 5)]
    others_below_int8 = [i for i, m in enumerate(p.downgrades)
                         if (m.layer, m.page) != (1, 5) and m.target_tier in ("INT4", "INT2")]
    assert [p.downgrades[i].target_tier for i in deep] == ["INT8", "INT4", "INT2"]
    assert deep[-1] < others_below_int8[0]


def test_int2_is_chosen_only_when_its_error_is_outweighed():
    """With equal scores no page goes to INT2 while another can still go to
    INT4. A page goes to INT2 ahead of another's INT4 only when its score
    is so low that score x INT2's added error per byte is the smaller."""
    tiers = cache(1, 4, tier="INT4") | {(1, 0): "INT8"}
    equal = make_plan(yellow_short_by(4000), {k: 0.5 for k in tiers}, tiers)
    assert [m.target_tier for m in equal.downgrades] == ["INT4"]

    per_byte = {t: (ERRORS[t] - ERRORS[s]) / (PAGE_BYTES[s] - PAGE_BYTES[t])
                for s, t in (("INT8", "INT4"), ("INT4", "INT2"))}
    low = 0.5 * per_byte["INT4"] / per_byte["INT2"]  # where the costs cross
    for score, first in ((low * 0.9, "INT2"), (low * 1.1, "INT4")):
        scores = {k: 0.5 for k in tiers}
        scores[(0, 2)] = score
        assert make_plan(yellow_short_by(1), scores, tiers).downgrades[0].target_tier == first


def test_unscored_pages_go_in_position_order_oldest_first():
    """Before any score exists (NaN), every page is taken as equally
    important: the plan goes breadth first, each tier in position order,
    the oldest page first."""
    tiers = cache(2, 5)
    p = make_plan(yellow_short_by(10**9), {k: float("nan") for k in tiers}, tiers)
    assert [(m.page, m.layer, m.target_tier) for m in p.downgrades[:10]] == [
        (page, layer, "INT8") for page in range(5) for layer in range(2)]
    assert [m.target_tier for m in p.downgrades] == ["INT8"] * 10 + ["INT4"] * 10 + ["INT2"] * 10


def test_a_page_not_yet_scored_goes_no_deeper_than_an_average_one():
    """Among scored pages, one not yet scored (NaN), as the newest often
    are, is taken as of their mean score: it does not go to INT2 ahead of
    the scored pages, as it would if it cost nothing."""
    tiers = cache(1, 6)
    scores = {(0, p): s for p, s in enumerate((0.9, 0.5, 0.3, 0.2, 0.1, float("nan")))}
    p = make_plan(yellow_short_by(10**9), scores, tiers)
    unscored = [m.target_tier for m in p.downgrades if m.page == 5]
    first_int4 = next(i for i, m in enumerate(p.downgrades) if m.target_tier == "INT4")
    assert p.downgrades[first_int4].page == 4  # the lowest-scored, not the unscored
    assert unscored == ["INT8", "INT4", "INT2"]


def test_yellow_splits_into_batches_within_the_budget_and_red_is_one():
    """At YELLOW the moves, in order, fall into batches whose estimated
    time is within the budget, each batch as full as it can be; at RED
    the plan is one batch of every move."""
    tiers = cache(4, 50)
    scores = {k: float(v) for k, v in zip(tiers, np.random.default_rng(3).random(len(tiers)))}
    budget = 25 * SECONDS[("FP16", "INT8")]
    y = make_plan(yellow_short_by(1_000_000), scores, tiers, budget=budget)
    assert y.level == monitor.YELLOW and len(y.batches) > 1
    assert [m for b in y.batches for m in b] == list(y.downgrades)
    for b in y.batches:
        assert sum(m.seconds for m in b) <= budget + 1e-12
    for b, after in zip(y.batches, y.batches[1:]):
        assert sum(m.seconds for m in b) + after[0].seconds > budget
    r = make_plan(THRESHOLDS.red_below_bytes - 1, scores, tiers, budget=budget)
    assert r.level == monitor.RED and r.batches == (r.downgrades,)
    long = make_plan(yellow_short_by(20_000), scores, tiers,
                     budget=SECONDS[("FP16", "INT8")] / 2)
    assert all(len(b) == 1 for b in long.batches)  # each over the budget, alone


def test_the_logged_errors_and_latencies():
    """The errors come from Milestone 0's logged roundtrips, and grow down
    the tiers; the latencies from #97's, for every downgrade one tier down."""
    errors = controller.logged_tier_errors("qwen2.5-1.5b-instruct")
    assert errors["FP16"] == 0 < errors["INT8"] < errors["INT4"] < errors["INT2"]
    seconds = controller.logged_move_seconds("qwen2.5-1.5b-instruct")
    assert set(zip(controller.TIERS, controller.TIERS[1:])) <= set(seconds)
    assert all(s > 0 for s in seconds.values())


def test_what_the_controller_refuses():
    """A page without a score, at a tier that is not one, or with a negative
    score."""
    tiers = cache(1, 2)
    with pytest.raises(ValueError, match="score"):
        make_plan(yellow_short_by(1), {(0, 0): 1.0}, tiers)
    with pytest.raises(ValueError, match="tier"):
        make_plan(yellow_short_by(1), {k: 1.0 for k in tiers}, {(0, 0): "INT3", (0, 1): "FP16"})
    with pytest.raises(ValueError, match="score"):
        make_plan(yellow_short_by(1), {(0, 0): -1.0, (0, 1): 1.0}, tiers)
    assert isinstance(make_plan(yellow_short_by(1), {k: 1.0 for k in tiers}, tiers), Plan)


# -- the recency floor (#102) -----------------------------------------------------


def test_no_plan_names_a_page_within_the_recency_floor():
    """The last W positions are never downgraded, nor the open pages: a
    page any of whose positions is among the last W, the partly filled last
    page with them, at any score and however short of memory, is in no
    plan, at YELLOW or at RED, nor is an open page, (layer, -1) or (layer,
    -2), given among the pages. A page whose last position is just before
    the floor may be."""
    rng = np.random.default_rng(102)
    for trial in range(30):
        floor = int(rng.choice([0, 1, 31, 32, 33, 64, 128, 200]))
        positions = int(rng.integers(1, 40 * P))
        tiers = cache(3, -(-positions // P))  # every page of positions held
        tiers |= {(layer, open_page): "FP16" for layer in range(3) for open_page in (-1, -2)}
        scores = {k: float(v) for k, v in zip(tiers, rng.random(len(tiers)))}
        for headroom in (yellow_short_by(10**9), THRESHOLDS.red_below_bytes - 1):
            p = make_plan(headroom, scores, tiers, positions=positions, floor=floor)
            assert p.recency_floor == floor
            named = {m.page for m in p.downgrades}
            for page in named:
                assert (page + 1) * P <= positions - floor, (trial, page, positions, floor)
            # Everything outside the floor may go, and does, so short of memory.
            free = {page for page in range(-(-positions // P))
                    if (page + 1) * P <= positions - floor}
            assert named == free, (trial, positions, floor)


def test_the_floor_is_a_parameter_recorded_in_every_plan():
    """W is 128 by default, any non-negative count of positions otherwise,
    and every plan says which, even one with nothing to do."""
    tiers = cache(1, 20)
    scores = {k: 1.0 for k in tiers}
    assert controller.DEFAULT_RECENCY_FLOOR == 128
    assert make_plan(yellow_short_by(1), scores, tiers).recency_floor == 128
    green = make_plan(THRESHOLDS.yellow_below_bytes, scores, tiers, floor=64)
    assert green.recency_floor == 64 and green.downgrades == ()
    held = make_plan(yellow_short_by(10**9), scores, tiers, positions=20 * P, floor=5 * P)
    assert {m.page for m in held.downgrades} == set(range(15))
    partly = make_plan(yellow_short_by(10**9), scores, tiers, positions=19 * P + 7, floor=0)
    assert {m.page for m in partly.downgrades} == set(range(19))  # page 19 is being filled
    with pytest.raises(ValueError, match="floor"):
        make_plan(yellow_short_by(1), scores, tiers, floor=-1)


# -- the upgrade policy (#103) ----------------------------------------------------

#: Headroom at which upgrades may begin: GREEN's threshold, T_high, and a
#: P90 spike above it.
UPGRADE_LINE = THRESHOLDS.yellow_below_bytes + controller.DEFAULT_UPGRADE_SPIKE_BYTES


def make_upgrades(headroom, scores, tiers, now=100.0, last_downgrade=0.0, budget=0.020,
                  shadowed=None):
    """An upgrade plan over `tiers`, every page holding a shadow unless
    `shadowed` says which do."""
    return controller.upgrade_plan(headroom, THRESHOLDS, scores, tiers, PAGE_BYTES, ERRORS,
                                   move_seconds=SECONDS, now_seconds=now,
                                   last_downgrade_seconds=last_downgrade, budget_seconds=budget,
                                   shadowed=set(tiers) if shadowed is None else shadowed)


def test_no_upgrade_within_five_seconds_of_a_downgrade():
    """An upgrade waits 5 s after the last downgrade: none at 4.99 s, some
    at 5; and with no downgrade yet, nothing holds it back."""
    tiers = cache(1, 4, tier="INT4")
    scores = {k: 1.0 for k in tiers}
    roomy = UPGRADE_LINE + 10 * MIB
    assert controller.DEFAULT_UPGRADE_COOLDOWN_SECONDS == 5.0
    early = make_upgrades(roomy, scores, tiers, now=104.99, last_downgrade=100.0)
    assert early.upgrades == () and "cooldown" in early.held_by
    assert make_upgrades(roomy, scores, tiers, now=105.0, last_downgrade=100.0).upgrades
    assert make_upgrades(roomy, scores, tiers, last_downgrade=None).upgrades


def test_no_upgrade_leaves_headroom_below_the_line():
    """Only at GREEN, and never past T_high + 512 MiB: the upgrades take at
    most the headroom above that line, at any headroom; below it, or at
    YELLOW or RED, there are none."""
    rng = np.random.default_rng(103)
    tiers = {key: str(rng.choice(["INT8", "INT4", "INT2"])) for key in cache(3, 30)}
    for trial in range(30):
        scores = {k: float(v) for k, v in zip(tiers, rng.random(len(tiers)))}
        headroom = UPGRADE_LINE + int(rng.integers(0, 400_000))
        u = make_upgrades(headroom, scores, tiers)
        assert headroom - u.taken_bytes >= UPGRADE_LINE, trial
        assert u.taken_bytes == sum(m.bytes_taken for m in u.upgrades)
    for headroom, reason in ((UPGRADE_LINE - 1, "headroom"),
                             (THRESHOLDS.yellow_below_bytes - 1, "GREEN"),
                             (THRESHOLDS.red_below_bytes - 1, "GREEN")):
        held = make_upgrades(headroom, {k: 1.0 for k in tiers}, tiers)
        assert held.upgrades == () and reason in held.held_by, headroom


def test_upgrades_come_in_reverse_marginal_cost_order():
    """Each upgrade takes its page one tier up, and they come largest score
    x error removed / bytes taken first. With room for everything, the
    upgrades of what a downgrade plan did are that plan, backwards."""
    rng = np.random.default_rng(31)
    tiers = cache(3, 12)
    scores = {k: float(v) for k, v in zip(tiers, rng.random(len(tiers)))}
    down = make_plan(yellow_short_by(400_000), scores, tiers)
    after = dict(tiers)
    for m in down.downgrades:
        after[(m.layer, m.page)] = m.target_tier
    up = make_upgrades(UPGRADE_LINE + 10**9, scores, after)
    ladder = controller.TIERS
    for m in up.upgrades:
        assert ladder.index(m.target_tier) == ladder.index(m.current_tier) - 1
    gains = [m.gain for m in up.upgrades]
    assert gains == sorted(gains, reverse=True)
    assert [(m.layer, m.page, m.target_tier, m.current_tier) for m in up.upgrades] == [
        (m.layer, m.page, m.current_tier, m.target_tier) for m in reversed(down.downgrades)]


def test_upgrades_fall_into_batches_within_the_budget():
    """Upgrades are made at GREEN, in batches within the per-step budget,
    as a YELLOW plan's downgrades are."""
    tiers = cache(2, 40, tier="INT2")
    budget = 7 * SECONDS[("INT2", "INT4")]
    u = make_upgrades(UPGRADE_LINE + 10**9, {k: 1.0 for k in tiers}, tiers, budget=budget)
    assert len(u.batches) > 1 and [m for b in u.batches for m in b] == list(u.upgrades)
    assert all(sum(m.seconds for m in b) <= budget + 1e-12 for b in u.batches)


def test_a_page_without_a_shadow_is_never_upgraded():
    """A page born at a quantised tier never had FP16 bytes, and has no
    shadow to be restored from (#96): only the pages the caller names as
    shadowed are upgraded, whatever their scores and however much room."""
    tiers = {(0, 0): "INT4", (0, 1): "INT2", (0, 2): "INT8", (1, 0): "INT2"}
    scores = {k: 1.0 for k in tiers}
    scores[(0, 1)] = 1e6  # the most worth restoring, and born at INT2
    u = make_upgrades(UPGRADE_LINE + 10**9, scores, tiers, shadowed={(0, 0), (1, 0)})
    assert {(m.layer, m.page) for m in u.upgrades} == {(0, 0), (1, 0)}
    assert all(m.target_tier == "FP16" for m in u.upgrades if m.current_tier == "INT8")


def test_the_clock_is_monotonic_and_one():
    """`now_seconds` is time.monotonic() unless given, the clock the last
    downgrade's time must come from too: a last downgrade later than now is
    a clock mixed up, and refused rather than read as a cooldown."""
    tiers = cache(1, 2, tier="INT4")
    scores = {k: 1.0 for k in tiers}
    roomy = UPGRADE_LINE + 10 * MIB
    recent = controller.upgrade_plan(roomy, THRESHOLDS, scores, tiers, PAGE_BYTES, ERRORS,
                                     move_seconds=SECONDS, shadowed=set(tiers),
                                     last_downgrade_seconds=time.monotonic())
    assert recent.upgrades == () and "cooldown" in recent.held_by
    with pytest.raises(ValueError, match="clock"):
        make_upgrades(roomy, scores, tiers, now=100.0, last_downgrade=1.7e9)  # wall time
