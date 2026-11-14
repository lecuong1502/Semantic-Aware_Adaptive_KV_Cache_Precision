"""How the engine makes a plan's moves into batches, and paces them (#205).

next_batch is a pure function of a plan and the cache's tiers, so it is
tested on hand-made plans. Engine._apply is tested over a cache that only
records what it is asked to move, so that a batch that cannot allocate can
be made to happen on demand.
"""

import time
from types import SimpleNamespace

import numpy as np
from conftest import require_model

from microinfer import Engine, _microinfer, engine
from microinfer.engine import MoveBatch, move_seconds, next_batch
from microinfer.monitor import GREEN, RED, YELLOW

Tier = _microinfer.Tier
FP16, INT8, INT4, INT2 = range(4)
#: A page's bytes at each tier, small round numbers.
BYTES = [100, 60, 40, 30]
NAME = "qwen2.5-0.5b-instruct"


class Plan:
    """A plan of (layer, page, current, target) moves, as the engine reads one."""

    def __init__(self, moves, level=YELLOW):
        self.layers, self.pages, self.current, self.target = (np.array(c) for c in zip(*moves))
        self.level = level

    def __len__(self):
        return len(self.layers)


def held(tiers=None):
    """Two layers of four pages at FP16, but for `tiers`, {(layer, page): tier}."""
    out = np.zeros((2, 4), dtype=np.int64)
    for key, tier in (tiers or {}).items():
        out[key] = tier
    return out


NO_SHADOWS = np.zeros((2, 4), dtype=bool)


def batch(plan, first=0, tiers=None, shadowed=NO_SHADOWS, bytes_cap=np.inf,
          seconds_cap=np.inf, per=1e-3) -> MoveBatch:
    return next_batch(plan, first, held(tiers), shadowed, BYTES, bytes_cap, seconds_cap, per)


def test_a_page_s_chain_of_moves_is_one_move_to_where_it_ends():
    """FP16 to INT8 to INT4 to INT2 for one page and FP16 to INT8 for another,
    in the plan's order: one move each, to INT2 and to INT8, the bytes those
    pages take there."""
    plan = Plan([(0, 1, FP16, INT8), (0, 2, FP16, INT8), (0, 1, INT8, INT4),
                 (0, 1, INT4, INT2)])
    b = batch(plan)
    assert b.moves == [(0, 1, Tier.INT2), (0, 2, Tier.INT8)]
    assert (b.end, b.skipped, b.bytes) == (4, 0, BYTES[INT2] + BYTES[INT8])


def test_a_move_the_cache_or_the_batch_does_not_expect_is_skipped():
    """A page the cache holds lower than the plan expects is skipped, and so
    is the rest of its chain; another page's moves go on."""
    plan = Plan([(1, 0, FP16, INT8), (1, 0, INT8, INT4), (1, 3, FP16, INT4)])
    b = batch(plan, tiers={(1, 0): INT2})
    assert b.moves == [(1, 3, Tier.INT4)]
    assert (b.end, b.skipped) == (3, 2)


def test_a_batch_fits_its_bytes_and_its_time_and_takes_one_move_at_least():
    plan = Plan([(0, p, FP16, INT4) for p in range(4)])
    assert len(batch(plan, bytes_cap=2 * BYTES[INT4]).moves) == 2
    assert len(batch(plan, bytes_cap=0).moves) == 1
    # Three moves of 1 ms fit 3.05 ms; each that pins a shadow costs more,
    # and two of those fit.
    cap = 3e-3 + engine.SHADOW_PIN_SECONDS * 2.5
    assert len(batch(plan, seconds_cap=cap, shadowed=np.ones((2, 4), bool)).moves) == 3
    assert len(batch(plan, seconds_cap=cap).moves) == 2
    assert len(batch(plan, seconds_cap=0.0).moves) == 1
    second = batch(plan, first=2, bytes_cap=BYTES[INT4])
    assert second.moves == [(0, 2, Tier.INT4)] and second.end == 3


def test_a_batch_s_time_a_move_is_measured_past_its_fixed_part_when_large_enough():
    small = {"moves": 4, "shadows_taken": 4, "seconds": 0.004}
    assert move_seconds(small) == 0.001
    n = engine.PACED_BATCH_MOVES
    large = {"moves": n, "shadows_taken": n,
             "seconds": engine.BATCH_FIXED_SECONDS + n * (engine.SHADOW_PIN_SECONDS + 1e-5)}
    assert abs(move_seconds(large) - 1e-5) < 1e-12
    assert move_seconds({"moves": n, "shadows_taken": 0, "seconds": 0.0}) == 0.0


class Pages:
    """A cache's pages that only record what they are asked to move, and
    refuse to allocate a batch of more than `fits` moves."""

    def __init__(self, fits):
        self.tiers, self.fits, self.batches = held(), fits, []

    def page_tiers(self):
        return self.tiers.copy()

    def shadowed(self):
        return self.tiers > FP16

    def move_pages(self, moves):
        if len(moves) > self.fits:
            raise _microinfer.OutOfMemory(f"{len(moves)} new pages do not fit")
        for layer, page, tier in moves:
            self.tiers[layer, page] = int(tier)
        self.batches.append(len(moves))
        self.last_batch = {"moves": len(moves), "shadows_taken": len(moves), "seconds": 1e-4}


def applying(plan, fits, upgrades=False):
    """An engine applying `plan` at once over Pages(fits); the pages, and
    the plan's record."""
    e = Engine(require_model(NAME))
    pages = Pages(fits)
    cache = SimpleNamespace(pages=pages, nbytes=0)
    entry = SimpleNamespace(plan=plan, upgrades=upgrades, skipped=0, batches=[], ended="")
    e._pending, e._cursor = entry, 0
    e._recent_moves.append(1e-5)
    e._apply(cache, time.perf_counter(), True, None)
    return pages, entry, e


def test_a_batch_that_cannot_allocate_is_tried_at_half_its_size():
    """At RED every page goes at once; a batch the allocator refuses is
    tried again at half its bytes until one fits, and the plan is applied."""
    plan = Plan([(layer, p, FP16, INT4) for layer in range(2) for p in range(4)], level=RED)
    pages, entry, e = applying(plan, fits=3)
    assert entry.ended == "applied" and e._pending is None
    assert sum(pages.batches) == 8 and max(pages.batches) <= 3
    assert (pages.tiers == INT4).all()


def test_a_single_move_that_cannot_allocate_stops_the_plan():
    plan = Plan([(0, p, FP16, INT4) for p in range(4)], level=RED)
    pages, entry, e = applying(plan, fits=0)
    assert entry.ended.startswith("stopped:") and e._pending is None
    assert pages.batches == []


def test_an_upgrade_plan_is_applied_at_once_as_a_release_lets_it():
    """An upgrade plan at GREEN is not held to a step's budget (#205): with
    the budget used up before it starts, it still restores every page."""
    plan = Plan([(layer, p, INT4, FP16) for layer in range(2) for p in range(4)],
                level=GREEN)
    e = Engine(require_model(NAME))
    pages = Pages(fits=100)
    pages.tiers[:] = INT4
    cache = SimpleNamespace(pages=pages, nbytes=0)
    entry = SimpleNamespace(plan=plan, upgrades=True, skipped=0, batches=[], ended="")
    e._pending, e._cursor = entry, 0
    e._recent_moves.append(1.0)  # a move would take the whole budget
    e._apply(cache, time.perf_counter() - 10, False, None)
    assert entry.ended == "applied" and (pages.tiers == FP16).all()


def test_a_downgrade_plan_at_yellow_waits_for_the_next_step_once_the_budget_is_spent():
    """A step that made the plan, its making having spent the budget,
    moves nothing; one that did not moves one page at least, so that a plan
    always gets on."""
    plan = Plan([(0, p, FP16, INT4) for p in range(4)], level=YELLOW)
    e = Engine(require_model(NAME))
    for planned, moved in ((True, 0), (False, 1)):
        pages = Pages(fits=100)
        cache = SimpleNamespace(pages=pages, nbytes=0)
        entry = SimpleNamespace(plan=plan, upgrades=False, skipped=0, batches=[], ended="")
        e._pending, e._cursor = entry, 0
        e._recent_moves.clear()
        e._recent_moves.append(1.0)
        e._apply(cache, time.perf_counter() - 10, planned, None)
        assert len(pages.batches) == moved and e._pending is entry
        assert int((pages.tiers == INT4).sum()) == moved
