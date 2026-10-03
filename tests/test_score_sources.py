"""Three sources of scores for the precision controller (#104).

The controller plans from one of three: the scorer's semantic scores
(#99); uniform scores, RQ3's main baseline, every page alike, a partial
step spread evenly across positions; and random scores from a seed, a
secondary control. They plug into the same plan(), so each meets the same
byte target.
"""

import numpy as np
import pytest
from test_controller import PAGE_BYTES, ERRORS, SECONDS, THRESHOLDS, P, cache, yellow_short_by

from microinfer import controller, score_sources
from microinfer.score_sources import scores_for, uniform_scores


def plan_from(scores, tiers, need=300_000, positions=None):
    """A plan short of `need` bytes; unless `positions` says otherwise, the
    cache holds a recency floor beyond every page given."""
    if positions is None:
        positions = (max(page for _, page in tiers) + 1) * P + controller.DEFAULT_RECENCY_FLOOR
    return controller.plan(yellow_short_by(need), THRESHOLDS, scores, tiers, PAGE_BYTES,
                           ERRORS, move_seconds=SECONDS, positions=positions, page_tokens=P,
                           margin_bytes=0)


def semantic_download(layers, max_pages, rng):
    """What ImportanceScores.download() returns: (layers, max_pages), NaN
    for pages never observed."""
    scores = rng.random((layers, max_pages)).astype(np.float32)
    scores[:, max_pages // 2:] = np.nan
    return scores


def test_the_three_sources_plan_to_the_same_bytes():
    """Each source's plan has the same byte target and meets it, overshooting
    by less than one downgrade: they reclaim the same bytes."""
    tiers = cache(4, 60)
    rng = np.random.default_rng(104)
    download = semantic_download(4, 128, rng)
    plans = {source: plan_from(scores_for(source, tiers, semantic=download, seed=7), tiers)
             for source in score_sources.SOURCES}
    assert set(plans) == {"semantic", "uniform", "random"}
    largest = max(PAGE_BYTES[a] - PAGE_BYTES[b] for a, b in zip(controller.TIERS,
                                                               controller.TIERS[1:]))
    for source, p in plans.items():
        assert p.target_bytes == 300_000 and not p.short, source
        assert 0 <= p.reclaimed_bytes - p.target_bytes < largest, source


def test_semantic_scores_are_the_scorers():
    """The scorer's download, read at each page: NaN stays NaN, for a page
    never observed. An open page is never observed, and is NaN too, not the
    download's last column; a page past the download is refused."""
    download = semantic_download(2, 16, np.random.default_rng(1))
    tiers = cache(2, 12) | {(layer, -1): "FP16" for layer in range(2)}
    got = scores_for("semantic", tiers, semantic=download)
    for (layer, page), score in got.items():
        want = download[layer, page] if page >= 0 else np.nan
        assert (np.isnan(score) and np.isnan(want)) or score == float(want)
    with pytest.raises(ValueError, match="outside"):
        scores_for("semantic", cache(3, 2), semantic=download)


def discrepancy(taken: list[int], positions: int) -> float:
    """The most a window of positions holds more or fewer taken pages than
    its even share: over every window [a, b) of [0, positions)."""
    share = len(taken) / positions
    held = np.concatenate([[0], np.cumsum(np.isin(np.arange(positions), taken))])
    a, b = np.triu_indices(positions + 1)
    return float(np.abs(held[b] - held[a] - (b - a) * share).max())


def evenness_bound(positions: int) -> float:
    """What bit-reversed order keeps every window within: measured at most
    3.9 pages for every partial step of every count of positions below
    300, growing as log2(positions) / 2. Taken oldest first, a window holds
    the whole step."""
    return np.log2(positions) / 2 + 1


def test_bit_reversed_order_is_even_for_any_step():
    """Over every count of positions to 130 and every partial step of it,
    no window of positions is off its even share by more than the bound;
    and where both are powers of two, the pages taken are evenly spaced,
    every k-th exactly."""
    for positions in range(2, 131):
        scores = uniform_scores([(0, p) for p in range(positions)])
        order = sorted(range(positions), key=lambda p: scores[(0, p)])
        for k in range(1, positions + 1):
            assert discrepancy(order[:k], positions) <= evenness_bound(positions), (positions, k)
    for positions, k in ((64, 8), (128, 32), (32, 2)):
        scores = uniform_scores([(0, p) for p in range(positions)])
        taken = sorted(sorted(range(positions), key=lambda p: scores[(0, p)])[:k])
        assert set(np.diff(taken)) == {positions // k} and taken[0] == 0


def test_uniform_spreads_where_a_step_adds_no_error():
    """Were two tiers' logged errors equal, a step between them would cost
    every page nothing; the plan breaks ties by score, so uniform still
    spreads the step rather than taking the oldest pages first."""
    errors = dict(ERRORS, INT8=0.0)
    tiers = cache(1, 64)
    scores = scores_for("uniform", tiers)
    need = 16 * (PAGE_BYTES["FP16"] - PAGE_BYTES["INT8"])
    positions = 64 * P + controller.DEFAULT_RECENCY_FLOOR
    plan = controller.plan(yellow_short_by(need), THRESHOLDS, scores, tiers, PAGE_BYTES, errors,
                           move_seconds=SECONDS, positions=positions, page_tokens=P,
                           margin_bytes=0)
    taken = sorted(m.page for m in plan.downgrades)
    assert all(m.cost == 0 for m in plan.downgrades)
    assert set(np.diff(taken)) == {4}, taken


def test_uniform_spreads_a_partial_step_evenly_across_positions():
    """Uniform scores are alike, so the plan goes breadth first, as equal
    scores do (#101); and when a step is partial, the pages it takes are
    spread evenly across positions, not the oldest first: no window of
    positions is off its even share by more than the bound. Every layer of a
    position goes together. With the recency floor taking the newest pages
    out, and on a later step, INT8 to INT4, too."""
    for pages, fraction, floor_pages, tier in ((64, 0.25, 0, "FP16"), (60, 0.3, 0, "FP16"),
                                               (37, 0.5, 0, "FP16"), (100, 0.1, 0, "FP16"),
                                               (100, 0.1, 4, "FP16"), (64, 0.3, 4, "INT8")):
        tiers = cache(3, pages, tier=tier)
        scores = scores_for("uniform", tiers)
        values = list(scores.values())
        assert max(values) - min(values) < 1e-3 * max(values)  # alike, to ranking only
        lower = controller.TIERS[controller.TIERS.index(tier) + 1]
        candidates = pages - floor_pages
        need = int(fraction * candidates * 3) * (PAGE_BYTES[tier] - PAGE_BYTES[lower])
        plan = plan_from(scores, tiers, need, positions=candidates * P
                         + controller.DEFAULT_RECENCY_FLOOR if floor_pages else None)
        assert all(m.target_tier == lower for m in plan.downgrades)
        taken = sorted({m.page for m in plan.downgrades})
        assert taken[-1] < candidates
        by_layer = {}
        for m in plan.downgrades:
            by_layer.setdefault(m.page, set()).add(m.layer)
        last = plan.downgrades[-1].page  # the target may be met partway through its layers
        assert all(len(layers) == 3 for page, layers in by_layer.items() if page != last)
        assert discrepancy(taken, candidates) <= evenness_bound(candidates), (pages, taken)


def test_random_is_the_same_for_the_same_seed():
    """A seed's scores, and so its plan, are the same every time; another
    seed's differ. A page's score depends on nothing but the seed and the
    page: as the cache grows, the pages it held keep theirs, so one plan's
    scores are the next's."""
    tiers = cache(3, 40)
    a, b = scores_for("random", tiers, seed=11), scores_for("random", tiers, seed=11)
    assert a == b
    assert plan_from(a, tiers).downgrades == plan_from(b, tiers).downgrades
    assert scores_for("random", tiers, seed=12) != a
    assert scores_for("random", tiers, seed=11 + 2**40) != a  # every bit of the seed counts
    assert all(0 <= s < 1 for s in a.values())
    grown = scores_for("random", cache(4, 45), seed=11)
    assert all(grown[key] == score for key, score in a.items())
    spread = np.array(list(scores_for("random", cache(8, 500), seed=3).values()))
    assert abs(spread.mean() - 0.5) < 0.02 and len(set(spread)) == len(spread)


def test_what_the_sources_refuse():
    """A source that is not one, and a source without what it reads."""
    tiers = cache(1, 4)
    with pytest.raises(ValueError, match="source"):
        scores_for("entropy", tiers)
    with pytest.raises(ValueError, match="seed"):
        scores_for("random", tiers)
    with pytest.raises(ValueError, match="semantic"):
        scores_for("semantic", tiers)
