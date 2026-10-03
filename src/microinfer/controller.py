"""The precision controller's plan: a byte target met greedily by marginal
cost (#101, #88). Pure logic: no device code, nothing moved.

Given the headroom and ADR-0013's thresholds, the plan reclaims the bytes
that bring headroom back above YELLOW with a margin: the target is
yellow_below + margin - headroom, and nothing at GREEN. It takes, again and
again, the move with the smallest marginal cost among every page's next
move one tier down,

    score x (error(lower) - error(current)) / (bytes(current) - bytes(lower)),

until the moves reclaim the target; the last move is the one that meets
it, so the plan overshoots by less than one move. A page whose score is
equal to the others' goes one tier at a time, breadth first, since each
tier down costs more error per byte than the last; one the model hardly
attends costs almost nothing at every tier, and goes deep first. INT2's
error puts it last unless a score outweighs it.

A page with no score yet (NaN) is given a neutral one: the mean of the
scored candidates', or, before any has a score, the same for every page.
Pages within the floor are no candidates, and do not set it.
Before any score exists the plan therefore goes breadth first in position
order, the oldest page first (#88), and a page not yet scored, usually one
of the newest, goes no deeper than a page of average importance. Ties in
cost go to the lower score, then the oldest page, then the lowest layer:
so that scores order pages even where a step adds no error, as one
between two tiers whose logged errors were equal would.

The recency floor (#102): the open pages and the last W positions, W = 128
by default, are never downgraded, under every policy: every policy's plan
is made here. A page any of whose positions is among the last W, the
partly filled last page with them, is no candidate, whatever its score
and however short of memory the cache is, nor is an open page, (layer,
-1) or (layer, -2), whatever `tiers` holds. Every plan records W.

Upgrades (#103) are planned at GREEN only, at least 5 s after the last
downgrade, and never past a headroom of GREEN's threshold, T_high, plus
512 MiB, one P90 spike: in the reverse of a downgrade plan's order, the
largest score x error removed / bytes taken first, in batches within the
same budget. With room for every one, and the scores a plan was made from
unchanged and all present, they undo it backwards: each tier down costs
more error per byte than the last, so a plan's downgrades come in rising
cost, and the upgrades' gains are those costs.

A tier's error is its relative mean squared error on the keys and values
the model caches, keys and values weighed alike, from Milestone 0's logged
roundtrips (tools/quant_roundtrip.py); FP16's is 0. Values dominate it
below INT8, being the harder to quantise. At YELLOW the plan is split into
batches of downgrades whose estimated time, from #97's logged latencies,
is within a per-step budget, so that decoding goes on between them; at RED
it is one batch, reclaimed at once. The engine holds each step to the
budget by the time it measures, the batches being the plan's estimate. The budget is about 20 ms (#88): at
the 0.02-0.04 ms a downgrade took on Qwen2.5-1.5B, some 500 a batch.
"""

from __future__ import annotations

import functools
import heapq
import math
import time
from collections.abc import Collection
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import benchlog
from .footprint import MIB
from .monitor import GREEN, RED, Level, Thresholds

#: The tiers, from FP16 down: a downgrade takes a page one step along it.
TIERS = ("FP16", "INT8", "INT4", "INT2")
#: The headroom kept above YELLOW's threshold once a plan is applied.
DEFAULT_MARGIN_BYTES = 64 * MIB
#: The time one batch of moves may take between two decode steps.
DEFAULT_BUDGET_SECONDS = 0.020
#: The most recent positions no plan downgrades, the open page with them.
DEFAULT_RECENCY_FLOOR = 128
#: How long after the last downgrade an upgrade must wait (#88, #103).
DEFAULT_UPGRADE_COOLDOWN_SECONDS = 5.0
#: The headroom kept above GREEN's threshold, T_high, once upgrades are
#: made: one P90 spike of everyday applications, ADR-0013's 506 MiB rounded
#: up (#88).
DEFAULT_UPGRADE_SPIKE_BYTES = 512 * MIB


@dataclass(frozen=True)
class Downgrade:
    """One page, one tier down: an entry of a requantisation plan, with
    what it saves, its marginal cost and its estimated time."""

    layer: int
    page: int
    current_tier: str
    target_tier: str
    bytes_saved: int
    cost: float
    score: float
    seconds: float


@dataclass(frozen=True, eq=False)
class Plan:
    """The downgrades, in the order they were chosen, and the batches they
    are applied in. `short` if every page outside the recency floor went as
    low as it can and the byte target was still not met. `recency_floor`
    is the W the plan kept.

    The downgrades are held as arrays, one entry each, tiers as indices
    into TIERS, so that a plan of thousands costs a step no objects (#105);
    `downgrades`, `batches` and `batch_bounds`, each batch's [start, end) in
    them, are made on first reading. The batches are the plan's estimate:
    the engine holds each step to the budget by the time it measures."""

    level: Level
    target_bytes: int
    short: bool
    recency_floor: int
    layers: np.ndarray
    pages: np.ndarray
    current: np.ndarray
    lower: np.ndarray
    saved: np.ndarray
    costs: np.ndarray
    scores: np.ndarray
    seconds: np.ndarray
    budget_seconds: float

    @functools.cached_property
    def batch_bounds(self) -> tuple[tuple[int, int], ...]:
        return batch_bounds(self.seconds, self.level, self.budget_seconds)

    @property
    def reclaimed_bytes(self) -> int:
        return int(self.saved.sum())

    def __len__(self) -> int:
        return len(self.layers)

    @functools.cached_property
    def downgrades(self) -> tuple[Downgrade, ...]:
        return tuple(self.downgrade(i) for i in range(len(self)))

    @functools.cached_property
    def batches(self) -> tuple[tuple[Downgrade, ...], ...]:
        return tuple(self.downgrades[a:b] for a, b in self.batch_bounds)

    def downgrade(self, i: int) -> Downgrade:
        return Downgrade(int(self.layers[i]), int(self.pages[i]), TIERS[self.current[i]],
                         TIERS[self.lower[i]], int(self.saved[i]), float(self.costs[i]),
                         float(self.scores[i]), float(self.seconds[i]))


def _empty_plan(level: Level, recency_floor: int, budget_seconds: float) -> Plan:
    none, nothing = np.zeros(0, dtype=np.int64), np.zeros(0)
    return Plan(level=level, target_bytes=0, short=False, recency_floor=recency_floor,
                layers=none, pages=none, current=none, lower=none, saved=none, costs=nothing,
                scores=nothing, seconds=nothing, budget_seconds=budget_seconds)


def plan(headroom_bytes: int, thresholds: Thresholds, scores: dict[tuple[int, int], float],
         tiers: dict[tuple[int, int], str], page_bytes: dict[str, int],
         errors: dict[str, float], *, move_seconds: dict[tuple[str, str], float],
         positions: int, page_tokens: int,
         recency_floor: int = DEFAULT_RECENCY_FLOOR,
         margin_bytes: int = DEFAULT_MARGIN_BYTES,
         budget_seconds: float = DEFAULT_BUDGET_SECONDS) -> Plan:
    """The plan for this headroom: `scores` and `tiers` by (layer, page),
    for the cache's pages; `page_bytes` and `errors` by tier; `move_seconds`
    by (current, target) tier. The cache holds `positions` positions, pages
    of `page_tokens` each; no page with a position among the last
    `recency_floor` is downgraded. plan_arrays, on arrays."""
    _check_pages(scores, tiers)
    keys = [key for key in tiers if key[1] >= 0]  # an open page is never a candidate
    layers = max((layer for layer, _ in keys), default=-1) + 1
    pages = max((page for _, page in keys), default=-1) + 1
    tier_index = np.full((layers, pages), -1, dtype=np.int8)
    score_array = np.full((layers, pages), np.nan)
    for layer, page in keys:
        tier_index[layer, page] = TIERS.index(tiers[(layer, page)])
        score_array[layer, page] = scores[(layer, page)]
    return plan_arrays(headroom_bytes, thresholds, score_array, tier_index, page_bytes, errors,
                       move_seconds=move_seconds, positions=positions, page_tokens=page_tokens,
                       recency_floor=recency_floor, margin_bytes=margin_bytes,
                       budget_seconds=budget_seconds)


def plan_arrays(headroom_bytes: int, thresholds: Thresholds, scores: np.ndarray,
                tiers: np.ndarray, page_bytes: dict[str, int], errors: dict[str, float], *,
                move_seconds: dict[tuple[str, str], float], positions: int, page_tokens: int,
                recency_floor: int = DEFAULT_RECENCY_FLOOR,
                margin_bytes: int = DEFAULT_MARGIN_BYTES,
                budget_seconds: float = DEFAULT_BUDGET_SECONDS) -> Plan:
    """plan(), on arrays: `scores` and `tiers`, (layers, pages), page i of
    layer l at [l, i], a tier as its index into TIERS, -1 for a page the
    cache does not hold, NaN for a score not yet given.

    It computes the greedy by sorting: each page's downgrades cost more,
    tier after tier, so taking the cheapest of every page's next downgrade
    again and again takes every downgrade in order of cost, ties to the
    lower score, then the oldest page, then the layer, then the tier. It is
    exactly the greedy while costs rise tier after tier, as the logged
    errors make them. Were they logged out of order, a page's downgrade is
    taken no sooner than the one before it, at that one's cost, which the
    greedy would order slightly differently among pages tied there. Only as
    many of the cheapest as can meet the target are sorted, so a YELLOW
    plan for a whole 32K window costs a step a few milliseconds (#105)."""
    if recency_floor < 0:
        raise ValueError(f"the recency floor is a count of positions, >= 0; got {recency_floor}")
    if page_tokens <= 0 or positions < 0:
        raise ValueError(f"a cache holds positions >= 0 on pages of > 0; got {positions} "
                         f"positions on pages of {page_tokens}")
    if np.any(scores[tiers >= 0] < 0):
        raise ValueError("a score is a mass, >= 0")

    level = thresholds.classify(headroom_bytes)
    target_bytes = max(0, thresholds.yellow_below_bytes + margin_bytes - headroom_bytes)
    if level == GREEN or target_bytes == 0:
        return _empty_plan(level, recency_floor, budget_seconds)

    # A page is a candidate only if it is a page of positions whose last
    # position comes before the floor's first: no open page is, and the
    # partly filled last page never is.
    layers, pages = tiers.shape
    held = tiers >= 0
    ends = (np.arange(pages) + 1) * page_tokens
    candidate = held & (ends <= positions - recency_floor)[None, :]
    scored = scores[candidate & ~np.isnan(scores)]
    neutral = float(scored.mean()) if scored.size else 1.0
    score = np.where(np.isnan(scores), neutral, scores)

    steps = len(TIERS) - 1  # step k takes a page from tier k to k + 1
    bytes_at = np.array([page_bytes[t] for t in TIERS], dtype=np.int64)
    error_at = np.array([errors[t] for t in TIERS], dtype=np.float64)
    saved = bytes_at[:-1] - bytes_at[1:]
    seconds = np.array([move_seconds[(TIERS[k], TIERS[k + 1])] for k in range(steps)])
    k = np.arange(steps)[None, None, :]
    taken = k >= tiers[:, :, None]
    # score x added error / bytes saved, rounded in that order, as the
    # greedy's own arithmetic is.
    cost = score[:, :, None] * (error_at[1:] - error_at[:-1])[None, None, :] / saved
    cost = np.maximum.accumulate(np.where(taken, cost, -np.inf), axis=2)
    li, pi, ki = np.nonzero(taken & candidate[:, :, None])
    costs, step_scores = cost[li, pi, ki], score[li, pi]

    # Only the cheapest that could meet the target need sorting: no more
    # than target / the smallest saving of them.
    need = min(len(costs), int(-(-target_bytes // int(saved.min()))) + 1)
    if need < len(costs):
        kth = np.partition(costs, need - 1)[need - 1]
        keep = np.nonzero(costs <= kth)[0]
    else:
        keep = np.arange(len(costs))
    order = keep[np.lexsort((ki[keep], li[keep], pi[keep], step_scores[keep], costs[keep]))]
    reclaimed = np.cumsum(saved[ki[order]])
    count = int(np.searchsorted(reclaimed, target_bytes)) + 1
    short = len(order) == 0 or reclaimed[-1] < target_bytes
    order = order[:count]

    steps_taken = ki[order].astype(np.int64)
    return Plan(level=level, target_bytes=target_bytes, short=bool(short),
                recency_floor=recency_floor, layers=li[order].astype(np.int64),
                pages=pi[order].astype(np.int64), current=steps_taken, lower=steps_taken + 1,
                saved=saved[steps_taken], costs=costs[order], scores=step_scores[order],
                seconds=seconds[steps_taken], budget_seconds=budget_seconds)


def batch_bounds(seconds: np.ndarray, level: Level,
                 budget_seconds: float) -> tuple[tuple[int, int], ...]:
    """Each batch's [start, end) in moves of these estimated `seconds`: at
    RED, one of them all; otherwise as full as the budget allows, one longer
    than the budget a batch of its own, and over it."""
    if len(seconds) == 0:
        return ()
    if level == RED:
        return ((0, len(seconds)),)
    bounds, start, spent = [], 0, 0.0
    for i, s in enumerate(seconds.tolist()):
        if i > start and spent + s > budget_seconds:
            bounds.append((start, i))
            start, spent = i, 0.0
        spent += s
    bounds.append((start, len(seconds)))
    return tuple(bounds)


def _check_pages(scores: dict[tuple[int, int], float], tiers: dict[tuple[int, int], str]) -> None:
    for key, tier in tiers.items():
        if tier not in TIERS:
            raise ValueError(f"page {key} is at {tier!r}; a tier is one of {TIERS}")
        if key not in scores:
            raise ValueError(f"page {key} has no score; give NaN for one not yet scored")
        if scores[key] < 0:
            raise ValueError(f"page {key}'s score is {scores[key]}; a score is a mass, >= 0")


def _neutral(scores: dict[tuple[int, int], float], keys: list[tuple[int, int]] | dict) -> float:
    """The score a page not yet scored is taken at: the mean of the scored
    pages among `keys`, or 1 if none is scored, all alike then."""
    scored = [scores[key] for key in keys if not math.isnan(scores[key])]
    return sum(scored) / len(scored) if scored else 1.0


@dataclass(frozen=True)
class Upgrade:
    """One page, one tier up: what it takes, the marginal gain that orders
    it, score x error removed / bytes taken, and its estimated time."""

    layer: int
    page: int
    current_tier: str
    target_tier: str
    bytes_taken: int
    gain: float
    score: float
    seconds: float


@dataclass(frozen=True)
class UpgradePlan:
    """The upgrades, in the order they were chosen, and their batches; or
    none, and `held_by` says what held them."""

    level: Level
    available_bytes: int
    upgrades: tuple[Upgrade, ...]
    batches: tuple[tuple[Upgrade, ...], ...]
    held_by: str

    @property
    def taken_bytes(self) -> int:
        return sum(u.bytes_taken for u in self.upgrades)


def upgrade_plan(headroom_bytes: int, thresholds: Thresholds,
                 scores: dict[tuple[int, int], float], tiers: dict[tuple[int, int], str],
                 page_bytes: dict[str, int], errors: dict[str, float], *,
                 move_seconds: dict[tuple[str, str], float],
                 shadowed: Collection[tuple[int, int]],
                 last_downgrade_seconds: float | None,
                 now_seconds: float | None = None,
                 cooldown_seconds: float = DEFAULT_UPGRADE_COOLDOWN_SECONDS,
                 spike_bytes: int = DEFAULT_UPGRADE_SPIKE_BYTES,
                 budget_seconds: float = DEFAULT_BUDGET_SECONDS) -> UpgradePlan:
    """The upgrades for this headroom (#103), at GREEN only: at least
    `cooldown_seconds` after the last downgrade, and taking no more than the
    headroom above GREEN's threshold, T_high, plus `spike_bytes`, a P90
    spike (#88). `scores` and `tiers` by (layer, page). Only the pages in
    `shadowed`, those holding an FP16 shadow, are upgraded, each from it
    (#96): a page born at a quantised tier never had FP16 bytes, and stays
    where it is. The clock is time.monotonic(), `now_seconds` unless given;
    `last_downgrade_seconds` is from the same clock, or None if no
    downgrade was made. One later than now is a clock mixed up, refused.

    They come in the reverse of a downgrade plan's order: the upgrade with
    the largest score x (error(current) - error(higher)) / (bytes(higher) -
    bytes(current)) first, again and again, one tier at a time, until the
    next does not fit what is left of the headroom: deliberately, so that
    no upgrade comes before one of larger gain, at the cost of room a
    smaller one could have used. Ties go to the higher score, then the
    newest page, the reverse of a plan's. A page not yet scored takes the mean of the scored pages that
    can be upgraded. The upgrades fall into batches within
    `budget_seconds` each.

    The headroom kept is the net of each upgrade. While one runs, its new
    page is held beside the page it replaces, and, to a quantised tier, an
    FP16 page the shadow is uploaded to as well, for as long as the upgrade
    takes (#96)."""
    _check_pages(scores, tiers)
    now_seconds = time.monotonic() if now_seconds is None else now_seconds
    if last_downgrade_seconds is not None and last_downgrade_seconds > now_seconds:
        raise ValueError(f"the last downgrade, at {last_downgrade_seconds} s, is after now, "
                         f"{now_seconds} s: both times must come from one clock, "
                         f"time.monotonic()")
    level = thresholds.classify(headroom_bytes)
    available = headroom_bytes - thresholds.yellow_below_bytes - spike_bytes

    def held(reason: str) -> UpgradePlan:
        return UpgradePlan(level, max(0, available), (), (), reason)

    if level != GREEN:
        return held(f"the level is {level.value}; upgrades are made at GREEN only")
    if last_downgrade_seconds is not None and now_seconds - last_downgrade_seconds < cooldown_seconds:
        return held(f"cooldown: {now_seconds - last_downgrade_seconds:.2f} s since the last downgrade, "
                    f"{cooldown_seconds} s needed")
    if available <= 0:
        return held(f"headroom: {headroom_bytes} bytes leave none above T_high + "
                    f"{spike_bytes} bytes")

    upgradable = {key: tier for key, tier in tiers.items() if tier != "FP16" and key in shadowed}
    neutral = _neutral(scores, upgradable)

    def next_upgrade(key: tuple[int, int], current: str) -> Upgrade | None:
        up = TIERS.index(current) - 1
        if up < 0:
            return None
        higher = TIERS[up]
        taken = page_bytes[higher] - page_bytes[current]
        score = neutral if math.isnan(scores[key]) else scores[key]
        gain = score * (errors[current] - errors[higher]) / taken
        layer, page = key
        return Upgrade(layer, page, current, higher, taken, gain, score,
                       move_seconds[(current, higher)])

    def entry(u: Upgrade):
        # Ties to the higher score, then the newest page, then the layer.
        return -u.gain, -u.score, -u.page, -u.layer, u

    heap = [entry(u) for u in (next_upgrade(key, tier) for key, tier in upgradable.items())
            if u is not None]
    heapq.heapify(heap)

    chosen, left = [], available
    while heap and heap[0][-1].bytes_taken <= left:
        upgrade = heapq.heappop(heap)[-1]
        chosen.append(upgrade)
        left -= upgrade.bytes_taken
        further = next_upgrade((upgrade.layer, upgrade.page), upgrade.target_tier)
        if further is not None:
            heapq.heappush(heap, entry(further))
    return UpgradePlan(level, available, tuple(chosen), batches(chosen, level, budget_seconds),
                       "")


def batches(moves: list, level: Level, budget_seconds: float) -> tuple[tuple, ...]:
    """Downgrades or upgrades in batches, by batch_bounds on their seconds."""
    seconds = np.array([m.seconds for m in moves], dtype=np.float64)
    return tuple(tuple(moves[a:b]) for a, b in batch_bounds(seconds, level, budget_seconds))


def logged_tier_errors(model: str, log: str | Path = benchlog.DEFAULT_LOG) -> dict[str, float]:
    """Each tier's error, for `model`, from the last kv-quantisation-roundtrip
    entry logged for it: the mean of the keys' and values' relative mean
    squared error. FP16's is 0."""
    errors = {"FP16": 0.0}
    for entry in benchlog.read(log):
        if entry["kind"] == "kv-quantisation-roundtrip" and entry["model"] == model:
            (tier,) = entry["precision_tiers"]
            r = entry["results"]
            errors[tier] = (r["keys"]["relative_rms"] ** 2 + r["values"]["relative_rms"] ** 2) / 2
    missing = set(TIERS) - set(errors)
    if missing:
        raise ValueError(f"no roundtrip logged for {model} at {sorted(missing)}")
    return errors


def logged_move_seconds(model: str | None,
                        log: str | Path = benchlog.DEFAULT_LOG) -> dict[tuple[str, str], float]:
    """Each move's median time, by (current, target) tier, for `model`, from
    the requantisation-latency entries of the last commit that logged them
    for it (#97). A plan does not know which pages hold a shadow, so a
    downgrade from FP16 takes the slower of a first downgrade's and one with
    the shadow held. With `model` None, any model's: an estimate for one not
    yet measured."""
    entries = [e for e in benchlog.read(log) if e["kind"] == "requantisation-latency"
               and (model is None or e["model"] == model)]
    if not entries:
        raise ValueError(f"no requantisation latency logged for {model or 'any model'}")
    last = [e for e in entries if e["git_commit"] == entries[-1]["git_commit"]]
    seconds: dict[tuple[str, str], float] = {}
    for entry in last:
        pair = (entry["config"]["from_tier"], entry["config"]["to_tier"])
        seconds[pair] = max(entry["results"]["seconds"]["median"], seconds.get(pair, 0.0))
    # A plan downgrades one tier at a time; the other pairs are an upgrade's.
    missing = set(zip(TIERS, TIERS[1:])) - set(seconds)
    if missing:
        raise ValueError(f"no requantisation latency logged for {model} at {sorted(missing)}")
    return seconds
