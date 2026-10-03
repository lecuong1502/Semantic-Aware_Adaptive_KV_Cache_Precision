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
of the newest, goes no deeper than a page of average importance. Ties are
broken by page, then by layer: the oldest first.

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
it is one batch, reclaimed at once. The budget is about 20 ms (#88): at
the 0.02-0.04 ms a downgrade took on Qwen2.5-1.5B, some 500 a batch.
"""

from __future__ import annotations

import heapq
import math
from dataclasses import dataclass
from pathlib import Path

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
    seconds: float


@dataclass(frozen=True)
class Plan:
    """The downgrades, in the order they were chosen, and the batches they
    are applied in. `short` if every page outside the recency floor went as
    low as it can and the byte target was still not met. `recency_floor`
    is the W the plan kept."""

    level: Level
    target_bytes: int
    downgrades: tuple[Downgrade, ...]
    batches: tuple[tuple[Downgrade, ...], ...]
    short: bool
    recency_floor: int

    @property
    def reclaimed_bytes(self) -> int:
        return sum(d.bytes_saved for d in self.downgrades)


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
    `recency_floor` is downgraded."""
    if recency_floor < 0:
        raise ValueError(f"the recency floor is a count of positions, >= 0; got {recency_floor}")
    if page_tokens <= 0 or positions < 0:
        raise ValueError(f"a cache holds positions >= 0 on pages of > 0; got {positions} "
                         f"positions on pages of {page_tokens}")
    _check_pages(scores, tiers)

    level = thresholds.classify(headroom_bytes)
    target_bytes = max(0, thresholds.yellow_below_bytes + margin_bytes - headroom_bytes)
    if level == GREEN or target_bytes == 0:
        return Plan(level, 0, (), (), False, recency_floor)

    # A page is a candidate only if it is a page of positions whose last
    # position comes before the floor's first: no open page is, and the
    # partly filled last page never is.
    first_kept = positions - recency_floor
    candidates = {(layer, page): tier for (layer, page), tier in tiers.items()
                  if page >= 0 and (page + 1) * page_tokens <= first_kept}

    neutral = _neutral(scores, candidates)

    def next_downgrade(key: tuple[int, int], current: str) -> Downgrade | None:
        down = TIERS.index(current) + 1
        if down == len(TIERS):
            return None
        lower = TIERS[down]
        saved = page_bytes[current] - page_bytes[lower]
        score = neutral if math.isnan(scores[key]) else scores[key]
        cost = score * (errors[lower] - errors[current]) / saved
        layer, page = key
        return Downgrade(layer, page, current, lower, saved, cost,
                         move_seconds[(current, lower)])

    def entry(d: Downgrade):
        return d.cost, d.page, d.layer, d  # ties to the oldest page, then layer

    heap = [entry(d) for d in (next_downgrade(key, tier) for key, tier in candidates.items())
            if d is not None]
    heapq.heapify(heap)

    chosen, reclaimed = [], 0
    while heap and reclaimed < target_bytes:
        downgrade = heapq.heappop(heap)[-1]
        chosen.append(downgrade)
        reclaimed += downgrade.bytes_saved
        further = next_downgrade((downgrade.layer, downgrade.page), downgrade.target_tier)
        if further is not None:
            heapq.heappush(heap, entry(further))

    return Plan(level, target_bytes, tuple(chosen), batches(chosen, level, budget_seconds),
                reclaimed < target_bytes, recency_floor)


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
                 move_seconds: dict[tuple[str, str], float], now_seconds: float,
                 last_downgrade_seconds: float | None,
                 cooldown_seconds: float = DEFAULT_UPGRADE_COOLDOWN_SECONDS,
                 spike_bytes: int = DEFAULT_UPGRADE_SPIKE_BYTES,
                 budget_seconds: float = DEFAULT_BUDGET_SECONDS) -> UpgradePlan:
    """The upgrades for this headroom (#103), at GREEN only: at least
    `cooldown_seconds` after the last downgrade, and taking no more than the
    headroom above GREEN's threshold, T_high, plus `spike_bytes`, a P90
    spike (#88). `scores` and `tiers` by (layer, page), for the pages
    that may be upgraded, each from its shadow (#96): a page born at a
    quantised tier has none, and is the caller's to leave out. The clock is
    the caller's, in seconds; `last_downgrade_seconds` None if none was made.

    They come in the reverse of a downgrade plan's order: the upgrade with
    the largest score x (error(current) - error(higher)) / (bytes(higher) -
    bytes(current)) first, again and again, one tier at a time, until the
    next does not fit what is left of the headroom: deliberately, so that
    no upgrade comes before one of larger gain, at the cost of room a
    smaller one could have used. Ties go to the newest page, the reverse of
    a plan's. A page not yet scored takes the mean of the scored pages that
    can be upgraded. The upgrades fall into batches within
    `budget_seconds` each.

    The headroom kept is the net of each upgrade. While one runs, its new
    page is held beside the page it replaces, and, to a quantised tier, an
    FP16 page the shadow is uploaded to as well, for as long as the upgrade
    takes (#96)."""
    _check_pages(scores, tiers)
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

    neutral = _neutral(scores, [key for key, tier in tiers.items() if tier != "FP16"])

    def next_upgrade(key: tuple[int, int], current: str) -> Upgrade | None:
        up = TIERS.index(current) - 1
        if up < 0:
            return None
        higher = TIERS[up]
        taken = page_bytes[higher] - page_bytes[current]
        score = neutral if math.isnan(scores[key]) else scores[key]
        gain = score * (errors[current] - errors[higher]) / taken
        layer, page = key
        return Upgrade(layer, page, current, higher, taken, gain, move_seconds[(current, higher)])

    def entry(u: Upgrade):
        return -u.gain, -u.page, -u.layer, u  # ties to the newest page, then layer

    heap = [entry(u) for u in (next_upgrade(key, tier) for key, tier in tiers.items())
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
    """Downgrades or upgrades in batches. At RED, every move at once.
    Otherwise the moves, in order, in batches as full as the budget allows;
    one longer than the budget is a batch of its own, and over it."""
    if not moves:
        return ()
    if level == RED:
        return (tuple(moves),)
    out, batch, spent = [], [], 0.0
    for move in moves:
        if batch and spent + move.seconds > budget_seconds:
            out.append(tuple(batch))
            batch, spent = [], 0.0
        batch.append(move)
        spent += move.seconds
    out.append(tuple(batch))
    return tuple(out)


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


def logged_move_seconds(model: str,
                        log: str | Path = benchlog.DEFAULT_LOG) -> dict[tuple[str, str], float]:
    """Each move's median time, by (current, target) tier, for `model`, from
    the requantisation-latency entries of the last commit that logged them
    for it (#97). A plan does not know which pages hold a shadow, so a
    downgrade from FP16 takes the slower of a first downgrade's and one with
    the shadow held."""
    entries = [e for e in benchlog.read(log)
               if e["kind"] == "requantisation-latency" and e["model"] == model]
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
