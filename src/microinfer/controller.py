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
scored pages', or, before any page has a score, the same for every page.
Before any score exists the plan therefore goes breadth first in position
order, the oldest page first (#88), and a page not yet scored, usually one
of the newest, goes no deeper than a page of average importance. Ties are
broken by page, then by layer: the oldest first.

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
    are applied in. `short` if every page went as low as it can and the
    byte target was still not met."""

    level: Level
    target_bytes: int
    downgrades: tuple[Downgrade, ...]
    batches: tuple[tuple[Downgrade, ...], ...]
    short: bool

    @property
    def reclaimed_bytes(self) -> int:
        return sum(d.bytes_saved for d in self.downgrades)


def plan(headroom_bytes: int, thresholds: Thresholds, scores: dict[tuple[int, int], float],
         tiers: dict[tuple[int, int], str], page_bytes: dict[str, int],
         errors: dict[str, float], *, move_seconds: dict[tuple[str, str], float],
         margin_bytes: int = DEFAULT_MARGIN_BYTES,
         budget_seconds: float = DEFAULT_BUDGET_SECONDS) -> Plan:
    """The plan for this headroom: `scores` and `tiers` by (layer, page),
    for every page that may be downgraded; `page_bytes` and `errors` by
    tier; `move_seconds` by (current, target) tier."""
    for key, tier in tiers.items():
        if tier not in TIERS:
            raise ValueError(f"page {key} is at {tier!r}; a tier is one of {TIERS}")
        if key not in scores:
            raise ValueError(f"page {key} has no score; give NaN for one not yet scored")
        if scores[key] < 0:
            raise ValueError(f"page {key}'s score is {scores[key]}; a score is a mass, >= 0")

    level = thresholds.classify(headroom_bytes)
    target_bytes = max(0, thresholds.yellow_below_bytes + margin_bytes - headroom_bytes)
    if level == GREEN or target_bytes == 0:
        return Plan(level, 0, (), (), False)

    scored = [scores[key] for key in tiers if not math.isnan(scores[key])]
    neutral = sum(scored) / len(scored) if scored else 1.0

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

    heap = [entry(d) for d in (next_downgrade(key, tier) for key, tier in tiers.items())
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
                reclaimed < target_bytes)


def batches(downgrades: list[Downgrade], level: Level,
            budget_seconds: float) -> tuple[tuple[Downgrade, ...], ...]:
    """At RED, every downgrade at once. Otherwise the downgrades, in order,
    in batches as full as the budget allows; one longer than the budget is
    a batch of its own, and over it."""
    if not downgrades:
        return ()
    if level == RED:
        return (tuple(downgrades),)
    out, batch, spent = [], [], 0.0
    for downgrade in downgrades:
        if batch and spent + downgrade.seconds > budget_seconds:
            out.append(tuple(batch))
            batch, spent = [], 0.0
        batch.append(downgrade)
        spent += downgrade.seconds
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
