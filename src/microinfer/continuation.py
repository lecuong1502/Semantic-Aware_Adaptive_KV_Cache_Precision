"""Continuation fidelity on WikiText-2: RQ3's secondary task (#153).

ADR-0017 compresses a WikiText passage part way through and teacher-forces
the rest. A **passage** is PREFIX tokens of the WikiText-2 test split,
prefilled; WARM more, forced one position a step, so that the scorer folds
in attention mass as it does while decoding, as turn 1's summary gives the
needle runs their scores; then the controlled plan of a condition; then
MEASURED more, forced one position a step, whose logits are compared with
FP16's. The passages are consecutive and do not overlap, from the start of
the split, so that they need no seed.

FP16's logits are those of the same passage with no plan applied: the
adaptive, scoring engine changes nothing until a plan is (rq3.NO_PLAN). The
prefix and warm tokens are held alike under every source condition, so they
run once, and each condition and the reference start from their snapshot
(#172).

For each position measured:
- **top-1 agreement**, whether the condition's highest logit is FP16's;
- **KL divergence**, KL(FP16 || condition), of the softmaxed logits, in
  float64.

A passage's result under a condition is the mean of each over its measured
positions. The summary pools the passages: each condition's mean top-1
agreement and mean KL with bootstrap 95% intervals over passages, and, at
each level, the paired difference in mean KL of the semantic source against
uniform and against random, passage by passage, with its bootstrap
interval, as ADR-0017 fixes; and the same of top-1 agreement beside it.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterator, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from . import benchlog, controller, rq3

PREFIX, WARM, MEASURED = 2048, 64, 128
PASSAGES = 50
EXPERIMENT = "continuation"
KIND = "rq3-continuation"
#: The bootstrap's resamples and seed: the summary repeats exactly.
RESAMPLES, SEED = 10_000, 153
#: Rows of logits put through float64 at a time.
_ROWS = 32


@dataclass(frozen=True)
class Passage:
    """One passage of the split: its id, where it starts in the split's
    tokens, and its three parts."""

    passage_id: str
    start: int
    prefix: np.ndarray
    warm: np.ndarray
    measured: np.ndarray


def passages(tokens: np.ndarray, count: int = PASSAGES, *, prefix: int = PREFIX,
             warm: int = WARM, measured: int = MEASURED) -> list[Passage]:
    """The first `count` consecutive passages of `tokens`, the split's token
    ids; a ValueError if the split holds fewer."""
    length = prefix + warm + measured
    if count * length > len(tokens):
        raise ValueError(f"{count} passages of {length} tokens need {count * length}; the "
                         f"split holds {len(tokens)}")
    out = []
    for i in range(count):
        start = i * length
        ids = np.asarray(tokens[start:start + length], dtype=np.int32)
        out.append(Passage(f"p{i:02d}", start, ids[:prefix], ids[prefix:prefix + warm],
                           ids[prefix + warm:]))
    return out


def _log_softmax(rows: np.ndarray) -> np.ndarray:
    rows = rows.astype(np.float64)
    top = rows.max(axis=1, keepdims=True)
    return rows - top - np.log(np.exp(rows - top).sum(axis=1, keepdims=True))


def measures(reference: np.ndarray, logits: np.ndarray) -> dict:
    """Top-1 agreement and KL(reference || logits) at each position, (n,
    vocab) fp32 logits each, in float64."""
    if reference.shape != logits.shape:
        raise ValueError(f"logits of {logits.shape} against a reference of {reference.shape}")
    kl = []
    for first in range(0, len(reference), _ROWS):
        p = _log_softmax(reference[first:first + _ROWS])
        q = _log_softmax(logits[first:first + _ROWS])
        kl.extend((np.exp(p) * (p - q)).sum(axis=1).tolist())
    agree = (reference.argmax(axis=1) == logits.argmax(axis=1)).tolist()
    return {"top1": agree, "kl": kl}


def runs(engine, passage: Passage, conds: Sequence[rq3.Condition], *,
         plan_seed: int = 0) -> Iterator[tuple[rq3.Condition, dict]]:
    """Each of `conds`, conditions that share turn 1 (rq3.shares_turn_1), on
    one passage, in their order: what a log entry's results hold. The prefix
    and warm tokens run once, and the reference and each condition start
    from their snapshot."""
    unshared = [c.name for c in conds if not rq3.shares_turn_1(c)]
    if unshared:
        raise ValueError(f"continuation fidelity measures plans made at FP16: {unshared} "
                         f"hold the passage otherwise")
    with rq3.set_for(engine, rq3.NO_PLAN), engine.conversation(for_snapshot=True) as chat:
        chat.force(passage.prefix)
        chat.force(passage.warm, one_by_one=True)
        snapshot = chat.snapshot()
    with rq3.set_for(engine, rq3.NO_PLAN), engine.conversation(start=snapshot) as chat:
        reference = chat.force(passage.measured, one_by_one=True, logits=True)
    seed = rq3.sample_seed(passage.passage_id, plan_seed)
    for cond in conds:
        with rq3.set_for(engine, cond), engine.conversation(start=snapshot) as chat:
            record = None
            if cond.source is not None:
                record = chat.plan(cond.share, source=cond.source, seed=seed)
            bits = chat.effective_bits()
            tiers = chat.tiers()
            logits = chat.force(passage.measured, one_by_one=True, logits=True)
        yield cond, _results(measures(reference, logits), record, bits, tiers,
                             seed if cond.source == "random" else None)


def _results(measured: dict, record, bits: dict, tiers: np.ndarray, seed: int | None) -> dict:
    full = tiers[tiers >= 0]
    shares = Counter(controller.TIERS[t] for t in full.ravel())
    plan = None
    if record is not None:
        plan = {k: v for k, v in record.controlled.items()
                if k not in ("effective_bits", "candidate_effective_bits")}
        plan.update(moves=len(record.plan), short=record.plan.short, ended=record.ended)
    return {"top1": float(np.mean(measured["top1"])), "kl_mean": float(np.mean(measured["kl"])),
            "kl": measured["kl"], "top1_by_position": measured["top1"], "plan": plan,
            "random_seed": seed, "effective_bits": bits["effective_bits"],
            "candidate_effective_bits": bits["candidate_effective_bits"],
            "tier_shares": {t: shares[t] / full.size for t in controller.TIERS if shares[t]}}


def config(passage: Passage, cond: rq3.Condition, **more) -> dict:
    """What a log entry's config names of a run."""
    return {"passage_id": passage.passage_id, "start": passage.start,
            "prefix_tokens": len(passage.prefix), "warm_tokens": len(passage.warm),
            "measured_tokens": len(passage.measured), "condition": asdict(cond),
            "experiment": EXPERIMENT, **more}


def _shape(e: dict) -> tuple:
    c = e["config"]
    return c["prefix_tokens"], c["warm_tokens"], c["measured_tokens"], c["plan_seed"]


def logged(model: str, shape: tuple, commit: str,
           log=benchlog.DEFAULT_LOG) -> set[tuple[str, str]]:
    """The (passage, condition) runs logged for `model` and passages of
    `shape`, (prefix, warm, measured, plan seed), at the code of `commit`
    (benchlog.same_code): a run of hours stopped part way is taken up
    again from them."""
    if not Path(log).exists():
        return set()
    same: dict[str, bool] = {}

    def held(other: str) -> bool:
        if other not in same:
            same[other] = benchlog.same_code(other, commit, log)
        return same[other]

    return {(e["config"]["passage_id"], e["config"]["condition"]["name"])
            for e in benchlog.read(log)
            if e["kind"] == KIND and e["model"] == model and _shape(e) == shape
            and held(e["git_commit"])}


def summarise(entries: Sequence[dict], model: str, shape: tuple,
              conditions: Sequence[str] = rq3.SOURCE_CONDITIONS) -> dict:
    """The continuation summary of `entries`' runs for `model` and passages
    of `shape`, the latest of each passage under each of `conditions`, as
    the module describes; a ValueError naming what is missing."""
    latest: dict[tuple[str, str], dict] = {}
    for e in entries:
        if e["kind"] == KIND and e["model"] == model and _shape(e) == shape:
            latest[(e["config"]["passage_id"], e["config"]["condition"]["name"])] = e
    ids = sorted({p for p, _ in latest})
    missing = [(p, c) for p in ids for c in conditions if (p, c) not in latest]
    if not ids or missing:
        raise ValueError(f"no run of {len(missing) or 'any'} passages' conditions, such as "
                         f"{missing[:3]}; run tools/continuation.py for them")
    rng = np.random.default_rng(SEED)
    top1 = {c: np.array([latest[(p, c)]["results"]["top1"] for p in ids]) for c in conditions}
    kl = {c: np.array([latest[(p, c)]["results"]["kl_mean"] for p in ids]) for c in conditions}

    def interval(values: np.ndarray) -> list[float]:
        picks = rng.integers(0, len(values), size=(RESAMPLES, len(values)))
        return [float(v) for v in np.percentile(values[picks].mean(axis=1), [2.5, 97.5])]

    by_condition = {c: {"top1": float(top1[c].mean()), "top1_bootstrap_95": interval(top1[c]),
                        "kl_mean": float(kl[c].mean()), "kl_bootstrap_95": interval(kl[c]),
                        "effective_bits": float(np.mean(
                            [latest[(p, c)]["results"]["effective_bits"] for p in ids]))}
                    for c in conditions}
    paired = {}
    for share in sorted({rq3.condition(c).share for c in conditions
                         if rq3.condition(c).source is not None}):
        level = f"{round(share * 100)}%"
        names = {c.source: c.name for c in rq3.CONDITIONS if c.share == share}
        if "semantic" not in names or names["semantic"] not in conditions:
            continue
        paired[level] = {}
        for other in ("uniform", "random"):
            if names.get(other) not in conditions:
                continue
            d_kl = kl[names["semantic"]] - kl[names[other]]
            d_top1 = top1[names["semantic"]] - top1[names[other]]
            paired[level][f"semantic-{other}"] = {
                "kl_difference": float(d_kl.mean()), "kl_difference_bootstrap_95": interval(d_kl),
                "semantic_lower_kl": int((d_kl < 0).sum()),
                "top1_difference": float(d_top1.mean()),
                "top1_difference_bootstrap_95": interval(d_top1)}
    prefix, warm, measured, plan_seed = shape
    return {"model": model, "passages": len(ids), "prefix_tokens": prefix,
            "warm_tokens": warm, "measured_tokens": measured, "plan_seed": plan_seed,
            "bootstrap": {"resamples": RESAMPLES, "seed": SEED},
            "by_condition": by_condition, "paired": paired,
            "entries": [latest[(p, c)]["sha256"] for c in conditions for p in ids]}
