"""Continuation fidelity on WikiText-2: RQ3's secondary task (#153).

ADR-0017 compresses a WikiText passage part way through and teacher-forces
the rest. A **passage** is PREFIX tokens of the WikiText-2 test split,
prefilled; its **warm** tokens, WARM more, forced one position a step, so
that the scorer folds in attention mass as it does while decoding, as turn
1's summary gives the needle runs their scores; then the controlled plan of
a condition; then its **measured** tokens, MEASURED more, forced one position
a step, whose logits are compared with the reference's. The passages are
consecutive and do not overlap, from the start of the split, so that they
need no seed.

The **reference** is the same passage with no plan applied, on the same
adaptive, scoring engine at FP16, from the same snapshot: what the plan
alone changes is measured. It is FP16's to the token
(tests/test_rq3.py::test_fp16_and_no_plan_applied_give_the_same_answer); a
reference whose cache went below FP16, in an emergency, is refused. The
prefix and warm tokens are held alike under every source condition, so they
run once, and each condition and the reference start from their snapshot
(#172).

For each position measured:
- **top-1 agreement**, whether the condition's highest logit is the
  reference's;
- **KL divergence**, KL(reference || condition), of the softmaxed logits, in
  float64.

A passage's result under a condition is the mean of each over its measured
positions. The summary pools the passages of a run: each condition's mean
top-1 agreement and mean KL with bootstrap 95% intervals over passages, and,
at each level, the paired difference in mean KL of the semantic source
against uniform and against random, passage by passage, with its bootstrap
interval, as ADR-0017 fixes; and the same of top-1 agreement beside it.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import NamedTuple

import numpy as np

from . import benchlog, controller, rq3, rq3_summary

PREFIX, WARM, MEASURED = 2048, 64, 128
PASSAGES = 50
EXPERIMENT = "continuation"
KIND = "rq3-continuation"
#: The bootstrap's seed: the summary repeats exactly.
SEED = 153
#: Rows of logits put through float64 at a time.
_ROWS = 32


class Shape(NamedTuple):
    """What makes two runs' passages alike: their parts' lengths, and the
    random source's plan seed."""

    prefix: int = PREFIX
    warm: int = WARM
    measured: int = MEASURED
    plan_seed: int = 0

    @classmethod
    def of(cls, entry: dict) -> Shape:
        c = entry["config"]
        return cls(c["prefix_tokens"], c["warm_tokens"], c["measured_tokens"], c["plan_seed"])


@dataclass(frozen=True)
class Passage:
    """One passage of the split: its id, where it starts in the split's
    tokens, and its three parts."""

    passage_id: str
    start: int
    prefix: np.ndarray
    warm: np.ndarray
    measured: np.ndarray


def passage_id(i: int) -> str:
    return f"p{i:02d}"


def passages(tokens: np.ndarray, count: int = PASSAGES, shape: Shape = Shape()) -> list[Passage]:
    """The first `count` consecutive passages of `tokens`, the split's token
    ids; a ValueError if the split holds fewer."""
    length = shape.prefix + shape.warm + shape.measured
    if count * length > len(tokens):
        raise ValueError(f"{count} passages of {length} tokens need {count * length}; the "
                         f"split holds {len(tokens)}")
    out = []
    for i in range(count):
        start = i * length
        ids = np.asarray(tokens[start:start + length], dtype=np.int32)
        warm_end = shape.prefix + shape.warm
        out.append(Passage(passage_id(i), start, ids[:shape.prefix],
                           ids[shape.prefix:warm_end], ids[warm_end:]))
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
    """Each of `conds` on one passage, in their order: what a log entry's
    results hold. Every one makes its plan at FP16, adaptive
    (rq3.shares_turn_1), so the prefix and warm tokens run once, and the
    reference and each condition start from their snapshot."""
    elsewhere = [c.name for c in conds if not rq3.shares_turn_1(c)]
    if elsewhere:
        raise ValueError(f"continuation fidelity measures plans made at FP16: {elsewhere} "
                         f"hold the passage otherwise")
    with rq3.set_for(engine, rq3.NO_PLAN), engine.conversation(for_snapshot=True) as chat:
        chat.force(passage.prefix)
        chat.force(passage.warm, one_by_one=True)
        snapshot = chat.snapshot()
    with rq3.set_for(engine, rq3.NO_PLAN), engine.conversation(start=snapshot) as chat:
        reference = chat.force(passage.measured, one_by_one=True, logits=True)
        held = chat.tiers()
        if (held[held >= 0] != controller.TIERS.index("FP16")).any():
            raise RuntimeError(f"{passage.passage_id}'s reference went below FP16, in an "
                               f"emergency: it is no reference")
    seed = rq3.sample_seed(passage.passage_id, plan_seed)
    for cond in conds:
        with rq3.set_for(engine, cond), engine.conversation(start=snapshot) as chat:
            record = None
            if cond.source is not None:
                record = chat.plan(cond.share, source=cond.source, seed=seed)
            bits = chat.effective_bits()
            tiers = chat.tiers()
            logits = chat.force(passage.measured, one_by_one=True, logits=True)
        measured = measures(reference, logits)
        yield cond, {
            "top1": float(np.mean(measured["top1"])), "kl_mean": float(np.mean(measured["kl"])),
            "kl": measured["kl"], "top1_by_position": measured["top1"],
            "plan": rq3.plan_summary(record),
            "random_seed": seed if cond.source == "random" else None,
            "effective_bits": bits["effective_bits"],
            "candidate_effective_bits": bits["candidate_effective_bits"],
            # every page held at the plan, the open last one among them
            "tier_shares": rq3.tier_shares(tiers[tiers >= 0])}


def config(passage: Passage, cond: rq3.Condition, **more) -> dict:
    """What a log entry's config names of a run."""
    return {"passage_id": passage.passage_id, "start": passage.start,
            "prefix_tokens": len(passage.prefix), "warm_tokens": len(passage.warm),
            "measured_tokens": len(passage.measured), "condition": asdict(cond),
            "experiment": EXPERIMENT, **more}


def logged(model: str, shape: Shape, commit: str,
           log=benchlog.DEFAULT_LOG) -> set[tuple[str, str]]:
    """The (passage, condition) runs logged for `model` and passages of
    `shape` at the code of `commit` (benchlog.same_code): a run of hours
    stopped part way is taken up again from them."""
    if not Path(log).exists():
        return set()
    held = benchlog.same_code_as(commit, log)
    return {(e["config"]["passage_id"], e["config"]["condition"]["name"])
            for e in benchlog.read(log)
            if e["kind"] == KIND and e["model"] == model and Shape.of(e) == shape
            and held(e["git_commit"])}


def summarise(entries: Sequence[dict], model: str, shape: Shape, count: int,
              conditions: Sequence[str] = rq3.SOURCE_CONDITIONS) -> dict:
    """The continuation summary of the first `count` passages' runs in
    `entries` for `model` and passages of `shape`, the latest of each under
    each of `conditions`, as the module describes; a ValueError naming what
    is missing, so that a run stopped part way is not pooled as though it
    were whole."""
    latest: dict[tuple[str, str], dict] = {}
    for e in entries:
        if e["kind"] == KIND and e["model"] == model and Shape.of(e) == shape:
            latest[(e["config"]["passage_id"], e["config"]["condition"]["name"])] = e
    ids = [passage_id(i) for i in range(count)]
    missing = [(p, c) for p in ids for c in conditions if (p, c) not in latest]
    if missing:
        raise ValueError(f"no run of {len(missing)} of the {count} passages' conditions, such "
                         f"as {missing[:3]}; run tools/continuation.py for them")
    rng = np.random.default_rng(SEED)
    runs = {c: [latest[(p, c)] for p in ids] for c in conditions}
    top1 = {c: np.array([e["results"]["top1"] for e in runs[c]]) for c in conditions}
    kl = {c: np.array([e["results"]["kl_mean"] for e in runs[c]]) for c in conditions}

    def interval(values: np.ndarray) -> list[float]:
        return [float(v) for v in rq3_summary.bootstrap(values[None, :], rng)[0]]

    by_condition = {}
    for c in conditions:
        plans = [e["results"]["plan"] for e in runs[c]]
        by_condition[c] = {
            "top1": float(top1[c].mean()), "top1_bootstrap_95": interval(top1[c]),
            "kl_mean": float(kl[c].mean()), "kl_bootstrap_95": interval(kl[c]),
            "effective_bits": float(np.mean([e["results"]["effective_bits"] for e in runs[c]])),
            "plans_short_of_target": sum(1 for p in plans
                                         if p is not None and not p["target_met"])}
    paired = {}
    for share in rq3_summary.LEVELS:
        names = {c.source: c.name for c in rq3.CONDITIONS if c.share == share}
        if names["semantic"] not in conditions:
            continue
        level = rq3_summary.level_name(share)
        paired[level] = {}
        for other in rq3_summary.OTHER_SOURCES:
            if names[other] not in conditions:
                continue
            d_kl = kl[names["semantic"]] - kl[names[other]]
            d_top1 = top1[names["semantic"]] - top1[names[other]]
            paired[level][f"semantic-{other}"] = {
                "kl_difference": float(d_kl.mean()), "kl_difference_bootstrap_95": interval(d_kl),
                "semantic_lower_kl": int((d_kl < 0).sum()),
                "top1_difference": float(d_top1.mean()),
                "top1_difference_bootstrap_95": interval(d_top1)}
    return {"model": model, "passages": count, "prefix_tokens": shape.prefix,
            "warm_tokens": shape.warm, "measured_tokens": shape.measured,
            "plan_seed": shape.plan_seed,
            "bootstrap": {"resamples": rq3_summary.RESAMPLES, "seed": SEED},
            "by_condition": by_condition, "paired": paired,
            "commits": sorted({e["git_commit"] for c in conditions for e in runs[c]}),
            "entries": [e["sha256"] for c in conditions for e in runs[c]]}
