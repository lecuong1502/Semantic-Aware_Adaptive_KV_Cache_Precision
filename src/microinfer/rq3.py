"""RQ3's harness: one needle sample through its turns under one condition (#147).

ADR-0017 answers RQ3 on needle retrieval at equal bytes. A **condition** is
how the cache is held:
- **fp16:** every page at FP16, no plan;
- **a source at a level:** semantic, uniform or random scores, and a
  controlled plan that reclaims 70% or 80% of the FP16 bytes of the pages
  outside the recency floor (#142);
- **a static tier:** every page born at INT4 or INT2, for the whole
  session, the uniform-quantisation family.

An **experiment** is when the plan is made:
- **question-blind**, the primary: turn 1, the document and a general
  request, answered in up to 64 tokens; the plan; then turn 2, the
  question, answered in up to 32;
- **question-aware**, an upper bound: turn 1, then the question, whose
  answer's first 4 tokens, "The access code is", are decoded before the
  plan, so the scorer has seen the question attended to; then the rest of
  the answer, after them, in the same 32 tokens in all.

A turn stops at its end of turn. Turn 2 closes a turn 1 cut short at its
tokens, and goes on after one that stopped (needles.Sample.turn_2).

One engine serves every condition: a run sets the engine's tier, and
whether it adapts and scores, for its own conversation. NO_PLAN, the
adaptive, scoring engine with no plan applied, is a check rather than a
condition: its answer is FP16's.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass

import numpy as np

from . import controller, needles

EXPERIMENTS = ("question-blind", "question-aware")
SUMMARY_TOKENS = 64
ANSWER_TOKENS = 32
#: The answer's tokens the question-aware experiment decodes before its plan.
AWARE_TOKENS = 4


@dataclass(frozen=True)
class Condition:
    """How a run holds its cache: a controlled plan from `source` for
    `share`, a static `tier`, or neither, FP16."""

    name: str
    source: str | None = None
    share: float | None = None
    tier: str = "FP16"
    #: Whether the engine adapts and scores, as a controlled plan needs.
    adaptive: bool = False

    def __post_init__(self):
        if self.source is not None:
            object.__setattr__(self, "adaptive", True)


CONDITIONS = (
    Condition("fp16"),
    *(Condition(f"{source}-{round(share * 100)}", source=source, share=share)
      for source in ("semantic", "uniform", "random") for share in (0.7, 0.8)),
    Condition("static-int4", tier="INT4"),
    Condition("static-int2", tier="INT2"),
)
NO_PLAN = Condition("no-plan", adaptive=True)


def condition(name: str) -> Condition:
    for c in (*CONDITIONS, NO_PLAN):
        if c.name == name:
            return c
    raise ValueError(f"no condition {name!r}; ADR-0017's are {[c.name for c in CONDITIONS]}, "
                     f"and {NO_PLAN.name!r} the check")


def run(engine, sample: needles.Sample, cond: Condition, experiment: str, *,
        plan_seed: int = 0) -> dict:
    """One sample through its turns: what a log entry's results hold. The
    effective bits and the needle's pages' tiers are what the cache holds
    when the question is answered from it, after any plan."""
    if experiment not in EXPERIMENTS:
        raise ValueError(f"an experiment is one of {EXPERIMENTS}; got {experiment!r}")
    engine.kv_tier = cond.tier
    engine.kv_adaptive = cond.adaptive
    engine.kv_scoring = cond.adaptive
    with engine.conversation() as chat:
        summary = chat.turn(sample.turn_1, SUMMARY_TOKENS)
        stopped = int(summary[-1]) in engine.eos_token_ids
        turn_2 = sample.turn_2_after_eos if stopped else sample.turn_2
        if experiment == "question-aware":
            answer = chat.turn(turn_2, AWARE_TOKENS)
            answered = int(answer[-1]) in engine.eos_token_ids
        else:
            answer, answered = np.empty(0, dtype=np.int32), False
        plan_positions = chat.positions
        record = None
        if cond.source is not None:
            record = chat.plan(cond.share, source=cond.source, seed=plan_seed)
        bits = chat.effective_bits()
        tiers = chat.tiers()
        first, last = sample.needle_pages
        needle_tiers = Counter(controller.TIERS[t] for t in tiers[:, first:last + 1].ravel())
        if experiment == "question-blind":
            answer = chat.turn(turn_2, ANSWER_TOKENS)
        elif not answered:
            answer = np.concatenate([answer, chat.turn(np.empty(0, np.int32),
                                                       ANSWER_TOKENS - AWARE_TOKENS)])
    text = engine.decode(answer)
    return {
        "answer": text, "correct": needles.correct(text, sample.code),
        "answer_tokens": answer.tolist(), "summary": engine.decode(summary),
        "summary_length": len(summary), "summary_stopped": stopped,
        "plan_positions": plan_positions,
        "plan": None if record is None else {
            **record.controlled, "moves": len(record.plan), "short": record.plan.short,
            "ended": record.ended},
        "effective_bits": bits["effective_bits"],
        "candidate_effective_bits": bits["candidate_effective_bits"],
        "needle_page_tiers": dict(sorted(needle_tiers.items())),
    }


def config(sample: needles.Sample, cond: Condition, experiment: str, **more) -> dict:
    """What a log entry's config names of a run."""
    return {"sample_id": sample.sample_id, "depth": sample.depth, "code": sample.code,
            "needle_positions": list(sample.needle_positions),
            "needle_pages": list(sample.needle_pages),
            "condition": asdict(cond), "experiment": experiment,
            "summary_tokens": SUMMARY_TOKENS, "answer_tokens": ANSWER_TOKENS,
            "aware_tokens": AWARE_TOKENS if experiment == "question-aware" else None,
            **more}
