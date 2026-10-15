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
tokens, or stopped at <|endoftext|>, and goes on after one that stopped at
<|im_end|> (needles.Sample.turn_2). A question-aware run logs the answer's
first tokens, whether they open as the question asks, and whether the code
comes after them, after the plan: Qwen2.5-1.5B opens "The access code for
the ... archive is", which is not as asked, but holds no code yet.

One engine serves every condition: a run sets the engine's tier, whether it
adapts and scores, and its score source, which an emergency plan would use
too, for its own conversation, and puts them back after. NO_PLAN, the
adaptive, scoring engine with no plan applied, is a check rather than a
condition: its answer is FP16's.

Screening's FP16 runs (#148) are logged as RQ3's are, with screening true
in their config: they are the fp16 condition of the screened set, and an
analysis that pairs runs tells them apart by it.

The random source draws each sample's scores from a seed of its own,
sample_seed, so that the samples of a depth do not rank the needle's page
alike, and their outcomes stay unpaired but for the condition.
"""

from __future__ import annotations

import zlib
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from . import benchlog, controller, needles

EXPERIMENTS = ("question-blind", "question-aware")
SUMMARY_TOKENS = 64
ANSWER_TOKENS = 32
#: The answer's tokens the question-aware experiment decodes before its plan,
#: and what they are asked to be (needles.question_text).
AWARE_TOKENS = 4
AWARE_OPENING = "The access code is"
END_OF_TURN = "<|im_end|>"


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
#: The conditions a source at a level holds, question-blind's closing run (#149).
SOURCES = tuple(c.name for c in CONDITIONS if c.source is not None)


def condition(name: str) -> Condition:
    for c in (*CONDITIONS, NO_PLAN):
        if c.name == name:
            return c
    raise ValueError(f"no condition {name!r}; ADR-0017's are {[c.name for c in CONDITIONS]}, "
                     f"and {NO_PLAN.name!r} the check")


def code_after(opening: str, code: str) -> bool:
    """Whether an answer's opening holds none of the code yet: neither the
    code, nor, at its end, a start of it."""
    opening = opening.rstrip()
    return code not in opening and not any(opening.endswith(code[:k])
                                           for k in range(1, len(code)))


def sample_seed(sample_id: str, plan_seed: int) -> int:
    """The random source's seed for one sample: `plan_seed` and the sample's
    id, so that a sample draws the same scores whatever else is run."""
    return zlib.crc32(f"{plan_seed}:{sample_id}".encode())


def run(engine, sample: needles.Sample, cond: Condition, experiment: str, *,
        plan_seed: int = 0) -> dict:
    """One sample through its turns: what a log entry's results hold. The
    effective bits, the tiers' shares and the needle's pages' tiers are what
    the cache holds at the plan, or where it would be: the full pages before
    the question's answer, which adds its own at FP16."""
    if experiment not in EXPERIMENTS:
        raise ValueError(f"an experiment is one of {EXPERIMENTS}; got {experiment!r}")
    if engine.kv_cache != "paged" or engine.kv_halves != "both":
        raise ValueError("a run holds its cache on pages, both halves at a page's tier")
    held = (engine.kv_tier, engine.kv_adaptive, engine.kv_scoring, engine.kv_score_source)
    engine.kv_tier, engine.kv_adaptive, engine.kv_scoring = cond.tier, cond.adaptive, cond.adaptive
    engine.kv_score_source = cond.source or held[3]
    try:
        return _run(engine, sample, cond, experiment, sample_seed(sample.sample_id, plan_seed))
    finally:
        (engine.kv_tier, engine.kv_adaptive, engine.kv_scoring,
         engine.kv_score_source) = held


def _run(engine, sample: needles.Sample, cond: Condition, experiment: str, seed: int) -> dict:
    end_of_turn = engine.tokenizer.token_to_id(END_OF_TURN)
    opening = None
    with engine.conversation() as chat:
        summary = chat.turn(sample.turn_1, SUMMARY_TOKENS)
        stopped = int(summary[-1]) == end_of_turn
        turn_2 = sample.turn_2_after_eos if stopped else sample.turn_2
        if experiment == "question-aware":
            answer = chat.turn(turn_2, AWARE_TOKENS)
            answered = int(answer[-1]) in engine.eos_token_ids
            opening = engine.decode(answer)
        else:
            answer, answered = np.empty(0, dtype=np.int32), False
        plan_positions = chat.positions
        record = None
        if cond.source is not None:
            record = chat.plan(cond.share, source=cond.source, seed=seed)
        bits = chat.effective_bits()
        tiers = chat.tiers()
        if experiment == "question-blind":
            answer = chat.turn(turn_2, ANSWER_TOKENS)
        elif not answered:
            answer = np.concatenate([answer, chat.turn(np.empty(0, np.int32),
                                                       ANSWER_TOKENS - AWARE_TOKENS)])
    first, last = sample.needle_pages
    needle_tiers = Counter(controller.TIERS[t] for t in tiers[:, first:last + 1].ravel())
    full = tiers[:, :plan_positions // sample.page_tokens]
    shares = Counter(controller.TIERS[t] for t in full.ravel())
    text = engine.decode(answer)
    plan = None
    if record is not None:
        plan = {k: v for k, v in record.controlled.items()
                if k not in ("effective_bits", "candidate_effective_bits")}
        plan.update(moves=len(record.plan), short=record.plan.short, ended=record.ended)
    return {
        "answer": text, "correct": needles.correct(text, sample.code),
        "answer_tokens": answer.tolist(), "summary": engine.decode(summary),
        "summary_length": len(summary), "summary_stopped": stopped,
        "aware_opening": opening,
        "opened_as_asked": None if opening is None else opening.startswith(AWARE_OPENING),
        "code_after_plan": None if opening is None else code_after(opening, sample.code),
        "plan_positions": plan_positions, "plan": plan,
        "random_seed": seed if cond.source == "random" else None,
        "effective_bits": bits["effective_bits"],
        "candidate_effective_bits": bits["candidate_effective_bits"],
        "tier_shares": {t: shares[t] / full.size for t in controller.TIERS if shares[t]},
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


def logged(model: str, needle_seed: int, haystack_tokens: int, commit: str,
           log: str | Path = benchlog.DEFAULT_LOG, *, screening: bool = False,
           plan_seed: int = 0) -> dict[tuple[str, str, str], bool]:
    """Whether each run logged at `commit` for `model`, the set of
    `needle_seed` and `haystack_tokens`, and `plan_seed` was answered
    correctly, by (sample id, experiment, condition): screening's runs, or
    the others; none before the log exists. A run of hours stopped part way
    is taken up again from them, at the same commit."""
    if not Path(log).exists():
        return {}
    return {(e["config"]["sample_id"], e["config"]["experiment"],
             e["config"]["condition"]["name"]): e["results"]["correct"]
            for e in benchlog.read(log)
            if e["kind"] == "rq3-needle" and e["git_commit"] == commit
            and e["model"] == model
            and bool(e["config"].get("screening")) == screening
            and e["config"]["needle_seed"] == needle_seed
            and e["config"]["haystack_tokens"] == haystack_tokens
            and e["config"]["plan_seed"] == plan_seed}


class ShortScreening(ValueError):
    """A screening with a depth short of right answers: what it kept and
    rejected, and each short depth's count of right ones."""

    def __init__(self, kept: list, rejected: list, short: dict[float, int], per_depth: int):
        self.kept, self.rejected, self.short = kept, rejected, short
        depths = ", ".join(f"{round(d * 100)}%: {n}" for d, n in short.items())
        super().__init__(f"depths short of {per_depth} answered correctly at FP16 ({depths}); "
                         f"draw more")


def screen(samples: Sequence, is_right: Callable[[needles.Sample], bool],
           per_depth: int = needles.SAMPLES_PER_DEPTH) -> tuple[list, list]:
    """ADR-0017's screening (#148): of `samples`, a set drawn with more than
    `per_depth` a depth, in its order, keep for each depth the first
    `per_depth` that FP16 answers correctly, `is_right`, and reject those
    before them that it does not. A sample past a depth's last kept is not
    run. The kept and the rejected, each in the set's order; every depth is
    screened before a ShortScreening names those short of `per_depth`."""
    kept, rejected, short = [], [], {}
    for depth in needles.DEPTHS:
        held = 0
        for sample in (s for s in samples if s.depth == depth):
            if held == per_depth:
                break
            if is_right(sample):
                kept.append(sample)
                held += 1
            else:
                rejected.append(sample)
        if held < per_depth:
            short[depth] = held
    if short:
        raise ShortScreening(kept, rejected, short, per_depth)
    return kept, rejected


def screened(model: str, needle_seed: int, haystack_tokens: int,
             log: str | Path = benchlog.DEFAULT_LOG) -> tuple[list[str], int]:
    """The screened set's sample ids, for `model` and the set of
    `needle_seed` and `haystack_tokens`, from the last complete
    needle-screening entry logged for them, and the samples a depth it was
    drawn with, which gives those ids their samples."""
    entries = [e for e in benchlog.read(log) if e["kind"] == "needle-screening"
               and e["model"] == model and e["config"]["needle_seed"] == needle_seed
               and e["config"]["haystack_tokens"] == haystack_tokens
               and e["results"].get("complete", True)]
    if not entries:
        raise ValueError(f"no complete needle-screening logged for {model} at seed "
                         f"{needle_seed} over haystacks of {haystack_tokens} tokens; run "
                         f"tools/screen_needles.py")
    return entries[-1]["results"]["kept"], entries[-1]["config"]["draws"]
