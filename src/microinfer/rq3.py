"""RQ3's harness: one needle sample through its turns under one condition (#147).

ADR-0017 answers RQ3 on needle retrieval at equal bytes. A **condition** is
how the cache is held:
- **fp16:** every page at FP16, no plan;
- **a source at a level:** semantic, uniform or random scores, and a
  controlled plan that reclaims 70% or 80% of the FP16 bytes of the pages
  outside the recency floor (#142);
- **a static tier:** every page born at INT4 or INT2, for the whole
  session, the uniform-quantisation family;
- **an ablation** (ADR-0018, #158): the semantic source at 80%, the full
  policy, with one element changed, the recency floor, the scorer's
  cadence or EWMA weight, or an option of the planner, and no INT2 at
  70% too, against the semantic source there (`ablations`);
- **eviction at a level** (#160), H2O's family, a condition that does not
  hold the closing: the pages the semantic source's plan at that level
  would downgrade most are evicted instead, for the same bytes
  (`EVICTIONS`, Conversation.evict).

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

A sample's conditions that hold turn 1 alike, every source condition, at
FP16 and adaptive, run it once (`runs`, #172): until the plan nothing
differs between them, since the score source is read only by a plan. Turn
1, and question-aware the answer's first tokens, are kept on the host, and
each condition starts from their snapshot, which gives what it gives
alone.

The random source draws each sample's scores from a seed of its own,
sample_seed, so that the samples of a depth do not rank the needle's page
alike, and their outcomes stay unpaired but for the condition.
"""

from __future__ import annotations

import zlib
from collections import Counter
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, NamedTuple

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
    #: What an ablation changes (#158), each None for the engine's own: the
    #: plan's recency floor, the scorer's cadence and EWMA weight, and the
    #: planner's options.
    recency_floor: int | None = None
    score_every: int | None = None
    score_alpha: float | None = None
    options: controller.PlanOptions | None = None
    #: The element of the policy an ablation changes, as ADR-0018 names it.
    ablates: str | None = None
    #: Whether the source's plan is evicted rather than applied (#160).
    evicts: bool = False

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
#: The conditions of a score source at a level, question-blind's closing run (#149).
SOURCE_CONDITIONS = tuple(c.name for c in CONDITIONS if c.source is not None)
#: The levels, the shares a controlled plan reclaims.
LEVELS = sorted({c.share for c in CONDITIONS if c.share is not None})
#: Eviction at each level (#160), against the semantic source there.
EVICTIONS = tuple(Condition(f"evict-{round(share * 100)}", source="semantic", share=share,
                            evicts=True) for share in LEVELS)
EVICTION_CONDITIONS = tuple(c.name for c in EVICTIONS)


def source_condition(source: str, share: float) -> Condition:
    """The condition of `source`'s scores at the level `share`."""
    (cond,) = [c for c in CONDITIONS if c.source == source and c.share == share]
    return cond


#: The full policy ADR-0018's quality ablations change (#158): the closing
#: question's condition, question-blind at 80%.
FULL_POLICY = source_condition("semantic", 0.8)
#: The level no INT2 is also run at, against the semantic source there:
#: INT4 gives back 71% of a page's FP16 bytes at most, short of 80% (#158).
NO_INT2_SHARE = 0.7


def ablations(layer_weights: Sequence[float]) -> tuple[Condition, ...]:
    """ADR-0018's quality ablations, each the full policy with one element
    changed, and no INT2 at NO_INT2_SHARE too, the semantic source there
    with it changed (measured_against). The layer weights' ablation weighs
    scores by `layer_weights`, one weight a layer, as measured
    (controller.logged_layer_weights, #145)."""
    def ablated(suffix: str, element: str, against: Condition = FULL_POLICY,
                **change) -> Condition:
        return replace(against, name=f"{against.name}-{suffix}", ablates=element, **change)

    options = controller.PlanOptions
    return (
        ablated("w0", "recency floor", recency_floor=0),
        ablated("w512", "recency floor", recency_floor=512),
        ablated("every-step", "scoring cadence", score_every=1),
        ablated("alpha-0.05", "EWMA weight", score_alpha=0.05),
        ablated("alpha-0.5", "EWMA weight", score_alpha=0.5),
        ablated("importance-only", "marginal cost", options=options(importance_only=True)),
        ablated("no-int2", "the INT2 tier", options=options(lowest_tier="INT4")),
        ablated("no-int2", "the INT2 tier", source_condition("semantic", NO_INT2_SHARE),
                options=options(lowest_tier="INT4")),
        ablated("layer-weights", "layer weights",
                options=options(layer_weights=tuple(layer_weights))),
        ablated("first-page", "attention sinks", options=options(protect_first_page=True)),
    )


#: The ablations' names, in their order, which no layer weights change.
ABLATION_NAMES = tuple(c.name for c in ablations((1.0,)))


def measured_against(cond: Condition) -> Condition:
    """The condition an ablation is measured against: the semantic source at
    its level, the full policy but for no INT2 at NO_INT2_SHARE."""
    return source_condition(cond.source, cond.share)


#: The conditions the ablations are measured against, in their order.
ABLATED_FROM = tuple(dict.fromkeys(measured_against(c).name for c in ablations((1.0,))))


def condition(name: str) -> Condition:
    for c in (*CONDITIONS, *EVICTIONS, NO_PLAN):
        if c.name == name:
            return c
    raise ValueError(f"no condition {name!r}; ADR-0017's are {[c.name for c in CONDITIONS]}, "
                     f"eviction's {list(EVICTION_CONDITIONS)}, and {NO_PLAN.name!r} the check")


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


def _check(engine, experiment: str) -> None:
    if experiment not in EXPERIMENTS:
        raise ValueError(f"an experiment is one of {EXPERIMENTS}; got {experiment!r}")
    check_engine(engine)


def check_engine(engine) -> None:
    """Refuse an engine a run cannot hold its cache on."""
    if engine.kv_cache != "paged" or engine.kv_halves != "both":
        raise ValueError("a run holds its cache on pages, both halves at a page's tier")


#: The engine's attributes a condition sets, by the Condition field that
#: gives each; an ablation's, None in a field, leaves the engine's own.
_SETS = {"tier": "kv_tier", "adaptive": "kv_adaptive", "source": "kv_score_source",
         "recency_floor": "kv_recency_floor", "score_every": "kv_score_every",
         "score_alpha": "kv_score_alpha", "options": "kv_plan_options"}


@contextmanager
def set_for(engine, cond: Condition):
    """The engine set for `cond`: its tier, whether it adapts and scores,
    its score source, and what an ablation changes, put back after."""
    kept = {name: getattr(engine, name) for name in (*_SETS.values(), "kv_scoring")}
    try:
        for field, name in _SETS.items():
            if getattr(cond, field) is not None:
                setattr(engine, name, getattr(cond, field))
        engine.kv_scoring = cond.adaptive
        yield
    finally:
        for name, value in kept.items():
            setattr(engine, name, value)


def shares_turn_1(cond: Condition) -> bool:
    """Whether `cond` holds turn 1 as every source condition does: at FP16,
    adaptive and scoring at the engine's cadence and EWMA weight, which is
    all a cache is made with until the plan reads its score source and its
    options (#172). A static tier holds it at its own, and an ablation of
    the scorer (#158) scores it as no other does."""
    return (cond.adaptive and cond.tier == "FP16"
            and cond.score_every is None and cond.score_alpha is None)


#: What a run does after its question's answer, on the same conversation:
#: `then(engine, chat, sample, cond, seed, results)`, the results so far,
#: gives what to add to them. The recovery task's turn 3 (#155).
AfterAnswer = Callable[..., dict]


def run(engine, sample: needles.Sample, cond: Condition, experiment: str, *,
        plan_seed: int = 0, then: AfterAnswer | None = None) -> dict:
    """One sample through its turns: what a log entry's results hold. The
    effective bits, the tiers' shares and the needle's pages' tiers are what
    the cache holds at the plan, or where it would be: the full pages before
    the question's answer, which adds its own at FP16. With `then`, what it
    adds after the answer too."""
    _check(engine, experiment)
    seed = sample_seed(sample.sample_id, plan_seed)
    with set_for(engine, cond), engine.conversation() as chat:
        opened = before_plan(engine, chat, sample, experiment)
        return _close(engine, chat, sample, cond, experiment, seed, opened, then)


def runs(engine, sample: needles.Sample, conds: Sequence[Condition], experiment: str, *,
         plan_seed: int = 0, then: AfterAnswer | None = None) -> Iterator[tuple[Condition, dict]]:
    """Each of `conds` on one sample, with what `run` gives it: those that
    share turn 1 (shares_turn_1) first, in their order, from one turn 1,
    then the others, in theirs, each run alone (#172). Turn 1, and
    question-aware the answer's first tokens, are run once and each shared
    condition starts from their snapshot. Should the snapshot be refused, a
    page having gone below FP16 in an emergency, they run alone too."""
    _check(engine, experiment)
    seed = sample_seed(sample.sample_id, plan_seed)
    yield from from_one_opening(
        engine, conds, lambda chat: before_plan(engine, chat, sample, experiment),
        lambda chat, cond, opened: _close(engine, chat, sample, cond, experiment, seed, opened,
                                          then),
        lambda cond: run(engine, sample, cond, experiment, plan_seed=plan_seed, then=then))


def from_one_opening(engine, conds: Sequence[Condition], opening: Callable,
                     closing: Callable, alone: Callable) -> Iterator[tuple[Condition, dict]]:
    """Each of `conds`, those that share what is run before the plan
    (shares_turn_1) first, in their order, from one `opening`, then the
    others, in theirs, by `alone` (#172): `opening(chat)` runs it on a
    conversation made for a snapshot, and each shared condition starts from
    the snapshot and runs `closing(chat, cond, opened)`, `opened` what
    `opening` gave. A snapshot refused, a page having gone below FP16 in an
    emergency, runs them all alone. The needle runs' and the anchor-fact
    dialogue's (#154)."""
    shared = [c for c in conds if shares_turn_1(c)]
    rest = [c for c in conds if not shares_turn_1(c)]
    snapshot = None
    if len(shared) > 1:
        with set_for(engine, shared[0]), engine.conversation(for_snapshot=True) as chat:
            opened = opening(chat)
            try:
                snapshot = chat.snapshot()
            except ValueError:
                pass
    if snapshot is None:
        rest, shared = [*shared, *rest], []
    for cond in shared:
        with set_for(engine, cond), engine.conversation(start=snapshot) as chat:
            results = closing(chat, cond, opened)
        yield cond, results
    for cond in rest:
        yield cond, alone(cond)


@dataclass(frozen=True)
class _Opened:
    """What a run holds before its plan: turn 1's summary, whether it stopped
    at its end of turn, turn 2's prompt, and, question-aware, the answer's
    first tokens and whether they ended it."""

    summary: np.ndarray
    stopped: bool
    turn_2: np.ndarray
    answer: np.ndarray
    answered: bool
    opening: str | None


def before_plan(engine, chat, sample: needles.Sample, experiment: str) -> _Opened:
    """Turn 1, and question-aware the answer's first tokens: what a run
    holds before its plan, on `chat`."""
    end_of_turn = engine.tokenizer.token_to_id(END_OF_TURN)
    summary = chat.turn(sample.turn_1, SUMMARY_TOKENS)
    stopped = int(summary[-1]) == end_of_turn
    turn_2 = sample.turn_2_after_eos if stopped else sample.turn_2
    if experiment == "question-aware":
        answer = chat.turn(turn_2, AWARE_TOKENS)
        return _Opened(summary, stopped, turn_2, answer,
                       int(answer[-1]) in engine.eos_token_ids, engine.decode(answer))
    return _Opened(summary, stopped, turn_2, np.empty(0, dtype=np.int32), False, None)


def _close(engine, chat, sample: needles.Sample, cond: Condition, experiment: str, seed: int,
           opened: _Opened, then: AfterAnswer | None = None) -> dict:
    opening, answer = opened.opening, opened.answer
    planned = plan_at(chat, cond, seed)
    if experiment == "question-blind":
        answer = chat.turn(opened.turn_2, ANSWER_TOKENS)
    elif not opened.answered:
        answer = np.concatenate([answer, chat.turn(np.empty(0, np.int32),
                                                   ANSWER_TOKENS - AWARE_TOKENS)])
    text = engine.decode(answer)
    results = {
        "answer": text, "correct": needles.correct(text, sample.code),
        "answer_tokens": answer.tolist(), "summary": engine.decode(opened.summary),
        "summary_length": len(opened.summary), "summary_stopped": opened.stopped,
        "aware_opening": opening,
        "opened_as_asked": None if opening is None else opening.startswith(AWARE_OPENING),
        "code_after_plan": None if opening is None else code_after(opening, sample.code),
        "score_every": engine.kv_score_every, "score_alpha": engine.kv_score_alpha,
        **planned.results(cond, seed, sample.page_tokens),
        "needle_page_tiers": planned.page_tiers(sample.needle_pages),
    }
    if then is not None:
        results.update(then(engine, chat, sample, cond, seed, results))
    return results


@dataclass(frozen=True)
class Planned:
    """What a run's cache holds at its plan, or where it would be: the
    positions it holds, the plan's record, if one was applied, its
    effective bits and its pages' tiers."""

    positions: int
    record: Any
    bits: dict
    tiers: np.ndarray

    def results(self, cond: Condition, seed: int, page_tokens: int) -> dict:
        """What a log entry's results hold of it: the tiers' shares over the
        full pages, the plan's summary, and the random source's seed."""
        return {"plan_positions": self.positions, "plan": plan_summary(self.record),
                "random_seed": seed if cond.source == "random" else None,
                "effective_bits": self.bits["effective_bits"],
                "candidate_effective_bits": self.bits["candidate_effective_bits"],
                "tier_shares": tier_shares(self.tiers[:, :self.positions // page_tokens])}

    def page_tiers(self, pages: tuple[int, int]) -> dict[str, int]:
        """The tiers of every (layer, page) of `pages`, first and last, by count."""
        return page_tiers(self.tiers, pages)


def page_tiers(tiers: np.ndarray, pages: tuple[int, int]) -> dict[str, int]:
    """The tiers of every (layer, page) of `pages`, first and last, in
    `tiers`, indices into controller.PLACES, by count, evicted pages too."""
    first, last = pages
    counts = Counter(controller.PLACES[t] for t in tiers[:, first:last + 1].ravel())
    return dict(sorted(counts.items()))


def plan_at(chat, cond: Condition, seed: int) -> Planned:
    """`cond`'s controlled plan applied, or evicted, if it makes one, and
    what the cache then holds."""
    positions = chat.positions
    record = None
    if cond.source is not None:
        act = chat.evict if cond.evicts else chat.plan
        record = act(cond.share, source=cond.source, seed=seed)
    return Planned(positions, record, chat.effective_bits(), chat.tiers())


def plan_summary(record) -> dict | None:
    """What a log entry's results hold of a controlled plan's record: its
    controlled fields but the effective bits, given beside it, its moves,
    whether it fell short, how it ended, and the recency floor and options
    it was made with (#158); None for no plan."""
    if record is None:
        return None
    plan = {k: v for k, v in record.controlled.items()
            if k not in ("effective_bits", "candidate_effective_bits")}
    plan.update(moves=len(record.plan), short=record.plan.short, ended=record.ended,
                recency_floor=record.plan.recency_floor, options=asdict(record.plan.options))
    return plan


def tier_shares(tiers: np.ndarray) -> dict[str, float]:
    """Each tier's share of `tiers`, indices into controller.PLACES, and
    evicted pages' (#160)."""
    counts = Counter(controller.PLACES[t] for t in tiers.ravel())
    return {t: counts[t] / tiers.size for t in controller.PLACES if counts[t]}


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
    """The rq3-needle runs already logged, by (sample id, experiment,
    condition), each with whether it was answered correctly; none before
    the log exists. A run counts when:
    - its commit holds the code of `commit`, though the log may differ
      (benchlog.same_code), so that a log committed part way through does
      not start the run again;
    - its model, needle seed, haystack length and plan seed are these;
    - it is screening's when `screening`, and not otherwise.
    The issue is not matched: the same run made for another ticket is the
    same run. A run of hours stopped part way is taken up again from them."""
    if not Path(log).exists():
        return {}
    held = benchlog.same_code_as(commit, log)
    return {(e["config"]["sample_id"], e["config"]["experiment"],
             e["config"]["condition"]["name"]): e["results"]["correct"]
            for e in benchlog.read(log)
            if e["kind"] == "rq3-needle" and e["model"] == model and held(e["git_commit"])
            and bool(e["config"].get("screening")) == screening
            and e["config"]["needle_seed"] == needle_seed
            and e["config"]["haystack_tokens"] == haystack_tokens
            and e["config"]["plan_seed"] == plan_seed}


@dataclass(frozen=True)
class Grouping:
    """What a set is screened by: the value `of` each sample that groups
    it, the `groups` in their order, each shown by its `label`, and what
    the groups are `named` together."""

    groups: tuple
    of: Callable[[Any], Any]
    label: Callable[[Any], str]
    named: str


#: The needle set's grouping, by depth (#148).
BY_DEPTH = Grouping(needles.DEPTHS, lambda s: s.depth, lambda d: f"{round(d * 100)}%", "depths")


class ShortScreening(ValueError):
    """A screening with a group short of right answers: what it kept and
    rejected, and each short group's count of right ones."""

    def __init__(self, kept: list, rejected: list, short: dict, per_group: int,
                 grouping: Grouping = BY_DEPTH):
        self.kept, self.rejected, self.short = kept, rejected, short
        shown = ", ".join(f"{grouping.label(g)}: {n}" for g, n in short.items())
        super().__init__(f"{grouping.named} short of {per_group} answered correctly at FP16 "
                         f"({shown}); draw more")


def screen(samples: Sequence, is_right: Callable[[Any], bool],
           per_group: int = needles.SAMPLES_PER_DEPTH,
           grouping: Grouping = BY_DEPTH) -> tuple[list, list]:
    """ADR-0017's screening (#148): of `samples`, a set drawn with more than
    `per_group` a group, a depth by default, in its order, keep for each
    group the first `per_group` that FP16 answers correctly, `is_right`,
    and reject those before them that it does not. A sample past a group's
    last kept is not run. The kept and the rejected, each in the set's
    order; every group is screened before a ShortScreening names those
    short of `per_group`. The anchor-fact set is grouped by gap (#154)."""
    kept, rejected, short = [], [], {}
    for g in grouping.groups:
        held = 0
        for sample in (s for s in samples if grouping.of(s) == g):
            if held == per_group:
                break
            if is_right(sample):
                kept.append(sample)
                held += 1
            else:
                rejected.append(sample)
        if held < per_group:
            short[g] = held
    if short:
        raise ShortScreening(kept, rejected, short, per_group, grouping)
    return kept, rejected


def last_screening(entries: Sequence[dict], model: str, needle_seed: int,
                   haystack_tokens: int) -> dict:
    """The last complete needle-screening entry in `entries` for `model` and
    the set of `needle_seed` and `haystack_tokens`; a ValueError if there is
    none."""
    found = [e for e in entries if e["kind"] == "needle-screening"
             and e["model"] == model and e["config"]["needle_seed"] == needle_seed
             and e["config"]["haystack_tokens"] == haystack_tokens
             and e["results"].get("complete", True)]
    if not found:
        raise ValueError(f"no complete needle-screening logged for {model} at seed "
                         f"{needle_seed} over haystacks of {haystack_tokens} tokens; run "
                         f"tools/screen_needles.py")
    return found[-1]


class NeedleSet(NamedTuple):
    """The needle set a run is of, and the model it is run on."""

    model: str
    needle_seed: int
    haystack_tokens: int = needles.DEFAULT_HAYSTACK_TOKENS

    def holds(self, entry: dict) -> bool:
        """Whether a log entry is of this set and model."""
        return (entry["model"] == self.model
                and entry["config"]["needle_seed"] == self.needle_seed
                and entry["config"]["haystack_tokens"] == self.haystack_tokens)

    def screening(self, entries: Sequence[dict]) -> dict:
        """The set's last complete needle-screening entry (#148)."""
        return last_screening(entries, self.model, self.needle_seed, self.haystack_tokens)

    def config(self) -> dict:
        """What a log entry's config names of the set, but its model."""
        return {"needle_seed": self.needle_seed, "haystack_tokens": self.haystack_tokens}

    def logged_at(self, kind: str, commit: str,
                  log: str | Path = benchlog.DEFAULT_LOG) -> list[dict]:
        """The `kind` entries of the set logged at the code of `commit`
        (benchlog.same_code), though the log may differ, so that a run of
        hours stopped part way is taken up again; none before the log
        exists."""
        if not Path(log).exists():
            return []
        same_code = benchlog.same_code_as(commit, log)
        return [e for e in benchlog.read(log)
                if e["kind"] == kind and self.holds(e) and same_code(e["git_commit"])]


def screened(model: str, needle_seed: int, haystack_tokens: int,
             log: str | Path = benchlog.DEFAULT_LOG) -> tuple[list[str], int]:
    """The screened set's sample ids, for `model` and the set of
    `needle_seed` and `haystack_tokens`, from the last complete
    needle-screening entry logged for them, and the samples a depth it was
    drawn with, which gives those ids their samples."""
    entry = last_screening(benchlog.read(log), model, needle_seed, haystack_tokens)
    return entry["results"]["kept"], entry["config"]["draws"]
