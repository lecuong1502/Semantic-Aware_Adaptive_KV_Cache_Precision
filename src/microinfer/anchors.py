"""The anchor-fact dialogue: RQ3's third task (#154).

ADR-0017 states a fact in an early turn of a conversation, has other turns
follow, and asks for the fact back in a later one: the case eviction fails
and reallocation is built for. A **dialogue** is TURNS turns, each a passage
of WikiText-2's test split, PASSAGE_TOKENS tokens, and the request
"Summarize the passage above in one sentence.", answered in up to
REPLY_TOKENS decoded tokens, whose attention gives the scorer its scores.
One turn, the **fact's**, opens its message with a line of its own, "Note
for later: The access code for the Juniper archive is XQT-4827.", a needle
(needles.needle_text) the model cannot guess. After the last turn the
controlled plan of a condition is applied, question-blind, and the
**question** asks for the code, as the needle's does, answered in up to
rq3.ANSWER_TOKENS; it is right if, and only if, it holds the code.

A dialogue's **gap** is the turns between the fact's and the question,
0 to TURNS - 1: the fact's turn is TURNS - 1 - gap. The set holds
SAMPLES_PER_GAP dialogues a gap, 50 in all, from a seed. Each gap draws from
a generator of its own, seeded with the seed and the gap: an order of the
articles, then for each dialogue in turn the archive's name and the code,
drawn again while another dialogue holds it. Dialogue k's passages are the
consecutive lines from the k-th article of that order, each turn's cut to
PASSAGE_TOKENS at its last line and the next starting at the line after:
every dialogue's prompts are alike in length, but for the fact's line and
its turn, and drawing more a gap leaves the first ones as they are, for
screening.

A turn's prompt is built from pieces tokenized apart, as the needle's turn
1 is: the template's, the fact's line, each passage line, the request. A
turn after one cut short at its tokens opens with <|im_end|>
(needles.Sample.turn_2). The fact's positions are known once the turns
before it are run, their replies being decoded: a run gives them, and the
pages that hold them, with what those pages hold at the plan.

FP16 screens the set (rq3.screen by gap): the first SAMPLES_PER_GAP
dialogues of a gap that FP16 answers rightly are kept. Every source
condition holds the turns alike until the plan, so they run once and each
condition starts from their snapshot (rq3.from_one_opening). The summary
gives each condition's recall, overall and by gap, with Wilson and
bootstrap intervals, and at each level the semantic source paired against
uniform and random by McNemar's exact test, as the needle's summary does.
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Iterator, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from . import benchlog, controller, needles, rq3, rq3_summary

TURNS = 10
PASSAGE_TOKENS = 720
REPLY_TOKENS = 32
SAMPLES_PER_GAP = 5
#: The anchor-fact set's seed, fixed for Milestone 3 (#154).
ANCHOR_SEED = 154
KIND = "rq3-anchor"
SCREENING_KIND = "anchor-screening"
EXPERIMENT = "question-blind"
#: The bootstrap's seed: the summary repeats exactly.
SEED = 154

REQUEST = "Summarize the passage above in one sentence."
#: A turn's opening after one that stopped at its end of turn; the first
#: opens with needles.TURN_1_OPENING.
TURN_OPENING = "\n<|im_start|>user\n"
TURN_CLOSING = f"\n\n{REQUEST}<|im_end|>\n<|im_start|>assistant\n"


def note(fact: str) -> str:
    return f"Note for later: {fact}"


def gaps(turns: int = TURNS) -> tuple[int, ...]:
    return tuple(range(turns))


@dataclass(frozen=True, eq=False)
class Dialogue:
    """One dialogue: its fact, its turns' prompts and its question."""

    sample_id: str
    gap: int
    fact_turn: int
    name: str
    code: str
    fact: str
    question: str
    #: Each turn's prompt, after a turn that stopped at its end of turn; the
    #: first from position 0. A turn after one cut short opens with
    #: <|im_end|> before it.
    turns: tuple[np.ndarray, ...]
    question_turn: np.ndarray
    #: The fact's line in its turn's prompt, [start, end).
    fact_offsets: tuple[int, int]
    passage_tokens: int
    #: The WikiText article the passages start at, by index and title.
    first_article: int
    first_title: str
    page_tokens: int

    def as_dict(self) -> dict:
        return {"sample_id": self.sample_id, "gap": self.gap, "fact_turn": self.fact_turn,
                "name": self.name, "code": self.code, "fact": self.fact,
                "question": self.question, "turns": [t.tolist() for t in self.turns],
                "question_turn": self.question_turn.tolist(),
                "fact_offsets": list(self.fact_offsets),
                "passage_tokens": self.passage_tokens, "first_article": self.first_article,
                "first_title": self.first_title, "page_tokens": self.page_tokens}


def dumps(dialogues: list[Dialogue]) -> bytes:
    """The set as JSON lines, one dialogue each, keys in a fixed order."""
    return b"".join(json.dumps(d.as_dict(), ensure_ascii=False).encode() + b"\n"
                    for d in dialogues)


def dialogue_set(text: str, tokenizer, *, seed: int = ANCHOR_SEED,
                 samples_per_gap: int = SAMPLES_PER_GAP, turns: int = TURNS,
                 passage_tokens: int = PASSAGE_TOKENS,
                 page_tokens: int | None = None) -> list[Dialogue]:
    """The dialogues, gap by gap, from `text`, WikiText-2's test split as
    tools/perplexity.py reads it, and `tokenizer`, the checkpoint's, as the
    module describes."""
    if page_tokens is None:
        from . import _microinfer
        page_tokens = _microinfer.device.page_tokens

    def ids(s: str) -> np.ndarray:
        return np.asarray(tokenizer.encode(s).ids, dtype=np.int32)

    haystacks = needles.Haystacks(text, tokenizer, passage_tokens)
    first_opening, opening = ids(needles.TURN_1_OPENING), ids(TURN_OPENING)
    closing = ids(TURN_CLOSING)
    dialogues, codes = [], set()
    for gap in gaps(turns):
        rng = np.random.default_rng([seed, gap])
        order = rng.permutation(len(haystacks.found))
        fact_turn = turns - 1 - gap
        for k in range(samples_per_gap):
            article = int(order[k % len(order)])
            name = needles.NAMES[int(rng.integers(len(needles.NAMES)))]
            code = needles.draw_code(rng)
            while code in codes:
                code = needles.draw_code(rng)
            codes.add(code)
            fact, question = needles.needle_text(name, code), needles.question_text(name)
            planted = ids(needles.piece(note(fact)))
            prompts = []
            for i, passage in enumerate(_passages(haystacks, article, turns, passage_tokens)):
                head = first_opening if i == 0 else opening
                prompts.append(np.concatenate(
                    [head, *([planted] if i == fact_turn else []), *passage, closing]))
            start = len(first_opening if fact_turn == 0 else opening)
            dialogues.append(Dialogue(
                sample_id=f"g{gap}-{k}", gap=gap, fact_turn=fact_turn, name=name, code=code,
                fact=fact, question=question, turns=tuple(prompts),
                question_turn=ids(needles.TURN_2_AFTER_EOS.format(question=question)),
                fact_offsets=(start, start + len(planted)), passage_tokens=passage_tokens,
                first_article=article, first_title=haystacks.found[article][0],
                page_tokens=page_tokens))
    return dialogues


def _passages(haystacks: needles.Haystacks, article: int, turns: int,
              tokens: int) -> list[list[np.ndarray]]:
    """`turns` passages of `tokens` each, of consecutive lines from
    `article`'s first, wrapping round: each cut at its last line, and the
    next from the line after."""
    line, out = int(haystacks.first_line[article]), []
    for _ in range(turns):
        pieces, held = [], 0
        while held < tokens:
            taken = haystacks.line_ids[line % len(haystacks.line_ids)][:tokens - held]
            pieces.append(taken)
            held += len(taken)
            line += 1
        out.append(pieces)
    return out


# -- through the engine --------------------------------------------------------------------

@dataclass(frozen=True)
class _Opened:
    """What a run holds before its plan: each turn's reply, the fact's
    positions, and the question's prompt."""

    replies: tuple[np.ndarray, ...]
    fact_positions: tuple[int, int]
    question: np.ndarray


def _open(engine, chat, dialogue: Dialogue) -> _Opened:
    """Every turn, each after the last one's reply. A prompt starts where
    the last one's tokens and reply end: the reply's last token, not yet
    fed, takes the position before it."""
    end_of_turn = engine.tokenizer.token_to_id(rq3.END_OF_TURN)
    replies, position, cut, fact = [], 0, False, None
    for i, body in enumerate(dialogue.turns):
        prompt = np.concatenate([[end_of_turn], body]).astype(np.int32) if cut else body
        if i == dialogue.fact_turn:
            first = position + len(prompt) - len(body)
            fact = (first + dialogue.fact_offsets[0], first + dialogue.fact_offsets[1])
        reply = chat.turn(prompt, REPLY_TOKENS)
        position += len(prompt) + len(reply)
        cut = int(reply[-1]) != end_of_turn
        replies.append(reply)
    question = dialogue.question_turn
    if cut:
        question = np.concatenate([[end_of_turn], question]).astype(np.int32)
    return _Opened(tuple(replies), fact, question)


def _close(engine, chat, dialogue: Dialogue, cond: rq3.Condition, seed: int,
           opened: _Opened) -> dict:
    plan_positions = chat.positions
    record = None
    if cond.source is not None:
        record = chat.plan(cond.share, source=cond.source, seed=seed)
    bits = chat.effective_bits()
    tiers = chat.tiers()
    answer = chat.turn(opened.question, rq3.ANSWER_TOKENS)
    start, end = opened.fact_positions
    pages = (start // dialogue.page_tokens, (end - 1) // dialogue.page_tokens)
    fact_tiers = Counter(controller.TIERS[t] for t in tiers[:, pages[0]:pages[1] + 1].ravel())
    replies = [engine.decode(r) for r in opened.replies]
    end_of_turn = engine.tokenizer.token_to_id(rq3.END_OF_TURN)
    text = engine.decode(answer)
    return {
        "answer": text, "correct": needles.correct(text, dialogue.code),
        "answer_tokens": answer.tolist(), "replies": replies,
        "reply_tokens": [r.tolist() for r in opened.replies],
        "replies_stopped": [int(r[-1]) == end_of_turn for r in opened.replies],
        # a reply that repeats the code holds it at a second place
        "code_echoed": [i for i, r in enumerate(replies) if dialogue.code in r],
        "fact_positions": list(opened.fact_positions), "fact_pages": list(pages),
        "plan_positions": plan_positions, "plan": rq3.plan_summary(record),
        "random_seed": seed if cond.source == "random" else None,
        "effective_bits": bits["effective_bits"],
        "candidate_effective_bits": bits["candidate_effective_bits"],
        "tier_shares": rq3.tier_shares(tiers[:, :plan_positions // dialogue.page_tokens]),
        "fact_page_tiers": dict(sorted(fact_tiers.items())),
    }


def _check(engine) -> None:
    if engine.kv_cache != "paged" or engine.kv_halves != "both":
        raise ValueError("a run holds its cache on pages, both halves at a page's tier")


def run(engine, dialogue: Dialogue, cond: rq3.Condition, *, plan_seed: int = 0) -> dict:
    """One dialogue through its turns, the plan and the question under one
    condition: what a log entry's results hold. The effective bits and the
    tiers are the cache's at the plan."""
    _check(engine)
    with rq3.set_for(engine, cond), engine.conversation() as chat:
        return _close(engine, chat, dialogue, cond,
                      rq3.sample_seed(dialogue.sample_id, plan_seed), _open(engine, chat, dialogue))


def runs(engine, dialogue: Dialogue, conds: Sequence[rq3.Condition], *,
         plan_seed: int = 0) -> Iterator[tuple[rq3.Condition, dict]]:
    """Each of `conds` on one dialogue, with what `run` gives it, those that
    hold the turns alike from one run of them (rq3.from_one_opening)."""
    _check(engine)
    seed = rq3.sample_seed(dialogue.sample_id, plan_seed)
    yield from rq3.from_one_opening(
        engine, conds, lambda chat: _open(engine, chat, dialogue),
        lambda chat, cond, opened: _close(engine, chat, dialogue, cond, seed, opened),
        lambda cond: run(engine, dialogue, cond, plan_seed=plan_seed))


def config(dialogue: Dialogue, cond: rq3.Condition, **more) -> dict:
    """What a log entry's config names of a run."""
    return {"sample_id": dialogue.sample_id, "gap": dialogue.gap,
            "fact_turn": dialogue.fact_turn, "code": dialogue.code,
            "turns": len(dialogue.turns), "passage_tokens": dialogue.passage_tokens,
            "reply_tokens": REPLY_TOKENS, "answer_tokens": rq3.ANSWER_TOKENS,
            "condition": asdict(cond), "experiment": EXPERIMENT, **more}


def _matches(e: dict, model: str, anchor_seed: int, turns: int, passage_tokens: int) -> bool:
    c = e["config"]
    return (e["model"] == model and c["anchor_seed"] == anchor_seed and c["turns"] == turns
            and c["passage_tokens"] == passage_tokens)


def logged(model: str, anchor_seed: int, commit: str, log: str | Path = benchlog.DEFAULT_LOG,
           *, screening: bool = False, plan_seed: int = 0, turns: int = TURNS,
           passage_tokens: int = PASSAGE_TOKENS) -> dict[tuple[str, str], bool]:
    """The runs already logged, by (sample id, condition), each with whether
    it was answered rightly: those at the code of `commit`
    (benchlog.same_code), for this model, set and plan seed, screening's
    when `screening` and not otherwise, as rq3.logged counts a needle run."""
    if not Path(log).exists():
        return {}
    held = benchlog.same_code_as(commit, log)
    return {(e["config"]["sample_id"], e["config"]["condition"]["name"]): e["results"]["correct"]
            for e in benchlog.read(log)
            if e["kind"] == KIND and _matches(e, model, anchor_seed, turns, passage_tokens)
            and bool(e["config"].get("screening")) == screening
            and e["config"]["plan_seed"] == plan_seed and held(e["git_commit"])}


def last_screening(entries: Sequence[dict], model: str, anchor_seed: int, *,
                   turns: int = TURNS, passage_tokens: int = PASSAGE_TOKENS) -> dict:
    """The last complete anchor-screening entry for `model` and the set; a
    ValueError if there is none."""
    found = [e for e in entries if e["kind"] == SCREENING_KIND
             and _matches(e, model, anchor_seed, turns, passage_tokens)
             and e["results"]["complete"]]
    if not found:
        raise ValueError(f"no complete {SCREENING_KIND} logged for {model} at seed "
                         f"{anchor_seed}; run tools/anchors.py --screen")
    return found[-1]


def gap_name(gap: int) -> str:
    return f"{gap} turns"


def summarise(entries: Sequence[dict], model: str, anchor_seed: int, *, plan_seed: int = 0,
              turns: int = TURNS, passage_tokens: int = PASSAGE_TOKENS,
              conditions: Sequence[str] = ("fp16", *rq3.SOURCE_CONDITIONS)) -> dict:
    """The anchor-fact summary of the screened set's runs in `entries`, the
    latest of each dialogue under each of `conditions`, fp16's screening's,
    as the module describes; a ValueError naming what is missing."""
    screening = last_screening(entries, model, anchor_seed, turns=turns,
                               passage_tokens=passage_tokens)
    kept = screening["results"]["kept"]
    latest = {}
    for e in entries:
        c = e["config"]
        if (e["kind"] == KIND and _matches(e, model, anchor_seed, turns, passage_tokens)
                and c["plan_seed"] == plan_seed
                and bool(c.get("screening")) == (c["condition"]["name"] == "fp16")):
            latest[(c["sample_id"], c["condition"]["name"])] = e
    missing = [(s, n) for n in conditions for s in kept if (s, n) not in latest]
    if missing:
        raise ValueError(f"no run of {len(missing)} dialogues' conditions, such as "
                         f"{missing[:3]}; run tools/anchors.py for them")
    rng = np.random.default_rng(SEED)
    runs_of = {n: [latest[(s, n)] for s in kept] for n in conditions}
    right = {n: np.array([e["results"]["correct"] for e in runs_of[n]]) for n in conditions}
    gap_of = np.array([latest[(s, conditions[0])]["config"]["gap"] for s in kept])
    recall = {}
    for n in conditions:
        r, entries_n = right[n], runs_of[n]
        by_gap = {}
        for gap in gaps(turns):
            at = r[gap_of == gap]
            if len(at):
                by_gap[gap_name(gap)] = {"correct": int(at.sum()), "n": len(at),
                                         "recall": float(at.mean()),
                                         "wilson_95": list(rq3_summary.wilson(int(at.sum()),
                                                                              len(at)))}
        tiers = Counter()
        for e in entries_n:
            tiers.update(e["results"]["fact_page_tiers"])
        held = sum(tiers.values())
        recall[n] = {
            "correct": int(r.sum()), "n": len(r), "recall": float(r.mean()),
            "bootstrap_95": [float(v) for v in rq3_summary.bootstrap(r[None, :], rng)[0]],
            "wilson_95": list(rq3_summary.wilson(int(r.sum()), len(r))), "by_gap": by_gap,
            "effective_bits": float(np.mean([e["results"]["effective_bits"]
                                             for e in entries_n])),
            "fact_page_tiers": {t: k / held for t, k in sorted(tiers.items())},
            "code_echoed": sum(1 for e in entries_n if e["results"]["code_echoed"])}
    paired = {}
    for share in rq3_summary.LEVELS:
        names = {c.source: c.name for c in rq3.CONDITIONS if c.share == share}
        if names["semantic"] not in conditions:
            continue
        paired[rq3_summary.level_name(share)] = {
            f"semantic-{other}": rq3_summary.paired(right[names["semantic"]],
                                                    right[names[other]], rng)
            for other in rq3_summary.OTHER_SOURCES if names[other] in conditions}
    return {"model": model, "anchor_seed": anchor_seed, "plan_seed": plan_seed,
            "turns": turns, "passage_tokens": passage_tokens, "dialogues": len(kept),
            "bootstrap": {"resamples": rq3_summary.RESAMPLES, "seed": SEED},
            "recall": recall, "paired": paired, "screening": screening["sha256"],
            "commits": sorted({e["git_commit"] for n in conditions for e in runs_of[n]}),
            "entries": [e["sha256"] for n in conditions for e in runs_of[n]]}
