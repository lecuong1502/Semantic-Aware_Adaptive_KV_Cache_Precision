"""Recovery after pressure passes, on the needle set: RQ3's fourth task (#155).

ADR-0017: after the compressed turn, pressure passes, upgrades restore the
pages from their shadows, and a third turn asks again. A run is a needle
sample's question-blind run (rq3.run), turn 1, the plan and turn 2, the
question, and after turn 2's answer:
- **the restore**, Conversation.restore: every page that holds a shadow
  back at FP16, by the upgrade policy's plan, from the condition's source;
- **turn 3**, the same question again, answered in up to rq3.ANSWER_TOKENS.
  It opens as turn 2 does after turn 1: with <|im_end|> after an answer
  cut short at its tokens, or stopped at <|endoftext|> (needles.Sample).

The restore gives back each page's FP16 bytes, so the cache holds what an
FP16 run's held, but for turn 2's answer: decoded over the compressed
cache, it is held at FP16 like any new page, and turn 3 reads it. FP16, with
no plan, runs the same three turns, its turn 3 the baseline. A run logs
turn 2 as rq3 does, and beside it the restore, turn 3's answer, whether it
is right, and whether it is turn 2's word for word.

The conditions are FP16 and every score source at each level; a static
tier, born quantised, holds no shadow to restore from. The source
conditions hold turn 1 alike and run it once (rq3.from_one_opening).

The summary gives each condition's accuracy at turn 2 and at turn 3, with
Wilson and bootstrap intervals; the samples turn 3 answered and turn 2 did
not, those it lost, and McNemar's exact test on them; and at each level
and turn the semantic source paired against uniform and random, as the
needle's summary pairs them (rq3_summary).
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import NamedTuple

import numpy as np

from . import benchlog, needles, rq3, rq3_summary

KIND = "rq3-recovery"
SUMMARY_KIND = "recovery-summary"
EXPERIMENT = "question-blind"
CONDITIONS = ("fp16", *rq3.SOURCE_CONDITIONS)
TURNS = ("turn 2", "turn 3")
#: The summary's bootstrap seed.
SEED = 155


def turn_3(engine, chat, sample: needles.Sample, cond: rq3.Condition, seed: int,
           results: dict) -> dict:
    """rq3's `then`: the restore, for a condition that planned, and turn 3,
    the question asked again; what a run's results hold of them."""
    record = None
    if cond.source is not None:
        record = chat.restore(source=cond.source, seed=seed)
    bits = chat.effective_bits()
    end_of_turn = engine.tokenizer.token_to_id(rq3.END_OF_TURN)
    stopped = results["answer_tokens"][-1] == end_of_turn
    answer = chat.turn(sample.turn_2_after_eos if stopped else sample.turn_2,
                       rq3.ANSWER_TOKENS)
    text = engine.decode(answer)
    return {"restore": restore_summary(record),
            "turn_3_effective_bits": bits["effective_bits"],
            "turn_3_answer": text, "turn_3_correct": needles.correct(text, sample.code),
            "turn_3_answer_tokens": answer.tolist(),
            "turn_3_repeats_turn_2": text == results["answer"]}


def restore_summary(record) -> dict | None:
    """What a log entry's results hold of a restore's record: its upgrades,
    how many were applied, in how many steps and seconds, how it ended, and
    what it restored; None for no restore."""
    if record is None:
        return None
    controlled = record.controlled
    # The first batch's seconds count the plan's making too.
    return {"moves": len(record.plan), "applied": record.applied,
            "batches": len(record.batches), "ended": record.ended,
            "seconds": sum(b.seconds for b in record.batches),
            "restored": controlled["restored"], "restored_bytes": controlled["restored_bytes"]}


def run(engine, sample: needles.Sample, cond: rq3.Condition, *, plan_seed: int = 0) -> dict:
    """One sample through its three turns under `cond`: rq3.run's results,
    and turn_3's beside them."""
    return rq3.run(engine, sample, cond, EXPERIMENT, plan_seed=plan_seed, then=turn_3)


def runs(engine, sample: needles.Sample, conds: Sequence[rq3.Condition], *,
         plan_seed: int = 0) -> Iterator[tuple[rq3.Condition, dict]]:
    """Each of `conds` on one sample, as `run` gives it, the source
    conditions from one turn 1 (rq3.runs)."""
    yield from rq3.runs(engine, sample, conds, EXPERIMENT, plan_seed=plan_seed, then=turn_3)


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
        return rq3.last_screening(entries, self.model, self.needle_seed, self.haystack_tokens)

    def config(self) -> dict:
        """What a log entry's config names of the set, but its model."""
        return {"needle_seed": self.needle_seed, "haystack_tokens": self.haystack_tokens}


def logged(of: NeedleSet, commit: str, log: str | Path = benchlog.DEFAULT_LOG, *,
           plan_seed: int = 0) -> dict[tuple[str, str], bool]:
    """The runs already logged, by (sample id, condition), each with whether
    turn 3 was right: those of the set at the code of `commit`
    (benchlog.same_code) and this plan seed, as rq3.logged counts a needle
    run."""
    if not Path(log).exists():
        return {}
    same_code = benchlog.same_code_as(commit, log)
    return {(e["config"]["sample_id"], e["config"]["condition"]["name"]):
            e["results"]["turn_3_correct"]
            for e in _runs(benchlog.read(log), of, plan_seed) if same_code(e["git_commit"])}


def _runs(entries: Sequence[dict], of: NeedleSet, plan_seed: int) -> Iterator[dict]:
    """The recovery runs in `entries` of the set at `plan_seed`."""
    return (e for e in entries
            if e["kind"] == KIND and of.holds(e) and e["config"]["plan_seed"] == plan_seed)


def _accuracy(right: np.ndarray, rng: np.random.Generator) -> dict:
    return {**rq3_summary.proportion(right, rng), "accuracy": float(right.mean())}


def summarise(entries: Sequence[dict], of: NeedleSet, *, plan_seed: int = 0,
              conditions: Sequence[str] = CONDITIONS) -> dict:
    """The recovery summary of the screened set's runs in `entries`, the
    latest of each sample under each of `conditions` at `plan_seed`, as the
    module describes. A ValueError naming what is missing."""
    screening = of.screening(entries)
    kept = screening["results"]["kept"]
    latest = {(e["config"]["sample_id"], e["config"]["condition"]["name"]): e
              for e in _runs(entries, of, plan_seed)}  # the latest of each stands
    missing = [(s, name) for name in conditions for s in kept if (s, name) not in latest]
    if missing:
        raise ValueError(f"no run of {len(missing)} samples' conditions, such as "
                         f"{missing[:3]}; run tools/recovery.py for them")
    rng = np.random.default_rng(SEED)
    runs_of = {name: [latest[(s, name)] for s in kept] for name in conditions}
    right = {(name, turn): np.array([e["results"][key] for e in runs_of[name]])
             for name in conditions
             for turn, key in zip(TURNS, ("correct", "turn_3_correct"))}
    accuracy = {}
    for name in conditions:
        before, after = right[(name, "turn 2")], right[(name, "turn 3")]
        recovered, lost = int((~before & after).sum()), int((before & ~after).sum())
        results = [e["results"] for e in runs_of[name]]
        restores = [r["restore"] for r in results if r["restore"] is not None]
        accuracy[name] = {
            **{turn: _accuracy(right[(name, turn)], rng) for turn in TURNS},
            "recovered": recovered, "lost": lost,
            "mcnemar_p": rq3_summary.mcnemar(recovered, lost),
            "turn_3_repeats_turn_2": sum(r["turn_3_repeats_turn_2"] for r in results),
            "effective_bits": float(np.mean([r["effective_bits"] for r in results])),
            "turn_3_effective_bits": float(np.mean([r["turn_3_effective_bits"]
                                                    for r in results])),
            "restore": None if not restores else {
                "every_page_restored": all(r["restored"] for r in restores),
                "seconds_mean": float(np.mean([r["seconds"] for r in restores])),
                "seconds_max": float(max(r["seconds"] for r in restores)),
                "moves_mean": float(np.mean([r["moves"] for r in restores]))}}
    pairs = {turn: rq3_summary.paired_at_levels(
        {name: right[(name, turn)] for name in conditions}, rng) for turn in TURNS}
    return {"model": of.model, **of.config(), "plan_seed": plan_seed, "samples": len(kept),
            "bootstrap": {"resamples": rq3_summary.RESAMPLES, "seed": SEED},
            "accuracy": accuracy, "paired": pairs, "screening": screening["sha256"],
            "commits": sorted({e["git_commit"] for name in conditions for e in runs_of[name]}),
            "entries": [e["sha256"] for name in conditions for e in runs_of[name]]}
