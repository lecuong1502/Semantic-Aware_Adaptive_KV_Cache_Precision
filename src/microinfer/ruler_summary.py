"""RULER's runs pooled into one summary entry (#208, ADR-0017's amendment).

At one length, for each task screened there, its screened set is the last
complete "ruler-screening" entry's, and each condition's runs the latest
"rq3-ruler" run of each screened sample, FP16's its screening runs:

- **accuracy**: the samples answered, every reference found, with a
  Wilson 95% interval; and the mean score, RULER's string_match_all, with
  its bootstrap interval;
- **the ceiling**: a task is below it if static INT2 answers at most 80% of
  its screened set (`below_ceiling`); only such a task enters the test;
- **the closing test**: semantic against uniform at 80%, McNemar's exact
  test, two-sided, on the samples of the tasks below their ceiling only
  one of the two answered; a yes if p < 0.05 and semantic answered more.
  None while a task below its ceiling lacks either's runs, or no task is
  below it;
- **descriptive**: each task's semantic source paired against uniform and
  random at each level (rq3_summary.paired), and the paired difference in
  mean score with its bootstrap interval.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from . import rq3, rq3_summary

SUMMARY_KIND = "ruler-summary"
RUN_KIND = "rq3-ruler"
SCREENING_KIND = "ruler-screening"
#: The share of its screened set static INT2 may answer, at most, for a
#: task to be below its ceiling.
CEILING = 0.8
CLOSING = ("semantic-80", "uniform-80")
SEED = 208


def below_ceiling(int2_answered: int, screened: int) -> bool:
    """Whether static INT2's answers leave a task below its ceiling."""
    return int2_answered <= CEILING * screened


def screened_sets(entries: Sequence[dict], model: str, seed: int,
                  haystack_tokens: int) -> dict[str, list[str]]:
    """Each task's screened sample ids at this length, from its last
    complete screening."""
    out: dict[str, list[str]] = {}
    for e in entries:
        c = e["config"]
        if (e["kind"] == SCREENING_KIND and e["model"] == model and c["ruler_seed"] == seed
                and c["haystack_tokens"] == haystack_tokens and e["results"]["complete"]):
            out[c["task"]] = e["results"]["kept"]
    return out


def latest_runs(entries: Sequence[dict], model: str, seed: int, haystack_tokens: int,
                plan_seed: int = 0) -> dict[tuple[str, str], dict]:
    """The latest run of each (sample id, condition), FP16's its screening's."""
    out: dict[tuple[str, str], dict] = {}
    for e in entries:
        c = e["config"] if e["kind"] == RUN_KIND else None
        if (c is None or e["model"] != model or c["ruler_seed"] != seed
                or c["haystack_tokens"] != haystack_tokens or c["plan_seed"] != plan_seed):
            continue
        name = c["condition"]["name"]
        if c["screening"] != (name == "fp16"):
            continue
        out[(c["sample_id"], name)] = e
    return out


def _cell(runs: list[dict], rng: np.random.Generator) -> dict:
    answered = np.array([e["results"]["correct"] for e in runs])
    scores = np.array([e["results"]["score"] for e in runs], dtype=float)
    return {"answered": int(answered.sum()), "runs": len(runs),
            "accuracy": float(answered.mean()),
            "wilson_95": list(rq3_summary.wilson(int(answered.sum()), len(runs))),
            "score_mean": float(scores.mean()),
            "score_bootstrap_95": [float(v) for v in rq3_summary.bootstrap(scores[None, :],
                                                                          rng)[0]],
            "entries": [e["sha256"] for e in runs]}


def summarise(entries: Sequence[dict], model: str, seed: int, haystack_tokens: int, *,
              plan_seed: int = 0) -> dict:
    """The summary at one length, as the module describes."""
    rng = np.random.default_rng(rq3_summary.SEED)
    sets = screened_sets(entries, model, seed, haystack_tokens)
    latest = latest_runs(entries, model, seed, haystack_tokens, plan_seed)
    tasks, paired_runs, missing = {}, {name: [] for name in CLOSING}, {}
    for task, kept in sets.items():
        names = sorted({name for (sid, name) in latest if sid in kept})
        complete = {n: [latest[(sid, n)] for sid in kept] for n in names
                    if all((sid, n) in latest for sid in kept)}
        int2 = complete.get("static-int2")
        below = None if int2 is None else below_ceiling(
            sum(e["results"]["correct"] for e in int2), len(kept))
        right = {n: np.array([e["results"]["correct"] for e in runs])
                 for n, runs in complete.items()}
        tasks[task] = {
            "screened": len(kept), "below_ceiling": below,
            "by_condition": {n: _cell(runs, rng) for n, runs in complete.items()},
            "incomplete": [n for n in names if n not in complete],
            "paired": rq3_summary.paired_at_levels(right, rng)}
        if below:
            lacking = [n for n in CLOSING if n not in complete]
            if lacking:
                missing[task] = lacking
            else:
                for n in CLOSING:
                    paired_runs[n].extend(complete[n])
    entered = [t for t, row in tasks.items() if row["below_ceiling"]]
    return {"model": model, "ruler_seed": seed, "haystack_tokens": haystack_tokens,
            "plan_seed": plan_seed, "ceiling": CEILING, "tasks": tasks,
            "closing": _closing(entered, paired_runs, missing, rng)}


def _closing(entered: list[str], runs: dict[str, list[dict]], missing: dict,
             rng: np.random.Generator) -> dict:
    out = {"tasks": entered, "conditions": list(CLOSING), "missing": missing,
           "significance": rq3_summary.SIGNIFICANCE, "answer": None}
    if not entered or missing:
        return out
    semantic, uniform = (np.array([e["results"]["correct"] for e in runs[n]]) for n in CLOSING)
    pair = rq3_summary.paired(semantic, uniform, rng, named=("semantic", "uniform"))
    scores = np.array([[e["results"]["score"] for e in runs[n]] for n in CLOSING], dtype=float)
    difference = scores[0] - scores[1]
    out.update(pair, samples=len(semantic), semantic_answered=int(semantic.sum()),
               uniform_answered=int(uniform.sum()),
               score_difference=float(difference.mean()),
               score_difference_bootstrap_95=[float(v) for v in rq3_summary.bootstrap(
                   difference[None, :], rng)[0]])
    out["answer"] = ("yes" if pair["mcnemar_p"] < rq3_summary.SIGNIFICANCE
                     and pair["semantic_only"] > pair["uniform_only"] else "no")
    return out


#: The conditions a screened set is run under, B2 to B8 at both levels, in
#: their order: every one of ADR-0017's but FP16, which screening ran, and
#: eviction at each level.
CONDITIONS = tuple([c.name for c in rq3.CONDITIONS if c.name != "fp16"]
                   + list(rq3.EVICTION_CONDITIONS))
