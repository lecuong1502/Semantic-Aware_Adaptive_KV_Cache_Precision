"""RULER's runs pooled into one summary entry (#208, ADR-0017's amendment).

At one length, for each task screened there, its screened set is the last
complete "ruler-screening" entry's, and each condition's runs the latest
"rq3-ruler" run of each screened sample, FP16's its screening runs, on the
data named, when it is:

- **accuracy**: the samples answered, every expected string found, with a
  Wilson 95% interval; and the mean score, RULER's string_match_all, with
  its bootstrap interval;
- **the ceiling**: a task is below it if static INT2 answers at most 80% of
  its screened set (`below_ceiling`); only such a task enters the test;
- **the closing test**: semantic against uniform at 80%, McNemar's exact
  test, two-sided, on the samples of the tasks below their ceiling only
  one of the two answered; semantic beats uniform if p < 0.05 and it
  answered more. No answer while a screened task lacks static INT2's runs,
  or a task below its ceiling either source's, or no task is below it;
- **descriptive**: each task's semantic source paired against uniform and
  random at each level, by answered samples (rq3_summary.paired) and by
  the mean score's paired difference, with its bootstrap interval.
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
CEILING_CONDITION = "static-int2"
CLOSING_LEVEL = rq3_summary.level_name(rq3_summary.CLOSING_SHARE)
CLOSING = ("semantic-80", "uniform-80")
SEED = 208
#: The conditions a screened set is run under, B2 to B8 at both levels, in
#: their order: every one of ADR-0017's but FP16, which screening ran, and
#: eviction at each level.
CONDITIONS = tuple([c.name for c in rq3.CONDITIONS if c.name != "fp16"]
                   + list(rq3.EVICTION_CONDITIONS))


def below_ceiling(int2_answered: int, screened: int) -> bool:
    """Whether static INT2's answers leave a task below its ceiling."""
    return int2_answered <= CEILING * screened


def screenings(entries: Sequence[dict], model: str, seed: int,
               haystack_tokens: int) -> dict[str, dict]:
    """Each task's last complete screening entry at this length."""
    out: dict[str, dict] = {}
    for e in entries:
        c = e["config"]
        if (e["kind"] == SCREENING_KIND and e["model"] == model and c["ruler_seed"] == seed
                and c["haystack_tokens"] == haystack_tokens and e["results"]["complete"]):
            out[c["task"]] = e
    return out


def screened_sets(entries: Sequence[dict], model: str, seed: int,
                  haystack_tokens: int) -> dict[str, list[str]]:
    """Each task's screened sample ids at this length, from its last
    complete screening."""
    return {task: e["results"]["kept"]
            for task, e in screenings(entries, model, seed, haystack_tokens).items()}


def latest_runs(entries: Sequence[dict], model: str, seed: int, haystack_tokens: int,
                plan_seed: int = 0, data: dict | None = None) -> dict[tuple[str, str], dict]:
    """The latest run of each (sample id, condition), FP16's its screening's,
    on `data` if it is given."""
    out: dict[tuple[str, str], dict] = {}
    for e in entries:
        c = e["config"] if e["kind"] == RUN_KIND else None
        if (c is None or e["model"] != model or c["ruler_seed"] != seed
                or c["haystack_tokens"] != haystack_tokens or c["plan_seed"] != plan_seed
                or (data is not None and c.get("data") != data)):
            continue
        name = c["condition"]["name"]
        if c["screening"] != (name == "fp16"):
            continue
        out[(c["sample_id"], name)] = e
    return out


def _answered(runs: list[dict]) -> np.ndarray:
    return np.array([e["results"]["correct"] for e in runs], dtype=bool)


def _scores(runs: list[dict]) -> np.ndarray:
    return np.array([e["results"]["score"] for e in runs], dtype=float)


def _interval(values: np.ndarray, rng: np.random.Generator) -> list[float]:
    return [float(v) for v in rq3_summary.bootstrap(values[None, :], rng)[0]]


def _cell(runs: list[dict], rng: np.random.Generator) -> dict:
    answered, scores = _answered(runs), _scores(runs)
    return {"answered": int(answered.sum()), "runs": len(runs),
            "accuracy": float(answered.mean()),
            "wilson_95": list(rq3_summary.wilson(int(answered.sum()), len(runs))),
            "score_mean": float(scores.mean()), "score_bootstrap_95": _interval(scores, rng),
            "entries": [e["sha256"] for e in runs]}


def _score_paired(runs: dict[str, list[dict]], rng: np.random.Generator) -> dict:
    """At each level, the semantic source's mean score less uniform's and
    random's, sample by sample, for the conditions `runs` holds."""
    out: dict[str, dict] = {}
    for share in rq3_summary.LEVELS:
        semantic = runs.get(rq3.source_condition("semantic", share).name)
        for other in rq3_summary.OTHER_SOURCES:
            against = runs.get(rq3.source_condition(other, share).name)
            if semantic is None or against is None:
                continue
            difference = _scores(semantic) - _scores(against)
            out.setdefault(rq3_summary.level_name(share), {})[f"semantic-{other}"] = {
                "difference": float(difference.mean()),
                "difference_bootstrap_95": _interval(difference, rng)}
    return out


def summarise(entries: Sequence[dict], model: str, seed: int, haystack_tokens: int, *,
              plan_seed: int = 0, data: dict | None = None) -> dict:
    """The summary at one length, as the module describes."""
    rng = np.random.default_rng(rq3_summary.SEED)
    sets = screened_sets(entries, model, seed, haystack_tokens)
    latest = latest_runs(entries, model, seed, haystack_tokens, plan_seed, data)
    tasks, closing_runs, missing = {}, {name: [] for name in CLOSING}, {}
    for task, kept in sets.items():
        names = sorted({name for (sid, name) in latest if sid in kept})
        complete = {n: [latest[(sid, n)] for sid in kept] for n in names
                    if all((sid, n) in latest for sid in kept)}
        int2 = complete.get(CEILING_CONDITION)
        below = None if int2 is None else below_ceiling(int(_answered(int2).sum()), len(kept))
        tasks[task] = {
            "screened": len(kept), "below_ceiling": below,
            "by_condition": {n: _cell(runs, rng) for n, runs in complete.items()},
            "incomplete": [n for n in names if n not in complete],
            "paired": rq3_summary.paired_at_levels(
                {n: _answered(runs) for n, runs in complete.items()}, rng),
            "score_paired": _score_paired(complete, rng)}
        lacking = ([CEILING_CONDITION] if below is None else
                   [n for n in CLOSING if n not in complete] if below else [])
        if lacking:
            missing[task] = lacking
        elif below:
            for n in CLOSING:
                closing_runs[n].extend(complete[n])
    entered = [t for t, row in tasks.items() if row["below_ceiling"]]
    return {"model": model, "ruler_seed": seed, "haystack_tokens": haystack_tokens,
            "plan_seed": plan_seed, "data": data, "ceiling": CEILING, "tasks": tasks,
            "closing": _closing(entered, closing_runs, missing, haystack_tokens, rng)}


def _closing(entered: list[str], runs: dict[str, list[dict]], missing: dict,
             haystack_tokens: int, rng: np.random.Generator) -> dict:
    out = {"question": f"Does the semantic source answer more of RULER's samples than uniform "
                       f"at {CLOSING_LEVEL} of the FP16 bytes outside the recency floor "
                       f"reclaimed, {rq3_summary.CLOSING_EXPERIMENT}, over the tasks below "
                       f"their ceiling, at haystacks of {haystack_tokens} tokens?",
           "tasks": entered, "conditions": list(CLOSING), "missing": missing,
           "significance": rq3_summary.SIGNIFICANCE, "semantic_beats_uniform": None,
           "answer": None}
    if not entered or missing:
        return out
    semantic, uniform = (_answered(runs[n]) for n in CLOSING)
    pair = rq3_summary.paired(semantic, uniform, rng, named=("semantic", "uniform"))
    difference = _scores(runs[CLOSING[0]]) - _scores(runs[CLOSING[1]])
    n = len(semantic)
    beats = (pair["mcnemar_p"] < rq3_summary.SIGNIFICANCE
             and pair["semantic_only"] > pair["uniform_only"])
    out.update(pair, samples=n, semantic_answered=int(semantic.sum()),
               uniform_answered=int(uniform.sum()), score_difference=float(difference.mean()),
               score_difference_bootstrap_95=_interval(difference, rng),
               semantic_beats_uniform=beats)
    out["answer"] = (
        f"{'Yes' if beats else 'No'}: the semantic source answers {int(semantic.sum())}/{n} "
        f"and uniform {int(uniform.sum())}/{n} at {CLOSING_LEVEL}, over {', '.join(entered)}; "
        f"McNemar's exact p = {pair['mcnemar_p']:.3g} on {pair['semantic_only']} samples only "
        f"semantic answered and {pair['uniform_only']} only uniform did"
        + ("" if beats else f", so semantic does not beat uniform at "
                            f"{rq3_summary.SIGNIFICANCE}"))
    return out
