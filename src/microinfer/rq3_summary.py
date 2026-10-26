"""The RQ3 summary: the logged needle runs pooled into one entry (#152).

ADR-0017 answers RQ3 from the benchmark log alone. The runs pooled are the
screened set's (#148), for one model, needle seed, haystack length and plan
seed: the latest run of each sample in each experiment under each
condition, so that a repeat stands for the run before it. FP16's are
screening's runs, `screening` true in their config, and no other condition's
are; NO_PLAN, a check, is not pooled. Question-aware holds the source
conditions only. FP16 and the static tiers make no plan, so they decode the
tokens they decode question-blind: a turn's answer decoded in two parts is
the answer decoded whole (tests/test_rq3.py), and on d50-0 #147 logged FP16
answering token for token alike in both experiments.

For each experiment and condition:
- **Accuracy**, overall and at each depth, with a bootstrap 95% interval
  that resamples the samples, as ADR-0017 fixes. At the ceiling every
  resample is right and the interval is a point, so the Wilson score
  interval is given beside it, which stays open there; it was added here,
  after the runs were logged, for that reason.
- **Effective bits**, the mean over the samples, and the needle's pages'
  tiers, as shares of every (layer, page) of them.

For each experiment and level, the semantic source is paired by sample
against uniform and against random: the samples only one of them answered,
McNemar's exact test on them, two-sided, and the paired difference in
accuracy with its bootstrap interval.

**The closing question** is the semantic source against uniform at 80%,
question-blind, by McNemar's test, as ADR-0017 fixes. Its significance
level, SIGNIFICANCE, and the rule that semantic beats uniform only if it
answers more samples and the two-sided test rejects, which makes a one-sided
test at half the level, were set here, after the runs were logged: ADR-0017
names neither. The answer is reported whatever it is, with the fewest
discordant samples, all one way, that could have rejected.

The 32K subset (#156) is summarised apart, by summarise_long: its ten
samples confirm the direction, with no test claimed. So are ADR-0018's
quality ablations (#158), by summarise_ablations, each against the full
policy, and eviction at each level (#160), by summarise_evictions, against
the semantic source there (evictions, the anchor-fact dialogue's too).
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from math import comb, sqrt

import numpy as np

from . import needles, rq3

#: The closing test's significance level, two-sided (set in #152).
SIGNIFICANCE = 0.05
RESAMPLES = 10_000
#: The bootstrap's seed: the summary repeats exactly.
SEED = 152
Z_95 = 1.959964
CLOSING_EXPERIMENT = "question-blind"
CLOSING_SHARE = 0.8
LEVELS = rq3.LEVELS
OTHER_SOURCES = ("uniform", "random")
LONG_SUMMARY_KIND = "rq3-long-summary"
ABLATION_SUMMARY_KIND = "rq3-ablation-summary"
#: The needle's eviction summary and the anchor-fact dialogue's, the `task`
#: of its config telling them apart (#160).
EVICTION_SUMMARY_KIND = "rq3-eviction-summary"


def mcnemar(only_a: int, only_b: int) -> float:
    """McNemar's exact test, two-sided: the chance, were neither better, of
    discordant pairs split as unevenly as `only_a` against `only_b`, a
    binomial of their count at one half."""
    n = only_a + only_b
    if n == 0:
        return 1.0
    tail = sum(comb(n, i) for i in range(min(only_a, only_b) + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def fewest_to_reject(level: float = SIGNIFICANCE) -> int:
    """The fewest discordant samples, all one way, McNemar's exact test
    rejects at `level`."""
    n = 1
    while mcnemar(n, 0) >= level:
        n += 1
    return n


def bootstrap(outcomes: np.ndarray, rng: np.random.Generator,
              resamples: int = RESAMPLES) -> np.ndarray:
    """The 95% percentile interval of each row's mean, (rows, 2), the
    columns, samples, resampled with replacement."""
    n = outcomes.shape[1]
    picks = rng.integers(0, n, size=(resamples, n))
    means = outcomes[:, picks].mean(axis=2)  # (rows, resamples)
    return np.percentile(means, [2.5, 97.5], axis=1).T


def wilson(correct: int, n: int, z: float = Z_95) -> tuple[float, float]:
    """The Wilson score interval of a proportion."""
    p = correct / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    # at 0 or n correct, the bound beside it is the proportion itself
    return (0.0 if correct == 0 else centre - half), (1.0 if correct == n else centre + half)


def level_name(share: float) -> str:
    return f"{round(share * 100)}%"


def _depth_name(depth: float) -> str:
    return f"{round(depth * 100)}%"


def _source_condition(source: str, share: float) -> str:
    return rq3.source_condition(source, share).name


#: The 32K subset's conditions, question-blind (#156, ADR-0017): FP16, and
#: the semantic and uniform sources at the closing question's level.
LONG_CONDITIONS = ("fp16", *(_source_condition(source, CLOSING_SHARE)
                             for source in ("semantic", "uniform")))


def _conditions(experiment: str) -> list[str]:
    if experiment == "question-aware":
        return list(rq3.SOURCE_CONDITIONS)
    return [c.name for c in rq3.CONDITIONS]


def _pool(entries: Sequence[dict], model: str, needle_seed: int, haystack_tokens: int,
          plan_seed: int, kept: list[str], *,
          screened_fp16: bool = True) -> dict[tuple[str, str, str], dict]:
    """The latest pooled run of each (sample, experiment, condition): FP16's
    screening's, if `screened_fp16`, and no other condition's."""
    latest = {}
    for e in entries:
        if e["kind"] != "rq3-needle" or e["model"] != model:
            continue
        config = e["config"]
        name = config["condition"]["name"]
        if (config["needle_seed"] == needle_seed
                and config["haystack_tokens"] == haystack_tokens
                and config["plan_seed"] == plan_seed and config["sample_id"] in kept
                and name != rq3.NO_PLAN.name
                and bool(config.get("screening")) == (screened_fp16 and name == "fp16")):
            latest[(config["sample_id"], config["experiment"], name)] = e
    return latest


def _accuracy(runs: list[dict], right: np.ndarray, interval: np.ndarray,
              rng: np.random.Generator) -> dict:
    """One condition's accuracy, overall and at each depth (by_depth), its
    effective bits and its needle's pages' tiers."""
    correct = int(right.sum())
    return {"correct": correct, "n": len(right), "accuracy": correct / len(right),
            "bootstrap_95": [float(v) for v in interval],
            "wilson_95": list(wilson(correct, len(right))),
            "by_depth": by_depth(runs, right, rng),
            "effective_bits": _mean(runs, "effective_bits"),
            "needle_page_tiers": pooled_tiers(runs, "needle_page_tiers")}


def by_depth(runs: list[dict], right: np.ndarray, rng: np.random.Generator) -> dict:
    """At each depth the runs hold, the needles `right` holds as right, of
    how many, with the bootstrap and Wilson 95% intervals of their share."""
    depths = np.array([e["config"]["depth"] for e in runs])
    out = {}
    for depth in needles.DEPTHS:
        at = right[np.isclose(depths, depth)]
        if len(at):
            out[_depth_name(depth)] = {
                "correct": int(at.sum()), "n": len(at), "accuracy": float(at.mean()),
                "bootstrap_95": [float(v) for v in bootstrap(at[None, :], rng)[0]],
                "wilson_95": list(wilson(int(at.sum()), len(at)))}
    return out


def outcomes(runs: list[dict]) -> np.ndarray:
    """Whether each run was right, in the runs' order."""
    return np.array([e["results"]["correct"] for e in runs])


def pooled_tiers(runs: list[dict], key: str) -> dict[str, float]:
    """Each tier's share of every (layer, page) the runs' `key`, tier
    counts, hold together."""
    tiers = Counter()
    for e in runs:
        tiers.update(e["results"][key])
    held = sum(tiers.values())
    return {t: n / held for t, n in sorted(tiers.items())}


def paired(semantic: np.ndarray, other: np.ndarray, rng: np.random.Generator, *,
           named: tuple[str, str] = ("semantic", "other")) -> dict:
    """The semantic source against another, sample by sample, or, `named`
    otherwise, any two conditions: the samples only each answered, under
    its name, McNemar's test on them, and the first's difference in
    accuracy from the second's."""
    semantic_only = int((semantic & ~other).sum())
    other_only = int((~semantic & other).sum())
    difference = bootstrap((semantic.astype(float) - other)[None, :], rng)[0]
    return {f"{named[0]}_only": semantic_only, f"{named[1]}_only": other_only,
            "mcnemar_p": mcnemar(semantic_only, other_only),
            "difference": (int(semantic.sum()) - int(other.sum())) / len(semantic),
            "difference_bootstrap_95": [float(v) for v in difference]}


def proportion(right: np.ndarray, rng: np.random.Generator) -> dict:
    """The samples `right` holds as right, of how many, with the bootstrap
    and Wilson 95% intervals of their share: the anchor-fact and recovery
    summaries' (#154, #155)."""
    correct = int(right.sum())
    return {"correct": correct, "n": len(right),
            "bootstrap_95": [float(v) for v in bootstrap(right[None, :], rng)[0]],
            "wilson_95": list(wilson(correct, len(right)))}


def paired_at_levels(right: dict[str, np.ndarray], rng: np.random.Generator) -> dict:
    """At each level, the semantic source paired against uniform and random
    (`paired`), from each condition's outcomes in `right`, by name: for the
    levels and sources it holds."""
    pairs = {}
    for share in LEVELS:
        names = {c.source: c.name for c in rq3.CONDITIONS if c.share == share}
        if names["semantic"] not in right:
            continue
        pairs[level_name(share)] = {
            f"semantic-{other}": paired(right[names["semantic"]], right[names[other]], rng)
            for other in OTHER_SOURCES if names[other] in right}
    return pairs


def summarise(entries: Sequence[dict], model: str, needle_seed: int, haystack_tokens: int, *,
              plan_seed: int = 0, experiments: Sequence[str] = rq3.EXPERIMENTS) -> dict:
    """The RQ3 summary of the screened set's runs in `entries`, as the
    module describes, its closing answer always among it. A ValueError if
    `experiments` leaves out the closing one, or names what is missing if a
    sample has no run in an experiment under a condition."""
    if CLOSING_EXPERIMENT not in experiments:
        raise ValueError(f"the RQ3 summary answers the closing question, which is "
                         f"{CLOSING_EXPERIMENT}'s; got the experiments {list(experiments)}")
    screening = rq3.last_screening(entries, model, needle_seed, haystack_tokens)
    kept = screening["results"]["kept"]
    latest = _pool(entries, model, needle_seed, haystack_tokens, plan_seed, kept)
    missing = [(s, experiment, name) for experiment in experiments
               for name in _conditions(experiment) for s in kept
               if (s, experiment, name) not in latest]
    if missing:
        raise ValueError(f"no run of {len(missing)} samples' conditions, such as "
                         f"{missing[:3]}; run tools/rq3.py for them")

    rng = np.random.default_rng(SEED)
    accuracy, pairs, pooled = {}, {}, []
    for experiment in experiments:
        names = _conditions(experiment)
        runs = {name: [latest[(s, experiment, name)] for s in kept] for name in names}
        pooled += [e["sha256"] for name in names for e in runs[name]]
        right = {name: outcomes(runs[name]) for name in names}
        intervals = bootstrap(np.stack([right[name] for name in names]), rng)
        accuracy[experiment] = {
            name: _accuracy(runs[name], right[name], interval, rng)
            for name, interval in zip(names, intervals)}
        pairs[experiment] = {
            level_name(share): {
                f"semantic-{other}": paired(right[_source_condition("semantic", share)],
                                             right[_source_condition(other, share)], rng)
                for other in OTHER_SOURCES}
            for share in LEVELS}
    return {"model": model, "needle_seed": needle_seed, "haystack_tokens": haystack_tokens,
            "plan_seed": plan_seed, "samples": len(kept), "significance": SIGNIFICANCE,
            "bootstrap": {"resamples": RESAMPLES, "seed": SEED},
            "accuracy": accuracy, "paired": pairs, "closing": _closing(accuracy, pairs),
            "screening": screening["sha256"], "entries": pooled}


def summarise_long(entries: Sequence[dict], model: str, needle_seed: int, *,
                   plan_seed: int = 0) -> dict:
    """The 32K subset's summary (#156): of each of its ten samples, one a
    depth at needles.LONG_HAYSTACK_TOKENS, the latest question-blind run
    under each of LONG_CONDITIONS. Each condition's accuracy with its
    bootstrap and Wilson intervals, which of the depths it answered, and its effective bits; the
    semantic source against uniform, sample by sample, and the way they
    point. It confirms the direction, with no test claimed (ADR-0017): ten
    samples cannot reject, since McNemar's test needs fewest_to_reject()
    discordant ones. A ValueError names what is missing."""
    kept = [needles.sample_id(d, 0) for d in needles.DEPTHS]
    latest = _pool(entries, model, needle_seed, needles.LONG_HAYSTACK_TOKENS, plan_seed,
                   kept, screened_fp16=False)
    missing = [(s, name) for name in LONG_CONDITIONS for s in kept
               if (s, CLOSING_EXPERIMENT, name) not in latest]
    if missing:
        raise ValueError(f"no run of {len(missing)} samples' conditions, such as "
                         f"{missing[:3]}; run tools/rq3.py --long for them")
    runs = {name: [latest[(s, CLOSING_EXPERIMENT, name)] for s in kept]
            for name in LONG_CONDITIONS}
    right = {name: outcomes(runs[name])
             for name in LONG_CONDITIONS}
    rng = np.random.default_rng(SEED)
    accuracy = {}
    for name in LONG_CONDITIONS:
        accuracy[name] = {
            **proportion(right[name], rng), "accuracy": float(right[name].mean()),
            "by_depth": {_depth_name(d): bool(r) for d, r in zip(needles.DEPTHS, right[name])},
            "effective_bits": _mean(runs[name], "effective_bits"),
            "needle_page_tiers": pooled_tiers(runs[name], "needle_page_tiers")}
    semantic, uniform = (right[name] for name in LONG_CONDITIONS[1:])
    # named as the closing answer names its counts, with no test beside them
    pair = {"semantic_only": int((semantic & ~uniform).sum()),
            "uniform_only": int((~semantic & uniform).sum()),
            "both": int((semantic & uniform).sum()), "neither": int((~semantic & ~uniform).sum())}
    direction = (None if pair["semantic_only"] == pair["uniform_only"]
                 else "semantic" if pair["semantic_only"] > pair["uniform_only"] else "uniform")
    return {"model": model, "needle_seed": needle_seed,
            "haystack_tokens": needles.LONG_HAYSTACK_TOKENS, "plan_seed": plan_seed,
            "experiment": CLOSING_EXPERIMENT, "samples": len(kept),
            "bootstrap": {"resamples": RESAMPLES, "seed": SEED},
            "accuracy": accuracy, "paired": pair, "direction": direction,
            "claim": f"the direction only, no test: McNemar's exact test rejects at "
                     f"{SIGNIFICANCE} only on {fewest_to_reject()} discordant samples or more",
            "entries": [e["sha256"] for name in LONG_CONDITIONS for e in runs[name]]}


def summarise_ablations(entries: Sequence[dict], model: str, needle_seed: int,
                        haystack_tokens: int, *, plan_seed: int = 0) -> dict:
    """ADR-0018's quality ablations (#158) on the screened set, each against
    the condition it changes, on the same samples: the full policy,
    rq3.FULL_POLICY, or for no INT2 at rq3.NO_INT2_SHARE the semantic source
    there (rq3.measured_against). The latest question-blind run of each
    sample under each is pooled. Each ablation gives:
    - its accuracy, with its bootstrap and Wilson intervals;
    - its change in accuracy from the condition it changes, with the paired
      difference's bootstrap interval, and McNemar's exact test on the
      samples only one of the two answered;
    - its effective bits, over every full page and over the pages the share
      is of, the mean over the samples, and how many of its plans fell
      short of the bytes asked, so that equal bytes are checked, not
      assumed. No INT2 at 80% is short on every plan: INT4 gives back 71%
      of a page's FP16 bytes at most.
    A ValueError names what is missing, or the layer weights, should the
    layer weights' runs have been made with more than one."""
    if rq3.FULL_POLICY.share != CLOSING_SHARE:
        raise ValueError("the ablations change the closing question's condition")
    screening = rq3.last_screening(entries, model, needle_seed, haystack_tokens)
    kept = screening["results"]["kept"]
    latest = _pool(entries, model, needle_seed, haystack_tokens, plan_seed, kept)
    names = [*rq3.ABLATED_FROM, *rq3.ABLATION_NAMES]
    missing = [(s, name) for name in names for s in kept
               if (s, CLOSING_EXPERIMENT, name) not in latest]
    if missing:
        raise ValueError(f"no run of {len(missing)} samples' conditions, such as "
                         f"{missing[:3]}; run tools/rq3.py --ablations for them")
    runs = {name: [latest[(s, CLOSING_EXPERIMENT, name)] for s in kept] for name in names}
    weights = {tuple(e["config"]["condition"]["options"]["layer_weights"])
               for e in runs[f"{rq3.FULL_POLICY.name}-layer-weights"]}
    if len(weights) > 1:
        raise ValueError(f"the layer weights' runs were made with {len(weights)} sets of "
                         f"weights; run them again with the last logged")
    right = {name: outcomes(runs[name]) for name in names}
    rng = np.random.default_rng(SEED)

    def held(name: str) -> dict:
        return held_at(runs[name], right[name], rng)

    ablations = {}
    for name in rq3.ABLATION_NAMES:
        condition = runs[name][0]["config"]["condition"]
        against = rq3.source_condition(condition["source"], condition["share"]).name
        pair = paired(right[name], right[against], rng, named=("ablation", "against"))
        ablations[name] = {
            "element": condition["ablates"], "against": against, **held(name),
            "change": pair.pop("difference"),
            "change_bootstrap_95": pair.pop("difference_bootstrap_95"), **pair}
    return {"model": model, "needle_seed": needle_seed, "haystack_tokens": haystack_tokens,
            "plan_seed": plan_seed, "experiment": CLOSING_EXPERIMENT,
            "samples": len(kept), "significance": SIGNIFICANCE,
            "bootstrap": {"resamples": RESAMPLES, "seed": SEED},
            "against": {name: held(name) for name in rq3.ABLATED_FROM},
            "ablations": ablations, "layer_weights": list(weights.pop()),
            "screening": screening["sha256"],
            "entries": [e["sha256"] for name in names for e in runs[name]]}


def held_at(runs: list[dict], right: np.ndarray, rng: np.random.Generator) -> dict:
    """One condition's runs, `right` their outcomes: its accuracy with its
    bootstrap and Wilson intervals, its effective bits, over every full
    page and over the pages the share is of, the mean over the runs, and
    how many of its plans met the bytes asked and fell short, so that
    equal bytes are checked, not assumed; and the commits it ran at."""
    plans = [e["results"]["plan"] for e in runs]
    return {**proportion(right, rng), "accuracy": float(right.mean()),
            "effective_bits": _mean(runs, "effective_bits"),
            "candidate_effective_bits": _mean(runs, "candidate_effective_bits"),
            "reclaimed_bytes": float(np.mean([p["reclaimed_bytes"] for p in plans])),
            "short": sum(p["short"] for p in plans),
            "target_met": sum(p["target_met"] for p in plans),
            "commits": sorted({e["git_commit"] for e in runs})}


#: The conditions an eviction summary pools: each level's semantic source,
#: and eviction there (#160).
EVICTED_FROM = tuple(rq3.source_condition("semantic", c.share).name for c in rq3.EVICTIONS)
EVICTION_NAMES = (*EVICTED_FROM, *rq3.EVICTION_CONDITIONS)


def evictions(runs: dict[str, list[dict]], rng: np.random.Generator) -> dict:
    """At each level, eviction (#160) against the semantic source there, on
    the same samples, from each condition's runs in `runs`, by name: each
    held (held_at), the samples only each answered, McNemar's exact test
    on them, and eviction's change in accuracy from semantic, with the
    paired difference's bootstrap interval."""
    right = {name: outcomes(runs[name]) for name in EVICTION_NAMES}
    out = {}
    for cond in rq3.EVICTIONS:
        against = rq3.source_condition("semantic", cond.share).name
        pair = paired(right[cond.name], right[against], rng, named=("eviction", "semantic"))
        out[level_name(cond.share)] = {
            "eviction": {"name": cond.name, **held_at(runs[cond.name], right[cond.name], rng)},
            "semantic": {"name": against, **held_at(runs[against], right[against], rng)},
            "change": pair.pop("difference"),
            "change_bootstrap_95": pair.pop("difference_bootstrap_95"), **pair}
    return out


def eviction_table(levels: dict) -> str:
    """`evictions`' levels as a markdown table: the needle's and the
    anchor-fact dialogue's (#160)."""
    rows = ["| Level | Eviction | Semantic | Change, bootstrap 95% | Only eviction "
            "| Only semantic | McNemar p | Bits, eviction / semantic "
            "| Bytes reclaimed, eviction / semantic (MiB) | Short plans |",
            "|---|---|---|---|---|---|---|---|---|---|"]
    for level, x in levels.items():
        e, m = x["eviction"], x["semantic"]
        low, high = x["change_bootstrap_95"]
        rows.append(f"| {level} | {e['correct']}/{e['n']} | {m['correct']}/{m['n']} "
                    f"| {x['change']:+.2f} [{low:.2f}, {high:.2f}] | {x['eviction_only']} "
                    f"| {x['semantic_only']} | {x['mcnemar_p']:.3g} "
                    f"| {e['effective_bits']:.2f} / {m['effective_bits']:.2f} "
                    f"| {e['reclaimed_bytes'] / 2**20:.1f} / {m['reclaimed_bytes'] / 2**20:.1f} "
                    f"| {e['short']} / {m['short']} |")
    return "\n".join(rows)


def summarise_evictions(entries: Sequence[dict], model: str, needle_seed: int,
                        haystack_tokens: int, *, plan_seed: int = 0) -> dict:
    """Eviction on the screened needle set (#160), question-blind, against
    the semantic source at each level (evictions): the latest run of each
    sample under each, with each condition's accuracy by depth and where its
    needles' pages were held, or whether they were evicted. A ValueError
    names what is missing."""
    screening = rq3.last_screening(entries, model, needle_seed, haystack_tokens)
    kept = screening["results"]["kept"]
    latest = _pool(entries, model, needle_seed, haystack_tokens, plan_seed, kept)
    missing = [(s, name) for name in EVICTION_NAMES for s in kept
               if (s, CLOSING_EXPERIMENT, name) not in latest]
    if missing:
        raise ValueError(f"no run of {len(missing)} samples' conditions, such as "
                         f"{missing[:3]}; run tools/rq3.py --evictions for them")
    runs = {name: [latest[(s, CLOSING_EXPERIMENT, name)] for s in kept]
            for name in EVICTION_NAMES}
    rng = np.random.default_rng(SEED)
    levels = evictions(runs, rng)
    return {"model": model, "needle_seed": needle_seed, "haystack_tokens": haystack_tokens,
            "plan_seed": plan_seed, "experiment": CLOSING_EXPERIMENT,
            "samples": len(kept), "significance": SIGNIFICANCE,
            "bootstrap": {"resamples": RESAMPLES, "seed": SEED}, "levels": levels,
            "depths": [_depth_name(d) for d in needles.DEPTHS],
            "by_depth": {name: by_depth(runs[name], outcomes(runs[name]), rng)
                         for name in EVICTION_NAMES},
            "needle_page_tiers": {name: pooled_tiers(runs[name], "needle_page_tiers")
                                  for name in EVICTION_NAMES},
            "screening": screening["sha256"],
            "entries": [e["sha256"] for name in EVICTION_NAMES for e in runs[name]]}


def _mean(runs: list[dict], key: str) -> float:
    return float(np.mean([e["results"][key] for e in runs]))


def _closing(accuracy: dict, paired: dict) -> dict:
    experiment, level = CLOSING_EXPERIMENT, level_name(CLOSING_SHARE)
    pair = paired[experiment][level]["semantic-uniform"]
    semantic = accuracy[experiment][_source_condition("semantic", CLOSING_SHARE)]
    uniform = accuracy[experiment][_source_condition("uniform", CLOSING_SHARE)]
    beats = semantic["correct"] > uniform["correct"] and pair["mcnemar_p"] < SIGNIFICANCE
    n, needed = semantic["n"], fewest_to_reject()
    answer = (f"{'Yes' if beats else 'No'}: the semantic source answers "
              f"{semantic['correct']}/{n} and uniform {uniform['correct']}/{n} at {level}, "
              f"{experiment}; McNemar's exact p = {pair['mcnemar_p']:.3g} on "
              f"{pair['semantic_only']} samples only semantic answered and "
              f"{pair['other_only']} only uniform did"
              + ("" if beats else f", so semantic does not beat uniform at {SIGNIFICANCE}")
              + f". The test rejects only on {needed} discordant samples or more, all one way")
    return {"question": f"Does the semantic source keep needle accuracy better than uniform "
                        f"at {level} of the FP16 bytes outside the recency floor reclaimed, "
                        f"{experiment}?",
            "experiment": experiment, "level": level, "semantic_correct": semantic["correct"],
            "uniform_correct": uniform["correct"], "n": n,
            "semantic_only": pair["semantic_only"], "uniform_only": pair["other_only"],
            "mcnemar_p": pair["mcnemar_p"], "significance": SIGNIFICANCE,
            "fewest_discordant_to_reject": needed,
            "semantic_beats_uniform": beats, "answer": answer}
