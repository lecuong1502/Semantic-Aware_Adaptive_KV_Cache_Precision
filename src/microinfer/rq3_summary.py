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
LEVELS = sorted({c.share for c in rq3.CONDITIONS if c.share is not None})
OTHER_SOURCES = ("uniform", "random")


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
    (cond,) = [c for c in rq3.CONDITIONS if c.source == source and c.share == share]
    return cond.name


def _conditions(experiment: str) -> list[str]:
    if experiment == "question-aware":
        return list(rq3.SOURCE_CONDITIONS)
    return [c.name for c in rq3.CONDITIONS]


def _pool(entries: Sequence[dict], model: str, needle_seed: int, haystack_tokens: int,
          plan_seed: int, kept: list[str]) -> dict[tuple[str, str, str], dict]:
    """The latest pooled run of each (sample, experiment, condition)."""
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
                and bool(config.get("screening")) == (name == "fp16")):
            latest[(config["sample_id"], config["experiment"], name)] = e
    return latest


def _accuracy(runs: list[dict], right: np.ndarray, interval: np.ndarray,
              rng: np.random.Generator) -> dict:
    """One condition's accuracy, overall and at each depth, its effective
    bits and its needle's pages' tiers."""
    depths = np.array([e["config"]["depth"] for e in runs])
    by_depth = {}
    for depth in needles.DEPTHS:
        at = right[np.isclose(depths, depth)]
        by_depth[_depth_name(depth)] = {
            "correct": int(at.sum()), "n": len(at),
            "accuracy": float(at.mean()),
            "bootstrap_95": [float(v) for v in bootstrap(at[None, :], rng)[0]],
            "wilson_95": list(wilson(int(at.sum()), len(at)))}
    correct = int(right.sum())
    return {"correct": correct, "n": len(right), "accuracy": correct / len(right),
            "bootstrap_95": [float(v) for v in interval],
            "wilson_95": list(wilson(correct, len(right))), "by_depth": by_depth,
            "effective_bits": float(np.mean([e["results"]["effective_bits"] for e in runs])),
            "needle_page_tiers": pooled_tiers(runs, "needle_page_tiers")}


def pooled_tiers(runs: list[dict], key: str) -> dict[str, float]:
    """Each tier's share of every (layer, page) the runs' `key`, tier
    counts, hold together."""
    tiers = Counter()
    for e in runs:
        tiers.update(e["results"][key])
    held = sum(tiers.values())
    return {t: n / held for t, n in sorted(tiers.items())}


def paired(semantic: np.ndarray, other: np.ndarray, rng: np.random.Generator) -> dict:
    """The semantic source against another, sample by sample."""
    semantic_only = int((semantic & ~other).sum())
    other_only = int((~semantic & other).sum())
    difference = bootstrap((semantic.astype(float) - other)[None, :], rng)[0]
    return {"semantic_only": semantic_only, "other_only": other_only,
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
        outcomes = {name: np.array([e["results"]["correct"] for e in runs[name]])
                    for name in names}
        intervals = bootstrap(np.stack([outcomes[name] for name in names]), rng)
        accuracy[experiment] = {
            name: _accuracy(runs[name], outcomes[name], interval, rng)
            for name, interval in zip(names, intervals)}
        pairs[experiment] = {
            level_name(share): {
                f"semantic-{other}": paired(outcomes[_source_condition("semantic", share)],
                                             outcomes[_source_condition(other, share)], rng)
                for other in OTHER_SOURCES}
            for share in LEVELS}
    return {"model": model, "needle_seed": needle_seed, "haystack_tokens": haystack_tokens,
            "plan_seed": plan_seed, "samples": len(kept), "significance": SIGNIFICANCE,
            "bootstrap": {"resamples": RESAMPLES, "seed": SEED},
            "accuracy": accuracy, "paired": pairs, "closing": _closing(accuracy, pairs),
            "screening": screening["sha256"], "entries": pooled}


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
