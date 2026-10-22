"""RQ3's summary: the logged needle runs pooled into one entry (#152).

ADR-0017 answers RQ3 from the benchmark log alone. The runs pooled are the
screened set's (#148), for one model, needle seed, haystack length and plan
seed: the latest run of each sample in each experiment under each
condition, so that a repeat stands for the run before it. FP16 is
screening's runs; NO_PLAN, a check, is not pooled. Question-aware holds the
source conditions only: FP16 and the static tiers make no plan, so they
would decode in it the tokens they decode question-blind.

For each experiment and condition:
- **Accuracy**, overall and by depth, with a bootstrap 95% interval that
  resamples the samples, as ADR-0017 fixes. At the ceiling every resample
  is right and the interval is a point, so the Wilson score interval is
  given beside it, which stays open there.
- **Effective bits**, the mean over the samples, and the needle's pages'
  tiers, as shares of every (layer, page) of them.

For each experiment and level, semantic is paired by sample against uniform
and against random: the samples only one of them answered, McNemar's exact
test on them, two-sided, and the paired difference in accuracy with its
bootstrap interval.

The closing question is semantic against uniform at 80%, question-blind:
semantic beats uniform if it answers more samples and McNemar's test
rejects at ALPHA. Otherwise the answer is no, and it is reported as such.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from math import comb, sqrt

import numpy as np

from . import needles, rq3

#: The closing test's level, two-sided.
ALPHA = 0.05
RESAMPLES = 10_000
#: The bootstrap's seed: the summary repeats exactly.
SEED = 152
Z_95 = 1.959964
CLOSING = {"experiment": "question-blind", "level": "80%"}


def mcnemar(only_a: int, only_b: int) -> float:
    """McNemar's exact test, two-sided: the chance, were neither better, of
    discordant pairs split as unevenly as `only_a` against `only_b`, a
    binomial of their count at one half."""
    n = only_a + only_b
    if n == 0:
        return 1.0
    tail = sum(comb(n, i) for i in range(min(only_a, only_b) + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def bootstrap(outcomes: np.ndarray, rng: np.random.Generator,
              resamples: int = RESAMPLES) -> np.ndarray:
    """The 95% percentile interval of each row's mean, (rows, 2), the
    columns, samples, resampled with replacement alike for every row, so
    that rows of one sample stay paired."""
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


def _conditions(experiment: str) -> list[str]:
    if experiment == "question-aware":
        return list(rq3.SOURCE_CONDITIONS)
    return [c.name for c in rq3.CONDITIONS]


def summarise(entries: Sequence[dict], model: str, needle_seed: int, haystack_tokens: int, *,
              plan_seed: int = 0, experiments: Sequence[str] = rq3.EXPERIMENTS) -> dict:
    """The summary of the screened set's runs in `entries`, as the module
    describes; a ValueError naming what is missing if a sample has no run
    in an experiment under a condition."""
    screening = rq3.last_screening(entries, model, needle_seed, haystack_tokens)
    kept = screening["results"]["kept"]
    latest: dict[tuple[str, str, str], dict] = {}
    for e in entries:
        c = e.get("config", {})
        if (e["kind"] == "rq3-needle" and e["model"] == model
                and c["needle_seed"] == needle_seed and c["haystack_tokens"] == haystack_tokens
                and c["plan_seed"] == plan_seed and c["sample_id"] in kept
                and c["condition"]["name"] != rq3.NO_PLAN.name):
            latest[(c["sample_id"], c["experiment"], c["condition"]["name"])] = e
    missing = [(s, x, name) for x in experiments for name in _conditions(x) for s in kept
               if (s, x, name) not in latest]
    if missing:
        raise ValueError(f"no run of {len(missing)} samples' conditions, such as "
                         f"{missing[:3]}; run tools/rq3.py for them")

    rng = np.random.default_rng(SEED)
    accuracy, paired, pooled = {}, {}, []
    for x in experiments:
        names = _conditions(x)
        runs = [[latest[(s, x, name)] for s in kept] for name in names]
        pooled += [e["sha256"] for row in runs for e in row]
        outcomes = np.array([[e["results"]["correct"] for e in row] for row in runs])
        intervals = bootstrap(outcomes, rng)
        accuracy[x] = {}
        for name, row, right, interval in zip(names, runs, outcomes, intervals):
            depths = Counter()
            tiers = Counter()
            for e, ok in zip(row, right):
                depths[f"{round(e['config']['depth'] * 100)}%"] += int(ok)
                tiers.update(e["results"]["needle_page_tiers"])
            held = sum(tiers.values())
            accuracy[x][name] = {
                "correct": int(right.sum()), "n": len(right), "accuracy": float(right.mean()),
                "bootstrap_95": [float(v) for v in interval],
                "wilson_95": list(wilson(int(right.sum()), len(right))),
                "by_depth": {f"{round(d * 100)}%": depths[f"{round(d * 100)}%"]
                             for d in needles.DEPTHS},
                "effective_bits": float(np.mean([e["results"]["effective_bits"] for e in row])),
                "needle_page_tiers": {t: n / held for t, n in sorted(tiers.items())},
            }
        paired[x] = {}
        index = {name: i for i, name in enumerate(names)}
        for share in (0.7, 0.8):
            level = f"{round(share * 100)}%"
            semantic = outcomes[index[f"semantic-{level[:-1]}"]]
            paired[x][level] = {}
            for other_source in ("uniform", "random"):
                other = outcomes[index[f"{other_source}-{level[:-1]}"]]
                only_a = int((semantic & ~other).sum())
                only_b = int((~semantic & other).sum())
                difference = bootstrap((semantic.astype(float) - other)[None, :], rng)[0]
                paired[x][level][f"semantic-{other_source}"] = {
                    "semantic_only": only_a, "other_only": only_b,
                    "mcnemar_p": mcnemar(only_a, only_b),
                    "difference": (int(semantic.sum()) - int(other.sum())) / len(semantic),
                    "difference_bootstrap_95": [float(v) for v in difference],
                }
    return {"model": model, "needle_seed": needle_seed, "haystack_tokens": haystack_tokens,
            "plan_seed": plan_seed, "samples": len(kept), "alpha": ALPHA,
            "bootstrap": {"resamples": RESAMPLES, "seed": SEED},
            "accuracy": accuracy, "paired": paired,
            "closing": _closing(accuracy, paired) if CLOSING["experiment"] in experiments
            else None,
            "screening": screening["sha256"], "entries": pooled}


def _closing(accuracy: dict, paired: dict) -> dict:
    x, level = CLOSING["experiment"], CLOSING["level"]
    pair = paired[x][level]["semantic-uniform"]
    semantic = accuracy[x][f"semantic-{level[:-1]}"]["correct"]
    uniform = accuracy[x][f"uniform-{level[:-1]}"]["correct"]
    beats = semantic > uniform and pair["mcnemar_p"] < ALPHA
    n = accuracy[x][f"semantic-{level[:-1]}"]["n"]
    answer = (f"{'Yes' if beats else 'No'}: semantic answers {semantic}/{n} and uniform "
              f"{uniform}/{n} at {level}, {x}; McNemar's exact p = {pair['mcnemar_p']:.3g} on "
              f"{pair['semantic_only']} samples only semantic answered and "
              f"{pair['other_only']} only uniform did"
              + ("" if beats else f", so semantic does not beat uniform at alpha {ALPHA}"))
    return {"question": f"Does semantic allocation keep needle accuracy better than uniform "
                        f"at {level} of the FP16 bytes reclaimed, {x}?",
            "experiment": x, "level": level, "semantic_correct": semantic,
            "uniform_correct": uniform, "n": n, "mcnemar_p": pair["mcnemar_p"],
            "alpha": ALPHA, "semantic_beats_uniform": beats, "answer": answer}
