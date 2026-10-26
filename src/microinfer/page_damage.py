"""Page-level damage: a page's score against what its INT4 downgrade costs (#188).

The policy's premise is that a page the scorer scores higher costs more to
downgrade. This tests it directly, page by page, on RQ3's three tasks, each
run to its **plan point**, where a controlled plan would be applied:
- **needle:** the end of turn 1, question-blind (rq3.before_plan);
- **continuation:** the end of the warm tokens (continuation.Passage);
- **anchor:** the end of the last turn (anchors.before_plan).

The cache is held as every source condition holds it until the plan, at
FP16, adaptive and scoring (rq3.NO_PLAN), and snapshotted there. At the plan
point, a seeded draw of PAGES_PER_SAMPLE (layer, page) pairs of the sample,
among the pages a plan may move (controller.candidates, outside the recency
floor) that the scorer has observed: each pair's score is the scorer's EWMA
there. For each pair, a conversation started from the snapshot moves that
page alone to INT4, and the **measured** tokens are teacher-forced, in
prefill chunks:
- needle and anchor: the question, then the answer FP16 gives to it, a turn
  from the snapshot; the logits compared are those that predict each of the
  answer's tokens;
- continuation: the passage's measured tokens, as continuation fidelity
  measures them (microinfer.continuation); the logits after each.

The page's **damage** is the mean over those positions of KL(FP16 || INT4)
of the softmaxed logits, in float64 (Reference.damage), against the
same tokens forced from the snapshot with no page moved. Each pair starts
from the snapshot, so no pair's move reaches another's; a page moved and
moved back from its shadow gives the logits of a cache never moved, to the
bit, as tests/test_page_damage.py checks.

The summary gives, per task, Spearman's rho between score and damage:
pooled over every sample's pairs, and per sample, their mean, each with a
95% bootstrap interval over samples, since the pages are clustered within
them.
"""

from __future__ import annotations

import zlib
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import _microinfer, anchors, benchlog, controller, rq3, rq3_summary

TASKS = ("needle", "continuation", "anchor")
PAGES_PER_SAMPLE = 64
TIER = "INT4"
KIND = "page-damage"
SUMMARY_KIND = "page-damage-summary"
#: The bootstrap's seed: the summary repeats exactly.
SEED = 188


@dataclass(frozen=True)
class PlanPoint:
    """A sample held at its plan point: the snapshot there, the positions it
    holds, the tokens forced after it and how many of their last rows of
    logits are compared, and whether the last turn's chosen token is forced
    first (Conversation.force's after_turn)."""

    snapshot: object
    positions: int
    forced: np.ndarray
    measured: int
    after_turn: bool


def needle_point(engine, sample) -> PlanPoint:
    """A needle sample at the end of turn 1, question-blind; the question's
    answer FP16 gives, decoded from there."""
    def opening(chat):
        return rq3.before_plan(engine, chat, sample, rq3_summary.CLOSING_EXPERIMENT).turn_2

    return _answered(engine, opening, rq3.ANSWER_TOKENS)


def anchor_point(engine, dialogue) -> PlanPoint:
    """An anchor-fact dialogue at the end of its last turn; the question's
    answer FP16 gives, decoded from there."""
    return _answered(engine, lambda chat: anchors.before_plan(engine, chat, dialogue).question,
                     rq3.ANSWER_TOKENS)


def continuation_point(engine, passage) -> PlanPoint:
    """A passage at the end of its warm tokens, its measured tokens after."""
    with rq3.set_for(engine, rq3.NO_PLAN), engine.conversation(for_snapshot=True) as chat:
        chat.force(passage.prefix)
        chat.force(passage.warm, one_by_one=True)
        snapshot, positions = chat.snapshot(), chat.positions
    return PlanPoint(snapshot, positions, passage.measured, len(passage.measured), False)


def _answered(engine, opening, answer_tokens: int) -> PlanPoint:
    """The plan point after `opening(chat)`, which gives the question's
    prompt, and FP16's answer to it."""
    with rq3.set_for(engine, rq3.NO_PLAN), engine.conversation(for_snapshot=True) as chat:
        question = opening(chat)
        snapshot, positions = chat.snapshot(), chat.positions
    with rq3.set_for(engine, rq3.NO_PLAN), engine.conversation(start=snapshot) as chat:
        answer = chat.turn(question, answer_tokens)
    return PlanPoint(snapshot, positions, np.concatenate([question, answer[:-1]]),
                     len(answer), True)


def page_seed(sample_id: str, seed: int) -> int:
    """The draw's seed for one sample: `seed` and the sample's id."""
    return zlib.crc32(f"{seed}:{sample_id}".encode())


def drawable(scores: np.ndarray, tiers: np.ndarray, positions: int,
             page_tokens: int) -> np.ndarray:
    """The pages a plan may move that the scorer has observed, (layers,
    pages) as `tiers` holds them; `scores` may hold more pages."""
    return (controller.candidates(tiers, positions, page_tokens,
                                  controller.DEFAULT_RECENCY_FLOOR)
            & np.isfinite(scores[:, :tiers.shape[1]]))


def draw_pages(movable: np.ndarray, count: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """`count` (layer, page) pairs drawn without replacement, by `seed`,
    from `movable`, (layers, pages), all of them if fewer: their layers and
    pages, in the order drawn."""
    layers, pages = np.nonzero(movable)
    picked = np.random.default_rng(seed).permutation(len(layers))[:count]
    return layers[picked], pages[picked]


def forced_logits(engine, point: PlanPoint, move: tuple[int, int] | None = None) -> np.ndarray:
    """The rows of logits compared, forced from the snapshot, with page
    `move`, (layer, page), moved to TIER first if given."""
    with rq3.set_for(engine, rq3.NO_PLAN), engine.conversation(start=point.snapshot) as chat:
        if move is not None:
            chat.move(*move, TIER)
        logits = chat.force(point.forced, logits=True, after_turn=point.after_turn)
    return logits[-point.measured:]


class Reference:
    """The logits of the measured positions with no page moved, softmaxed
    once for every page's damage: KL(p || q) = sum p log p - sum p q +
    logsumexp(q), since p sums to 1, so a page's takes one pass over its
    logits q, in float64."""

    def __init__(self, logits: np.ndarray):
        log_p = logits.astype(np.float64)
        log_p -= _logsumexp(log_p)
        self.p = np.exp(log_p)
        self.p_log_p = (self.p * log_p).sum(axis=1)

    def damage(self, logits: np.ndarray) -> float:
        """The mean over positions of KL(reference || `logits`), (n, vocab)."""
        if logits.shape != self.p.shape:
            raise ValueError(f"logits of {logits.shape} against a reference of {self.p.shape}")
        q = logits.astype(np.float64)
        kl = self.p_log_p - (self.p * q).sum(axis=1) + _logsumexp(q)[:, 0]
        return float(kl.mean())


def _logsumexp(rows: np.ndarray) -> np.ndarray:
    """Each row's logsumexp, (n, 1)."""
    top = rows.max(axis=1, keepdims=True)
    return top + np.log(np.exp(rows - top).sum(axis=1, keepdims=True))


def run(engine, point: PlanPoint, count: int, seed: int) -> dict:
    """One sample's pairs at its plan point: what a log entry's results
    hold, each pair's layer, page, score and damage, the sample's rho."""
    rq3.check_engine(engine)
    reference = Reference(forced_logits(engine, point))
    with rq3.set_for(engine, rq3.NO_PLAN), engine.conversation(start=point.snapshot) as chat:
        scores, tiers = chat.scores(), chat.tiers()
    movable = drawable(scores, tiers, point.positions, _microinfer.device.page_tokens)
    layers, pages = draw_pages(movable, count, seed)
    kl = [reference.damage(forced_logits(engine, point, (layer, page)))
          for layer, page in zip(layers.tolist(), pages.tolist())]
    score = scores[layers, pages].astype(float).tolist()
    return {"positions": point.positions, "measured_positions": point.measured,
            "candidates": int(movable.sum()),
            "layers": layers.tolist(), "pages": pages.tolist(), "scores": score,
            "kl": kl, "rho": spearman(score, kl)}


def ranks(values: Sequence[float]) -> np.ndarray:
    """The ranks of `values`, from 1, ties given the mean of theirs."""
    _, group, counts = np.unique(np.asarray(values, dtype=float), return_inverse=True,
                                 return_counts=True)
    last = np.cumsum(counts)
    return (last - (counts - 1) / 2)[group]


def spearman(x: Sequence[float], y: Sequence[float]) -> float | None:
    """Spearman's rho of `x` and `y`: Pearson's correlation of their ranks;
    None if either is constant or there are fewer than two."""
    if len(x) != len(y):
        raise ValueError(f"{len(x)} values against {len(y)}")
    if len(x) < 2:
        return None
    rx, ry = ranks(x), ranks(y)
    rx, ry = rx - rx.mean(), ry - ry.mean()
    norm = np.sqrt((rx ** 2).sum() * (ry ** 2).sum())
    return None if norm == 0 else float((rx * ry).sum() / norm)


def config(task: str, sample_id: str, count: int, seed: int, **more) -> dict:
    """What a log entry's config names of a sample's run."""
    return {"task": task, "sample_id": sample_id, "pages_per_sample": count,
            "page_seed": seed, "tier": TIER,
            "recency_floor": controller.DEFAULT_RECENCY_FLOOR, **more}


def _of(entry: dict, task: str, model: str, count: int, seed: int) -> bool:
    c = entry["config"]
    return (entry["kind"] == KIND and entry["model"] == model and c["task"] == task
            and c["pages_per_sample"] == count and c["page_seed"] == seed)


def logged(task: str, model: str, count: int, seed: int, commit: str,
           log: str | Path = benchlog.DEFAULT_LOG) -> set[str]:
    """The samples of `task` logged for `model`, `count` pairs each drawn by
    `seed`, at the code of `commit` (benchlog.same_code): a run of hours
    stopped part way is taken up again from them."""
    if not Path(log).exists():
        return set()
    held = benchlog.same_code_as(commit, log)
    return {e["config"]["sample_id"] for e in benchlog.read(log)
            if _of(e, task, model, count, seed) and held(e["git_commit"])}


def summarise(entries: Sequence[dict], task: str, model: str, sample_ids: Sequence[str],
              count: int = PAGES_PER_SAMPLE, seed: int = 0) -> dict:
    """The page-damage summary of `task` over the samples `sample_ids`, the
    latest entry of each: rho pooled over every pair and its mean per
    sample, each with a 95% bootstrap interval over samples, as the module
    describes; a ValueError naming the samples missing."""
    latest = {e["config"]["sample_id"]: e for e in entries if _of(e, task, model, count, seed)}
    missing = [s for s in sample_ids if s not in latest]
    if missing:
        raise ValueError(f"no {task} run of {len(missing)} of the {len(sample_ids)} samples, "
                         f"such as {missing[:3]}; run tools/page_damage.py for them")
    runs = [latest[s]["results"] for s in sample_ids]
    per_sample = np.array([np.nan if r["rho"] is None else r["rho"] for r in runs])
    rng = np.random.default_rng(SEED)
    picks = rng.integers(0, len(runs), size=(rq3_summary.RESAMPLES, len(runs)))

    def pooled(chosen) -> float | None:
        return spearman([s for i in chosen for s in runs[i]["scores"]],
                        [k for i in chosen for k in runs[i]["kl"]])

    resampled_pooled = np.array([np.nan if (r := pooled(p)) is None else r for p in picks])
    resampled_mean = np.nanmean(per_sample[picks], axis=1)

    def interval(values: np.ndarray) -> list[float]:
        return [float(v) for v in np.nanpercentile(values, [2.5, 97.5])]

    return {"task": task, "model": model, "samples": len(runs),
            "pairs": sum(len(r["kl"]) for r in runs), "pages_per_sample": count,
            "page_seed": seed, "tier": TIER,
            "pooled_rho": pooled(range(len(runs))),
            "pooled_rho_bootstrap_95": interval(resampled_pooled),
            "mean_sample_rho": float(np.nanmean(per_sample)),
            "mean_sample_rho_bootstrap_95": interval(resampled_mean),
            "samples_without_rho": int(np.isnan(per_sample).sum()),
            "positive_samples": int((per_sample > 0).sum()),
            "kl_median": float(np.median([k for r in runs for k in r["kl"]])),
            "bootstrap": {"resamples": rq3_summary.RESAMPLES, "seed": SEED},
            "commits": sorted({latest[s]["git_commit"] for s in sample_ids}),
            "entries": [latest[s]["sha256"] for s in sample_ids]}
