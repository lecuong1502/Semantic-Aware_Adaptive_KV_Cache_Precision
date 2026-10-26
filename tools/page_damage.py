#!/usr/bin/env python3
"""Measure each page's damage against its score on RQ3's tasks, and log it (#188).

    .venv/bin/python tools/page_damage.py --issue 189 --task needle
    .venv/bin/python tools/page_damage.py --issue 189 --task needle --summarise

Each sample of the task's set (microinfer.page_damage), run to its plan
point; there, --pages-per-sample (layer, page) pairs drawn by --page-seed,
each moved alone to INT4 and its damage measured, the KL from FP16 over the
measured tokens. One "page-damage" entry a sample, naming the task, the
sample and the code that produced it, with every pair's score and damage
and the sample's Spearman rho. The sets are RQ3's: the screened needle set
and the screened anchor-fact set, from the last screening logged for the
model, and the first continuation passages; --unscreened draws the needle
and anchor sets' first samples instead, as a check on a model no screening
was logged for. A sample already logged at the same code is not run again,
so that a run of hours stopped part way is taken up by the same command.
--summarise pools the samples into one "page-damage-summary" entry, rho
pooled and per sample with bootstrap intervals over samples, and prints it.
Refuses a dirty tree, as the checkpoint recorder does.

On Qwen2.5-1.5B and the RTX 4050, a page takes about 1 s, so 64 a sample
about a minute, and running a sample to its plan point about 85 s for a
needle, 135 s for a dialogue and 12 s for a passage: the 50 samples of a
task take about 2 hours for the needle, 2.75 for the dialogue and 1 for
continuation.
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any, NamedTuple

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "tools"))

from perplexity import held_out_text  # noqa: E402
from rq3 import NEEDLE_SEED  # noqa: E402
from summarise_rq3 import interval  # noqa: E402

from microinfer import Engine, anchors, benchlog, continuation, needles, page_damage  # noqa: E402
from microinfer import rq3  # noqa: E402


def needles_of(model: str, engine: Engine, log: Path, unscreened: bool) -> list:
    """The screened needle set, in screening's order, or the set's first."""
    text = held_out_text()
    if unscreened:
        return needles.needle_set(text, engine.tokenizer, seed=NEEDLE_SEED)
    kept, draws = rq3.screened(model, NEEDLE_SEED, needles.DEFAULT_HAYSTACK_TOKENS, log)
    every = needles.needle_set(text, engine.tokenizer, seed=NEEDLE_SEED,
                               samples_per_depth=draws)
    return kept_of(every, kept)


def dialogues_of(model: str, engine: Engine, log: Path, unscreened: bool) -> list:
    """The screened anchor-fact set, in screening's order, or the set's first."""
    of, text = anchors.AnchorSet(model), held_out_text()
    if unscreened:
        return of.dialogues(text, engine.tokenizer, anchors.SAMPLES_PER_GAP)
    screening = anchors.last_screening(benchlog.read(log), of)
    every = of.dialogues(text, engine.tokenizer, screening["config"]["draws"])
    return kept_of(every, screening["results"]["kept"])


def kept_of(every: list, kept: list[str]) -> list:
    by_id = {s.sample_id: s for s in every}
    return [by_id[s] for s in kept]


def passages_of(model: str, engine: Engine, log: Path, unscreened: bool) -> list:
    """The first passages, which no screening chooses."""
    return continuation.passages(engine.encode(held_out_text()))


class Task(NamedTuple):
    """How the tool runs a task: its samples, by (model, engine, log,
    unscreened), each one's plan point, its id, and whether a screening
    chooses the samples."""

    samples: Callable[[str, Engine, Path, bool], list]
    point: Callable
    sample_id: Callable[[Any], str]
    screens: bool


TASKS = {
    "needle": Task(needles_of, page_damage.needle_point, lambda s: s.sample_id, True),
    "continuation": Task(passages_of, page_damage.continuation_point,
                         lambda p: p.passage_id, False),
    "anchor": Task(dialogues_of, page_damage.anchor_point, lambda d: d.sample_id, True),
}


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--issue", type=int, required=True,
                        help="the ticket these runs are made for")
    parser.add_argument("--task", required=True, choices=list(TASKS))
    parser.add_argument("--model", default="qwen2.5-1.5b-instruct")
    parser.add_argument("--samples", type=int, default=None,
                        help="the set's first samples only (default: every one)")
    parser.add_argument("--pages-per-sample", type=int, default=page_damage.PAGES_PER_SAMPLE)
    parser.add_argument("--page-seed", type=int, default=0,
                        help="the draw's seed, with each sample's id (page_damage.page_seed)")
    parser.add_argument("--unscreened", action="store_true",
                        help="the needle or anchor set's first samples, unscreened: a check")
    parser.add_argument("--summarise", action="store_true",
                        help="pool the samples logged into one summary entry, and run none")
    parser.add_argument("--log", type=Path, default=benchlog.DEFAULT_LOG)
    args = parser.parse_args(argv)
    environment = benchlog.environment(args.log)
    if environment["git_dirty"]:
        parser.error("tracked files have uncommitted changes; commit first, so that "
                     "each entry names the code that produced it")
    task = TASKS[args.task]
    of = page_damage.Draws(args.task, args.model, args.pages_per_sample, args.page_seed,
                           screened=task.screens and not args.unscreened)
    engine = Engine(REPO / "models" / args.model)
    try:
        chosen = task.samples(args.model, engine, args.log, args.unscreened)[:args.samples]
    except ValueError as exc:
        parser.error(str(exc))
    if args.summarise:
        return summarise(parser, args, of, [task.sample_id(s) for s in chosen])

    done = page_damage.logged(of, environment["git_commit"], args.log)
    todo = [s for s in chosen if task.sample_id(s) not in done]
    print(f"{len(chosen) - len(todo)} of {len(chosen)} samples logged at this code already; "
          f"{len(todo)} to run", flush=True)
    if not todo:
        return 0
    engine.load_weights()
    start = time.perf_counter()
    for i, sample in enumerate(todo, 1):
        name = task.sample_id(sample)
        point = task.point(engine, sample)
        results = page_damage.run(engine, point, of.pages_per_sample,
                                  page_damage.page_seed(name, of.page_seed))
        benchlog.append(
            page_damage.KIND, model=args.model, context_length=point.positions,
            precision_tiers={"FP16": 1.0}, config=of.config(name, issue=args.issue),
            results=results, log=args.log)
        rho = "--" if results["rho"] is None else f"{results['rho']:+.3f}"
        print(f"{i}/{len(todo)} {name}: {len(results['kl'])} pages, rho {rho}, median KL "
              f"{statistics.median(results['kl']):.2e}  "
              f"({time.perf_counter() - start:.0f} s)", flush=True)
    return 0


def summarise(parser, args, of: page_damage.Draws, ids: list[str]) -> int:
    benchlog.verify(args.log)
    try:
        summary = page_damage.summarise(benchlog.read(args.log), of, ids)
    except ValueError as exc:
        parser.error(str(exc))
    entry = benchlog.append(page_damage.SUMMARY_KIND, model=args.model, context_length=None,
                            precision_tiers=None, config={**of._asdict(), "issue": args.issue},
                            results=summary, log=args.log)
    s = summary
    print(f"\n{s['task']}: {s['samples']} samples, {s['pairs']} pages at {s['tier']}\n")
    print("| Task | Pooled rho | Bootstrap 95% | Mean rho per sample | Bootstrap 95% "
          "| Samples rho > 0 | Median KL |")
    print("|---|---|---|---|---|---|---|")
    print(f"| {s['task']} | {s['pooled_rho']:+.3f} | {interval(s['pooled_rho_bootstrap_95'], 3)} "
          f"| {s['mean_sample_rho']:+.3f} | {interval(s['mean_sample_rho_bootstrap_95'], 3)} "
          f"| {s['positive_samples']}/{s['samples']} | {s['kl_median']:.2e} |")
    print(f"\nlogged {entry['sha256'][:12]}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
