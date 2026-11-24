#!/usr/bin/env python3
"""Run RULER's tasks under RQ3's conditions, and log each run (#208).

    .venv/bin/python tools/ruler.py --issue 208 --length 8K --screen
    .venv/bin/python tools/ruler.py --issue 208 --length 8K --ceiling
    .venv/bin/python tools/ruler.py --issue 208 --length 8K
    .venv/bin/python tools/ruler.py --issue 208 --length 8K --summarise
        [--tasks niah_multikey_1 ... vt] [--conditions ...] [--dry-run]

ADR-0017's amendment, in its order, at one length, 8K or 32K:
1. `--screen`: each task's samples drawn in order, up to --draws, run at
   FP16 until 50 are answered (rq3.screen); each FP16 run an "rq3-ruler"
   entry with screening true, and each task's screened set a
   "ruler-screening" entry.
2. `--ceiling`: static INT2 on each task's screened set. A task is below
   its ceiling if INT2 answers at most 40 of the 50.
3. The conditions, by default every one of B2 to B8 but those --ceiling
   ran (ruler_summary.CONDITIONS), on the tasks below their ceiling; with
   --tasks, on those named.
4. `--summarise`: a "ruler-summary" entry, and the tables.

Every run is question-blind (rq3.run), its sample one of microinfer.ruler's,
from RULER's data as tools/fetch_ruler.py fetched it, checked against the
last "ruler-data" entry's hashes. A run already logged at the same code is
not run again, so that a run of hours stopped part way is taken up by the
same command. A sample's conditions that share turn 1 run it once
(rq3.runs, #172). --dry-run prints the runs still to make and an estimate
of their GPU-hours on the RTX 4050 (`hours`). Refuses a dirty tree.
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import asdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from microinfer import Engine, benchlog, rq3, rq3_summary, ruler, ruler_summary  # noqa: E402

EXPERIMENT = "question-blind"
FP16 = rq3.condition("fp16")
PER_TASK = 50
#: The estimate's parts, in seconds, on the RTX 4050 after #204 and #205:
#: a turn 1, its prefill and summary, and what a condition adds to it, its
#: plan and answer. 32K's prefill is the prefill-step-cost entries' 200 s.
TURN_1_S = {"8K": 12.0, "32K": 205.0}
CLOSING_S = {"8K": 4.0, "32K": 10.0}


def hours(runs: dict[str, list[str]], length: str) -> float:
    """GPU-hours for `runs`, each sample's condition names: one turn 1 for
    those that share it (rq3.shares_turn_1) and one for each of the rest,
    and a closing each."""
    seconds = 0.0
    for names in runs.values():
        conds = [rq3.condition(n) for n in names]
        shared = sum(rq3.shares_turn_1(c) for c in conds)
        turn_1s = (1 if shared else 0) + len(conds) - shared
        seconds += turn_1s * TURN_1_S[length] + len(conds) * CLOSING_S[length]
    return seconds / 3600


def data_hashes(log: Path) -> dict[str, str]:
    """The hashes of RULER's data, refused unless they are the last
    "ruler-data" entry's: every run names the data it ran on."""
    _, _, hashes = ruler.load()
    logged = [e for e in benchlog.read(log) if e["kind"] == ruler.DATA_KIND]
    if not logged:
        raise SystemExit("no ruler-data entry is logged; run tools/fetch_ruler.py")
    expected = {k: v for k, v in logged[-1]["results"]["files"].items() if k in hashes}
    if expected != hashes:
        raise SystemExit(f"data/ruler/ is not the last ruler-data entry's: {hashes} against "
                         f"{expected}")
    return hashes


def config(sample: ruler.Sample, cond: rq3.Condition, args, hashes, *, screening: bool) -> dict:
    return {"issue": args.issue, "task": sample.task, "sample_id": sample.sample_id,
            "length": args.length, "haystack_tokens": sample.haystack_tokens,
            "ruler_seed": args.seed, "plan_seed": args.plan_seed, "screening": screening,
            "condition": asdict(cond), "experiment": EXPERIMENT,
            "summary_tokens": rq3.SUMMARY_TOKENS, "answer_tokens": sample.answer_tokens,
            "query": sample.query, "expected": list(sample.expected),
            "target_positions": [list(p) for p in sample.target_positions],
            "target_pages": [list(p) for p in sample.target_pages], "data": hashes}


def chosen_tasks(named: list[str] | None, summary: dict) -> list[str]:
    """The tasks a run of the conditions is of: those `named`, or else those
    the summary finds below their ceiling (ADR-0017's amendment)."""
    if named:
        return named
    return [t for t, row in summary["tasks"].items() if row["below_ceiling"]]


def logged(args, commit: str) -> dict[tuple[str, str], bool]:
    """The runs logged at this code, by (sample id, condition), with whether
    each was answered: FP16's from screening."""
    if not args.log.exists():
        return {}
    same = benchlog.same_code_as(commit, args.log)
    return {key: e["results"]["correct"] for key, e in ruler_summary.latest_runs(
        [e for e in benchlog.read(args.log) if e["kind"] == ruler_summary.RUN_KIND
         and same(e["git_commit"])], args.model, args.seed, ruler.LENGTHS[args.length],
        args.plan_seed).items()}


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--issue", type=int, required=True)
    parser.add_argument("--model", default="qwen2.5-1.5b-instruct")
    parser.add_argument("--length", choices=list(ruler.LENGTHS), required=True)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--screen", action="store_true", help="screen each task at FP16")
    mode.add_argument("--ceiling", action="store_true",
                      help="static INT2 on each task's screened set")
    mode.add_argument("--summarise", action="store_true", help="log the summary")
    parser.add_argument("--tasks", nargs="+", choices=ruler.TASK_NAMES, default=None)
    parser.add_argument("--conditions", nargs="+", choices=ruler_summary.CONDITIONS,
                        default=None)
    parser.add_argument("--draws", type=int, default=70,
                        help="a task's samples drawn, at most, to screen 50")
    parser.add_argument("--seed", type=int, default=ruler_summary.SEED)
    parser.add_argument("--plan-seed", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--log", type=Path, default=benchlog.DEFAULT_LOG)
    args = parser.parse_args(argv)
    environment = benchlog.environment(args.log)
    if environment["git_dirty"] and not args.dry_run:
        parser.error("tracked files have uncommitted changes; commit first, so that each "
                     "entry names the code that produced it")
    tokens = ruler.LENGTHS[args.length]
    entries = benchlog.read(args.log) if args.log.exists() else []

    if args.summarise:
        s = ruler_summary.summarise(entries, args.model, args.seed, tokens,
                                    plan_seed=args.plan_seed, data=data_hashes(args.log))
        if not s["tasks"]:
            parser.error(f"no task is screened at {args.length}")
        if not args.dry_run:
            benchlog.append(ruler_summary.SUMMARY_KIND, model=args.model, context_length=tokens,
                            precision_tiers=None, config={"issue": args.issue,
                                                          "length": args.length},
                            results=s, log=args.log)
        print_summary(s)
        return 0

    hashes = data_hashes(args.log)
    done = logged(args, environment["git_commit"])
    screened = ruler_summary.screenings(entries, args.model, args.seed, tokens)
    sets = {t: e["results"]["kept"] for t, e in screened.items()}
    if args.screen:
        tasks = args.tasks or list(ruler.TASK_NAMES)
        names = [FP16.name]
    else:
        names = ["static-int2"] if args.ceiling else (args.conditions or [
            n for n in ruler_summary.CONDITIONS if n != "static-int2"])
        if args.ceiling:
            tasks = args.tasks or list(ruler.TASK_NAMES)
        else:
            tasks = chosen_tasks(args.tasks, ruler_summary.summarise(
                entries, args.model, args.seed, tokens, plan_seed=args.plan_seed, data=hashes))
            if not tasks:
                parser.error(f"no task is known below its ceiling at {args.length}; run "
                             f"--ceiling first, or name --tasks")
        unscreened = [t for t in tasks if t not in sets]
        if unscreened:
            parser.error(f"{unscreened} are not screened at {args.length}; run --screen first")

    engine = Engine(REPO / "models" / args.model)
    essays, words, _ = ruler.load()
    drawn = {t: ruler.task_set(t, essays, words, engine.tokenizer, seed=args.seed,
                               template=engine.chat_template,
                               samples=args.draws if args.screen
                               else screened[t]["config"]["draws"],
                               haystack_tokens=tokens) for t in tasks}
    if args.screen:
        todo = {s.sample_id: [FP16.name] for t in tasks for s in drawn[t][:PER_TASK]
                if (s.sample_id, FP16.name) not in done}
        print(f"screening {tasks}: at least {len(todo)} FP16 runs, "
              f"~{hours(todo, args.length):.1f} GPU-hours and more for replacements")
    else:
        todo = {}
        for t in tasks:
            for s in drawn[t]:
                if s.sample_id in sets[t]:
                    left = [n for n in names if (s.sample_id, n) not in done]
                    if left:
                        todo[s.sample_id] = left
        print(f"{sum(map(len, todo.values()))} runs to make on {tasks}: "
              f"~{hours(todo, args.length):.1f} GPU-hours")
    if args.dry_run:
        return 0

    engine.load_weights()
    start = time.perf_counter()

    def log_run(sample: ruler.Sample, cond: rq3.Condition, results: dict, screening: bool):
        benchlog.append(ruler_summary.RUN_KIND, model=args.model,
                        context_length=len(sample.turn_1),
                        precision_tiers=results["tier_shares"],
                        config=config(sample, cond, args, hashes, screening=screening),
                        results=results, log=args.log)
        print(f"{sample.sample_id} {cond.name}: score {results['score']:.2f} "
              f"{results['effective_bits']:.2f} bits {results['answer'][:80]!r}  "
              f"({time.perf_counter() - start:.0f} s)", flush=True)

    if args.screen:
        for t in tasks:
            def is_right(sample: ruler.Sample) -> bool:
                if (sample.sample_id, FP16.name) in done:
                    return done[(sample.sample_id, FP16.name)]
                results = rq3.run(engine, sample, FP16, EXPERIMENT, plan_seed=args.plan_seed)
                log_run(sample, FP16, results, screening=True)
                return results["correct"]

            grouping = rq3.Grouping((t,), lambda s: s.task, str, "tasks")
            try:
                kept, rejected = rq3.screen(drawn[t], is_right, PER_TASK, grouping)
                short = None
            except rq3.ShortScreening as failed:
                kept, rejected, short = failed.kept, failed.rejected, failed
            benchlog.append(
                ruler_summary.SCREENING_KIND, model=args.model, context_length=tokens,
                precision_tiers={"FP16": 1.0},
                config={"issue": args.issue, "task": t, "length": args.length,
                        "haystack_tokens": tokens, "ruler_seed": args.seed,
                        "per_task": PER_TASK, "draws": args.draws, "data": hashes},
                results={"kept": [s.sample_id for s in kept],
                         "rejected": [s.sample_id for s in rejected],
                         "complete": short is None}, log=args.log)
            print(f"{t}: kept {len(kept)}, replaced {len(rejected)}"
                  + ("" if short is None else f"; NOT COMPLETE: {short}"), flush=True)
        return 0

    by_id = {s.sample_id: s for t in tasks for s in drawn[t]}
    for sample_id, left in todo.items():
        sample = by_id[sample_id]
        for cond, results in rq3.runs(engine, sample, [rq3.condition(n) for n in left],
                                      EXPERIMENT, plan_seed=args.plan_seed):
            log_run(sample, cond, results, screening=False)
    return 0


def print_summary(s: dict) -> None:
    names = ["fp16", *ruler_summary.CONDITIONS]
    print("| Task | Ceiling | " + " | ".join(names) + " |")
    print("|---|---|" + "---|" * len(names))
    for task, row in s["tasks"].items():
        cells = [row["by_condition"].get(n) for n in names]
        mark = {True: "below", False: "at", None: "-"}[row["below_ceiling"]]
        print(f"| {task} | {mark} | " + " | ".join(
            "-" if c is None else f"{c['answered']}/{c['runs']} ({c['score_mean']:.2f})"
            for c in cells) + " |")
    sources = ("semantic", *rq3_summary.OTHER_SOURCES)
    placed = [rq3.source_condition(src, share).name
              for share in rq3_summary.LEVELS for src in sources]
    placed = [n for n in placed if any("needle_tier_shares" in row["by_condition"].get(n, {})
                                       for row in s["tasks"].values())]
    if placed:
        print("\nINT2's share of the cache / of the needle's cells / of runs with INT2 on "
              "the needle:")
        print("| Task | " + " | ".join(placed) + " |")
        print("|---|" + "---|" * len(placed))
        for task, row in s["tasks"].items():
            cells = [row["by_condition"].get(n, {}) for n in placed]
            print(f"| {task} | " + " | ".join(
                "-" if "needle_tier_shares" not in c else
                f"{c['tier_shares'].get('INT2', 0):.2f} / "
                f"{c['needle_tier_shares'].get('INT2', 0):.2f} / {c['needle_int2_runs']:.2f}"
                for c in cells) + " |")
    c = s["closing"]
    print(f"\nclosing over {c['tasks']}: {c['answer']}"
          + ("" if not c["missing"] else f"; missing {c['missing']}"))


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
