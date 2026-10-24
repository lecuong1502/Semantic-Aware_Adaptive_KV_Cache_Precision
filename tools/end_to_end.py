"""Run the needle task under simulated contention, and log each run (#157).

Two steps, each a command (microinfer.end_to_end):
1. By default: the screened needle set (#148), question-blind, under the
   semantic and the uniform source at 80%, the plan made by the monitor
   and the controller while the contention simulator takes memory part way
   through turn 1, by one rule for both. One rq3-end-to-end entry a run,
   its answer beside the bytes it reclaimed, with the device recorded to
   data/end_to_end/ and the recording's files named by their hashes.
2. `--summarise`: the end-to-end-summary entry, each source's accuracy and
   bytes, and semantic paired against uniform, and its table.

    .venv/bin/python tools/end_to_end.py --issue 157
    .venv/bin/python tools/end_to_end.py --issue 157 --summarise

A run already logged at the same code, for the model and set, is not run
again, so that a run stopped part way is taken up by the same command; a
recording an unlogged run left is replaced. Close every application first:
what the desktop takes or gives back during a run is contention too.
Refuses a dirty tree, as the checkpoint recorder does.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "tools"))

from perplexity import held_out_text  # noqa: E402
from rq3 import NEEDLE_SEED  # noqa: E402

from microinfer import Engine, benchlog, end_to_end, monitor, needles, recorder, replay, rq3  # noqa: E402,E501
from microinfer.footprint import MIB  # noqa: E402

OUT = REPO / "data" / "end_to_end"


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--issue", type=int, required=True,
                        help="the ticket these runs are made for")
    parser.add_argument("--model", default="qwen2.5-1.5b-instruct")
    parser.add_argument("--summarise", action="store_true",
                        help="log the summary of the screened set's runs")
    parser.add_argument("--sources", nargs="+", default=list(end_to_end.SOURCES),
                        choices=end_to_end.SOURCES)
    parser.add_argument("--samples", nargs="+", default=None,
                        help="sample ids of the screened set (default: all of them)")
    parser.add_argument("--needle-seed", type=int, default=NEEDLE_SEED)
    parser.add_argument("--haystack-tokens", type=int, default=needles.DEFAULT_HAYSTACK_TOKENS)
    parser.add_argument("--out", type=Path, default=OUT, help="where recordings are written")
    parser.add_argument("--log", type=Path, default=benchlog.DEFAULT_LOG)
    args = parser.parse_args(argv)
    of = rq3.NeedleSet(args.model, args.needle_seed, args.haystack_tokens)
    sources = list(dict.fromkeys(args.sources))

    if args.summarise:
        benchlog.verify(args.log)
        try:
            summary = end_to_end.summarise(benchlog.read(args.log), of, sources=sources)
        except ValueError as exc:
            parser.error(str(exc))
        entry = benchlog.append(end_to_end.SUMMARY_KIND, model=args.model, context_length=None,
                                precision_tiers=None,
                                config={**of.config(), "sources": sources, "issue": args.issue},
                                results=summary, log=args.log)
        print(table(summary))
        print(f"\nlogged {entry['sha256'][:12]}")
        return 0

    environment = benchlog.environment(args.log)
    if environment["git_dirty"]:
        parser.error("tracked files have uncommitted changes; commit first, so that "
                     "each entry names the code that produced it")
    kept, draws = rq3.screened(args.model, args.needle_seed, args.haystack_tokens, args.log)
    if args.samples is not None:
        unknown = set(args.samples) - set(kept)
        if unknown:
            parser.error(f"no samples {sorted(unknown)} in the screened set")
        kept = [s for s in kept if s in args.samples]
    done = end_to_end.logged(of, environment["git_commit"], args.log)
    todo = [(s, source) for s in kept for source in sources if (s, source) not in done]
    print(f"{len(kept) * len(sources) - len(todo)} of {len(kept) * len(sources)} runs logged "
          f"at this code already; {len(todo)} to run", flush=True)
    if not todo:
        return 0

    args.out.mkdir(parents=True, exist_ok=True)  # before minutes of loading
    # The simulator's context is taken as it starts, after the headroom the
    # take is set from has been read: measured once, before the engine loads.
    context_bytes = replay.simulator_context_bytes(args.out)
    engine = Engine(REPO / "models" / args.model)
    every = needles.needle_set(held_out_text(), engine.tokenizer, seed=args.needle_seed,
                               samples_per_depth=draws, haystack_tokens=args.haystack_tokens)
    by_id = {s.sample_id: s for s in every}
    engine.load_weights()
    start = time.perf_counter()
    for ran, (sample_id, source) in enumerate(todo, 1):
        sample = by_id[sample_id]
        out = args.out / f"{sample_id}-{source}.csv.gz"
        for left in replay.files(out) + [recorder.companion(out, ".pressure.jsonl")]:
            left.unlink(missing_ok=True)  # an unlogged run's, stopped part way
        results = end_to_end.run(engine, sample, source, out=out,
                                 simulator_context_bytes=context_bytes)
        pressure = recorder.companion(out, ".pressure.jsonl")
        pressure.write_text("".join(
            json.dumps({**asdict(r.event), "positions_held": r.positions_held,
                        "drained_ns": r.drained_ns}) + "\n" for r in engine.pressure_events))
        files = replay.files(out) + [pressure]
        benchlog.append(
            end_to_end.KIND, model=args.model, context_length=len(sample.turn_1),
            precision_tiers=results["at_question"]["tier_shares"],
            config=end_to_end.config(
                sample, source, of, issue=args.issue, simulator_context_bytes=context_bytes,
                poll_s=monitor.POLL_S, thresholds=asdict(monitor.DEFAULT),
                plan_budget_seconds=engine.kv_plan_budget_seconds,
                files={f.name: benchlog.file_sha256(f) for f in files if f.exists()}),
            results=results, log=args.log)
        at = results["at_question"]
        print(f"{ran}/{len(todo)} {sample_id} {source}: "
              f"{'right' if results['correct'] else 'WRONG'} {results['answer']!r}  "
              f"target {results['contention']['target_bytes'] / MIB:.0f} MiB, reclaimed "
              f"{at['reclaimed_bytes'] / MIB:.0f}, {at['effective_bits']:.2f} bits"
              + ("" if at["plans_done"] else ", a plan still going at the question")
              + f"  ({time.perf_counter() - start:.0f} s)", flush=True)
    return 0


def table(summary: dict) -> str:
    """The summary's sources and their pair, as markdown."""
    rows = [f"{summary['samples']} needle samples, {summary['experiment']}, under contention",
            "", "| Source | Correct | Wilson 95% | Effective bits | Target MiB | Reclaimed MiB "
            "| Reclaimed MiB, answered | Returned MiB | Target met | Plans done, question "
            "| Plans done, answered |",
            "|---|---|---|---|---|---|---|---|---|---|---|"]
    for source, s in summary["sources"].items():
        low, high = s["wilson_95"]
        rows.append(f"| {source} | {s['correct']}/{s['n']} | [{low:.2f}, {high:.2f}] "
                    f"| {s['effective_bits']:.2f} | {s['target_mib_mean']:.1f} "
                    f"| {s['reclaimed_mib_mean']:.1f} | {s['reclaimed_mib_mean_at_answer']:.1f} "
                    f"| {s['returned_mib_mean']:.1f} | {s['target_met']} "
                    f"| {s['plans_done_before_question']} | {s['plans_done_at_answer']} |")
    rows.append(f"\n{summary['same_contention']} of {summary['samples']} samples' runs met "
                "the same contention, the same target at the same position")
    for pair, p in summary["paired"].items():
        rows.append(f"\n{pair}: {p['semantic_only']} only semantic answered, {p['other_only']} "
                    f"only the other; McNemar p = {p['mcnemar_p']:.3g}, difference "
                    f"{p['difference']:+.2f}")
    return "\n".join(rows)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
