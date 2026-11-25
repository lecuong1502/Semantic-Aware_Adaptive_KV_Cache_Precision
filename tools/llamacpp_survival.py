#!/usr/bin/env python3
"""Play the survival experiment's contention against llama.cpp (#211).

    .venv/bin/python tools/llamacpp_survival.py --issue 211
        [--cache-types f16 q8_0 q4_0] [--scenarios S1 S5] [--runs 3]
        [--out-dir data/survival/llamacpp] [--dry-run]
    .venv/bin/python tools/llamacpp_survival.py --issue 211 --summarise

Each run starts a llama.cpp server of its own, Qwen2.5-1.5B's weights at
FP16 and its 32K cache at one ggml type, fills its cache to 16384 positions
of the survival experiment's prompt, has the contention simulator play S1's
pulse grid on its base or S5's step, and completes the prompt, 32704
positions, and 64 tokens (microinfer.llamacpp_survival). It is made by
tools/build_llamacpp.py, whose logged build every run is checked against.
Each run is one "llamacpp-survival" entry: how it ended and when, what the
contention asked for, was given and was refused, and the sha256 of its
trace, the simulator's schedule and events, and the server's log. A run
already logged at the same code and build is not made again, so the same
command takes it up; a run stopped part way leaves its recording, and the
run again records beside it, numbered.

S1's grid is about 494 s, through which the generation, about 12 s, is made
again and again; S5's run is one generation. A run of S1 takes about 9
minutes and one of S5 under 1: the 18 runs, about 1.5 GPU-hours.

--summarise logs one "llamacpp-summary" entry: each cell's endings, and
beside them MicroInfer's survival in S1 and S5 with what its contention was
refused, read from the events its runs' entries name, where data/survival
still holds them.

Close every application first, as for the survival experiment: what the
desktop takes or gives back moves the headroom the contention is set from.
Refuses a dirty tree.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from microinfer import benchlog, llamacpp, replay  # noqa: E402
from microinfer import llamacpp_survival as runs_of, survival_matrix as matrix  # noqa: E402
from microinfer import survival_summary  # noqa: E402
from microinfer.config import ModelConfig  # noqa: E402
from microinfer.footprint import MIB  # noqa: E402

#: The survival experiment's generation (#108).
MODEL = "qwen2.5-1.5b-instruct"
SEED, PROMPT_POSITIONS, NEW_TOKENS, CONTENTION_AT = 108, 32704, 64, 16384
CONTEXT = PROMPT_POSITIONS + NEW_TOKENS
#: How long the device is left between a server's end and the next's start.
SETTLE_S = 2.0
#: Where MicroInfer's recordings are kept: the matrix's, the experiment's.
MICROINFER_RECORDINGS = (REPO / "data" / "survival" / "matrix", REPO / "data" / "survival")


def out_for(out_dir: Path, cell: runs_of.Cell) -> Path:
    """The run's recording, numbered past any a run stopped part way left."""
    cache_type, scenario, run = cell
    stem = f"{scenario}-{cache_type}-r{run}"
    path, k = out_dir / f"{stem}.csv.gz", 2
    while path.exists():
        path, k = out_dir / f"{stem}.{k}.csv.gz", k + 1
    return path


def make(cell: runs_of.Cell, args: argparse.Namespace, build: dict, cfg: ModelConfig,
         prompt: np.ndarray, rq1: matrix.Rq1) -> dict:
    cache_type, scenario, run = cell
    out = out_for(args.out_dir, cell)
    log_path = out.with_name(out.name.removesuffix(".csv.gz") + ".server.log")
    contention = runs_of.Contention(scenario, run, rq1)
    context_bytes = replay.simulator_context_bytes(args.out_dir)
    with llamacpp.Server(llamacpp.BINARY, llamacpp.gguf_path(MODEL), context=CONTEXT,
                         cache_type=cache_type, log=log_path) as server:
        result = runs_of.run(
            server, cfg, prompt, NEW_TOKENS, contention_at=CONTENTION_AT, out=out,
            schedule=lambda headroom: contention.schedule(
                cfg, headroom, positions=CONTENTION_AT, context=CONTEXT - 1,
                simulator_context_bytes=context_bytes))
        returncode = server.returncode
    files = [*replay.files(out), log_path]
    played = result.schedule
    config = {"issue": args.issue, "engine": "llama.cpp", "cache_type": cache_type,
              "scenario": scenario, "run": run, "command": server.command,
              "build": {k: build[k] for k in ("version", "binary_sha256", "gguf_sha256")},
              "prompt_positions": PROMPT_POSITIONS, "new_tokens": NEW_TOKENS,
              "prompt_seed": SEED, "contention_at": CONTENTION_AT,
              "shortfall_mib": runs_of.SHORTFALL_BYTES / MIB if scenario == "S5" else None,
              "scenario_rule": None if scenario == "S5"
              else matrix.describe(scenario, run, rq1),
              "simulator_context_bytes": context_bytes,
              "contention_rule": "the headroom is counted as an FP16 MicroInfer's at the "
                                 "position, and the generation is made again from it until "
                                 "the schedule has been played",
              "files": {f.name: benchlog.file_sha256(f) for f in files if f.exists()}}
    results = {"survived": result.survived, "ending": result.ending,
               "returncode": returncode, "load_seconds": server.load_seconds,
               "buffers_mib": server.buffers_mib,
               "contention_position": result.contention_position,
               "headroom_at_contention_mib": result.headroom_bytes / MIB,
               "asked_mib": result.asked_bytes / MIB, "taken_mib": result.taken_bytes / MIB,
               "refused_mib": result.refused_bytes / MIB,
               "schedule": {"peak_mib": played.peak_bytes / MIB, "points": len(played.points),
                            "duration_s": played.duration_s, "played_s": result.played_s,
                            "cut": result.played_s < played.duration_s},
               "generations": result.generations}
    benchlog.append(runs_of.KIND, model=MODEL, context_length=CONTEXT - 1, precision_tiers=None,
                    config=config, results=results, log=args.log)
    return results


def summarise(args: argparse.Namespace) -> int:
    entries = benchlog.read(args.log)
    cells = runs_of.summarise(entries)
    if not cells:
        raise SystemExit("no llama.cpp run is logged")
    beside: dict[str, dict] = {}
    pooled = survival_summary.pool(entries)
    for scenario in cells:
        for condition in runs_of.BESIDE:
            made = pooled.get((condition, scenario), [])
            if made:
                beside.setdefault(scenario, {})[condition] = {
                    "survived": sum(bool(e["results"]["survived"]) for e in made),
                    "runs": len(made),
                    "refused_mib": [runs_of.refused_mib_of(e, MICROINFER_RECORDINGS)
                                    for e in made],
                    "entries": [e["sha256"] for e in made]}
    benchlog.append(runs_of.SUMMARY_KIND, model=MODEL, context_length=CONTEXT - 1,
                    precision_tiers=None, config={"issue": args.issue},
                    results={"llamacpp": cells, "microinfer": beside}, log=args.log)
    for scenario, by_type in cells.items():
        for name, cell in by_type.items():
            print(f"{scenario} llama.cpp {name:5} {cell['survived']}/{cell['runs']} survived, "
                  f"refused {max(cell['refused_mib']):.0f} MiB at most")
        for name, cell in beside.get(scenario, {}).items():
            most = [r for r in cell["refused_mib"] if r is not None]
            print(f"{scenario} MicroInfer {name:4} {cell['survived']}/{cell['runs']} survived, "
                  f"refused " + (f"{max(most):.0f} MiB at most" if most else "not recorded"))
    return 0


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--issue", type=int, required=True)
    parser.add_argument("--cache-types", nargs="+", choices=list(llamacpp.CACHE_TYPES),
                        default=list(llamacpp.CACHE_TYPES))
    parser.add_argument("--scenarios", nargs="+", choices=runs_of.SCENARIOS,
                        default=list(runs_of.SCENARIOS))
    parser.add_argument("--runs", type=int, default=runs_of.RUNS)
    parser.add_argument("--out-dir", type=Path, default=REPO / "data" / "survival" / "llamacpp")
    parser.add_argument("--dry-run", action="store_true",
                        help="print the runs still to make, and make none")
    parser.add_argument("--summarise", action="store_true")
    parser.add_argument("--log", type=Path, default=benchlog.DEFAULT_LOG)
    args = parser.parse_args(argv)
    if args.runs < 1:
        parser.error(f"--runs is at least 1; got {args.runs}")
    environment = benchlog.environment(args.log)
    if environment["git_dirty"] and not args.dry_run:
        parser.error("tracked files have uncommitted changes; commit first, so that each "
                     "entry names the code that produced it")
    if args.summarise:
        return summarise(args)
    try:
        build = llamacpp.logged_build(args.log, model=MODEL)
    except ValueError as exc:
        parser.error(str(exc))
    every = runs_of.runs(args.cache_types, args.scenarios, args.runs)
    done = runs_of.logged(environment["git_commit"], build, args.log)
    todo = [cell for cell in every if cell not in done]
    print(f"{len(every) - len(todo)} of {len(every)} runs logged at this code and build "
          f"already; {len(todo)} to run", flush=True)
    if args.dry_run:
        for cell in todo:
            print(*cell)
        return 0
    if not todo:
        return 0
    args.out_dir.mkdir(parents=True, exist_ok=True)
    cfg = ModelConfig.from_card(MODEL)
    prompt = np.random.default_rng(SEED).integers(
        1000, cfg.vocab_size - 1000, PROMPT_POSITIONS).astype(np.int32)
    rq1 = matrix.Rq1.read(args.log)
    start = time.perf_counter()
    for i, cell in enumerate(todo, 1):
        print(f"\n{i}/{len(todo)} {'-'.join(map(str, cell))}  "
              f"({(time.perf_counter() - start) / 3600:.1f} h)", flush=True)
        r = make(cell, args, build, cfg, prompt, rq1)
        print(json.dumps({k: r[k] for k in ("survived", "ending", "asked_mib", "taken_mib",
                                            "refused_mib")}), flush=True)
        time.sleep(SETTLE_S)
    print(f"\nthe runs are logged ({(time.perf_counter() - start) / 3600:.1f} h)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
