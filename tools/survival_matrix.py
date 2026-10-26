#!/usr/bin/env python3
"""Run the survival matrix (#190): B1 to B6 against S1 to S4, three runs a cell.

    .venv/bin/python tools/survival_matrix.py --issue 191
        [--model qwen2.5-1.5b-instruct] [--runs 3] [--scenarios S1 S2 S3 S4]
        [--conditions B1 B2 B3 B4 B5 B6] [--out-dir data/survival/matrix] [--dry-run]

Each run of the matrix is one run of tools/survival_experiment.py, a
process of its own: Qwen2.5-1.5B prefills 32704 positions and decodes 64,
and when its cache reaches 16384 positions the simulator plays the
scenario on its base (microinfer.survival_matrix). B1 to B3 are static
engines at FP16, INT4 and INT2; B4 to B6 the adaptive engine from the
uniform, random and semantic sources. The runs go run by run, every cell's
first before any cell's second, so that a matrix stopped part way is
balanced; a run already logged at the same code is not run again, so the
same command takes it up. A run stopped part way leaves its recording,
which is kept; the run again records beside it, numbered.

The S5 cells the survival experiment has not run, B2, B3 and B5, are the
same command with `--scenarios S5 --runs 1 --conditions B2 B3 B5`.

On the RTX 4050 a run takes about 15 to 20 minutes for a static engine and
about 30 for an adaptive one under pressure, its prefill chunks sized by
time (#135): the 72 runs of the matrix take about 30 GPU-hours.

Close every application first, as for the survival experiment: what the
desktop takes or gives back moves the headroom the base is set from.
Refuses a dirty tree; a run that fails stops the matrix, with its command.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from microinfer import benchlog, survival_matrix as matrix  # noqa: E402

EXPERIMENT = REPO / "tools" / "survival_experiment.py"


def out_for(out_dir: Path, cell: matrix.Cell) -> Path:
    """The cell's recording, numbered past any a run stopped part way left."""
    path, k = out_dir / f"{cell.name}.csv.gz", 2
    while path.exists():
        path, k = out_dir / f"{cell.name}.{k}.csv.gz", k + 1
    return path


def command(cell: matrix.Cell, out: Path, args: argparse.Namespace) -> list[str]:
    """The survival experiment's command for one run of the matrix."""
    condition = matrix.CONDITIONS[cell.condition]
    engine = (["adaptive", "--score-source", condition.score_source] if condition.adaptive
              else ["static", "--kv-tier", condition.kv_tier])
    return [sys.executable, str(EXPERIMENT), *engine, "--scenario", cell.scenario,
            "--run", str(cell.run), "--out", str(out), "--issue", str(args.issue),
            "--model", args.model, "--log", str(args.log)]


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--issue", type=int, required=True)
    parser.add_argument("--model", default="qwen2.5-1.5b-instruct")
    parser.add_argument("--runs", type=int, default=matrix.RUNS)
    parser.add_argument("--scenarios", nargs="+", choices=matrix.SCENARIOS,
                        default=list(matrix.SCHEDULED))
    parser.add_argument("--conditions", nargs="+", choices=list(matrix.CONDITIONS),
                        default=list(matrix.CONDITIONS))
    parser.add_argument("--out-dir", type=Path, default=REPO / "data" / "survival" / "matrix")
    parser.add_argument("--dry-run", action="store_true",
                        help="print the runs still to make, and make none")
    parser.add_argument("--log", type=Path, default=benchlog.DEFAULT_LOG)
    args = parser.parse_args(argv)
    if args.runs < 1:
        parser.error(f"--runs is at least 1; got {args.runs}")
    environment = benchlog.environment(args.log)
    if environment["git_dirty"] and not args.dry_run:
        parser.error("tracked files have uncommitted changes; commit first, so that each "
                     "entry names the code that produced it")

    cells = matrix.cells(args.conditions, args.scenarios, args.runs)
    done = matrix.logged(environment["git_commit"], args.model, args.log)
    todo = [c for c in cells if c not in done]
    print(f"{len(cells) - len(todo)} of {len(cells)} runs logged at this code already; "
          f"{len(todo)} to run", flush=True)
    if args.dry_run:
        for cell in todo:
            print(" ".join(command(cell, out_for(args.out_dir, cell), args)))
        return 0
    start = time.perf_counter()
    for i, cell in enumerate(todo, 1):
        cmd = command(cell, out_for(args.out_dir, cell), args)
        print(f"\n{i}/{len(todo)} {cell.name}  ({(time.perf_counter() - start) / 3600:.1f} h)",
              flush=True)
        if subprocess.run(cmd, cwd=REPO).returncode != 0:
            print(f"{cell.name} failed; the matrix stops here:\n  {' '.join(cmd)}",
                  file=sys.stderr)
            return 1
    print(f"\nthe matrix is logged ({(time.perf_counter() - start) / 3600:.1f} h)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
