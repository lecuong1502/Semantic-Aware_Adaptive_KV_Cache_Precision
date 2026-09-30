#!/usr/bin/env python3
"""Check RQ1's traces against the sha256 their log entries record (#54, #57).

    .venv/bin/python tools/verify_release.py data/rq1/*.csv.gz

Each recording's contention-trace entry in the benchmark log (written by
tools/analyse_contention.py) records the sha256 of its trace, processes and
labels files. Before those files are published, and after they are
downloaded again, each must match the latest entry that names it: this says
which do, which do not, and which no entry names. Exits 1 unless every file
matches.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from microinfer import benchlog  # noqa: E402


def logged_hashes(log: Path) -> dict[str, str]:
    """The sha256 of every file a contention-trace entry names, by file
    name; where several entries name one, the latest."""
    hashes: dict[str, str] = {}
    for entry in benchlog.read(log):
        if entry["kind"] == "contention-trace":
            hashes.update(entry["config"].get("files", {}))
    return hashes


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("files", type=Path, nargs="+")
    parser.add_argument("--log", type=Path, default=benchlog.DEFAULT_LOG)
    args = parser.parse_args(argv)

    benchlog.verify(args.log)  # a chain that does not hold proves nothing
    hashes = logged_hashes(args.log)
    bad = 0
    for path in args.files:
        want = hashes.get(path.name)
        if want is None:
            print(f"UNLOGGED  {path}")
            bad += 1
        elif benchlog.file_sha256(path) != want:
            print(f"MISMATCH  {path}")
            bad += 1
        else:
            print(f"ok        {path}")
    print(f"{len(args.files) - bad} of {len(args.files)} match the log")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
