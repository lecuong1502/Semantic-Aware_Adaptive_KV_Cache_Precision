#!/usr/bin/env python3
"""Summarise RQ1's recordings, and the machine they were made on (#186).

    .venv/bin/python tools/summarise_contention.py --issue 186

Pools the spikes of every RQ1 recording in the benchmark log
(microinfer.contention.pooled): amplitude and rise time at P10, median, P90
and max, the duration of the recovered, the endings and how many name a
process. Logs them as a "contention-summary" entry naming each recording by
its entry's hash and its trace's, beside the host: the CPU, its cores and
threads, host memory and the CUDA toolkit the engine is built with. The GPU
and its driver are in every entry's environment already. Prints both as
tables for the thesis. Refuses a dirty tree.
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from microinfer import benchlog, contention  # noqa: E402


def host() -> dict:
    """The CPU's model name, its physical cores and its threads, host memory
    in bytes, and nvcc's release, None where it is not on the PATH."""
    cpuinfo = Path("/proc/cpuinfo").read_text()
    model = re.search(r"^model name\s*:\s*(.+)$", cpuinfo, re.M).group(1).strip()
    cores = {(m.group(1), m.group(2)) for m in re.finditer(
        r"^physical id\s*:\s*(\d+).*?^core id\s*:\s*(\d+)", cpuinfo, re.M | re.S)}
    kib = int(re.search(r"^MemTotal:\s*(\d+) kB", Path("/proc/meminfo").read_text(),
                        re.M).group(1))
    try:
        nvcc = subprocess.run(["nvcc", "--version"], capture_output=True, text=True,
                              check=True).stdout
        cuda = re.search(r"release ([\d.]+)", nvcc).group(1)
    except (OSError, subprocess.CalledProcessError, AttributeError):
        cuda = None
    return {"cpu": model, "cores": len(cores), "threads": os.cpu_count(),
            "memory_bytes": kib * 1024, "cuda_toolkit": cuda, "kernel": os.uname().release}


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--issue", type=int, required=True)
    parser.add_argument("--log", type=Path, default=benchlog.DEFAULT_LOG)
    args = parser.parse_args(argv)
    if benchlog.environment(args.log)["git_dirty"]:
        parser.error("tracked files have uncommitted changes; commit first, so that the "
                     "entry names the code that produced it")
    benchlog.verify(args.log)  # the hashes the summary names are of the entries as appended

    entries = list(contention.logged_recordings(args.log))
    if not entries:
        parser.error("no RQ1 recording in the log")
    spikes = contention.pooled(entries)
    machine = host()
    benchlog.append("contention-summary", model=None, context_length=None,
                    precision_tiers=None,
                    config={"issue": args.issue, "issues": list(contention.RQ1_ISSUES)},
                    results={"spikes": spikes, "host": machine}, log=args.log)

    print(f"{spikes['count']} spikes in {len(entries)} recordings; endings "
          f"{spikes['endings']}; {spikes['attributed']} attributed\n")
    print("| Measure | n | P10 | Median | P90 | Max |\n|---|---|---|---|---|---|")
    for name, key, scale in (("Amplitude (MiB)", "amplitude_mib", 1),
                             ("Rise time, 10% to 90% (s)", "rise_ms", 1e-3),
                             ("Duration, recovered spikes (s)", "duration_ms", 1e-3)):
        p = spikes[key]
        print(f"| {name} | {p['n']} | " + " | ".join(
            f"{p[k] * scale:.3g}" for k in ("p10", "median", "p90", "max")) + " |")
    print(f"\nCPU: {machine['cpu']}, {machine['cores']} cores, {machine['threads']} threads; "
          f"host memory {machine['memory_bytes'] / 2**30:.1f} GiB; "
          f"CUDA toolkit {machine['cuda_toolkit']}; kernel {machine['kernel']}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
