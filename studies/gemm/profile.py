#!/usr/bin/env python3
"""Run the GEMM study and record what it measured (#11).

    python studies/gemm/profile.py time         # event timings; needs no privilege
    python studies/gemm/profile.py ncu-command  # prints the one command that needs sudo
    python studies/gemm/profile.py summarize studies/gemm/results/<name>.ncu-rep

The shapes are every distinct projection of both models, read from the model
cards, at one row (decode) and PREFILL_ROWS rows (prefill). v_proj repeats
k_proj, up_proj repeats gate_proj and o_proj repeats q_proj, so each is
measured once.

Performance counters need root on this machine (`RmProfilingAdminOnly: 1`), so
`ncu` is the one step run with sudo, and it only writes a report. Reading the
report back (`ncu --import`) needs no privilege, so everything else happens
here as the user.

Results go to `results/` as JSON with the git commit, the hardware and whether
anything else held the GPU. That is the form #13's benchmark log asks each
entry to carry. #13 does not exist yet; when it does, these records should be
migrated into it rather than left beside it.

This script imports the engine's config reader and nothing else of the engine.
The dependency runs from the study to the engine, never back
(tests/test_studies_isolation.py).
"""

from __future__ import annotations

import csv
import io
import json
import os
import re
import subprocess
import sys
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
STUDY = Path(__file__).resolve().parent
BUILD = REPO / "build" / "studies-gemm"
BENCH = BUILD / "gemm_bench"
RESULTS = STUDY / "results"

sys.path.insert(0, str(REPO / "src"))
from microinfer.config import ModelConfig  # noqa: E402
from microinfer.models import VERIFIED  # noqa: E402

#: A prefill of 512 tokens: long enough that the GEMM is compute-shaped, short
#: enough that the naive kernel finishes under ncu's replays in minutes.
PREFILL_ROWS = 512
DECODE_ROWS = 1

#: Kernel names of the two hand-written stages; anything else in a call is
#: cuBLAS's.
STUDY_KERNELS = {"naive_kernel": "naive", "tiled_kernel": "tiled"}

#: Raw ncu metrics, by the name the write-up uses.
METRICS = {
    "duration_ns": "gpu__time_duration.sum",
    "achieved_occupancy_pct": "sm__warps_active.avg.pct_of_peak_sustained_active",
    "theoretical_occupancy_pct": "sm__maximum_warps_per_active_cycle_pct",
    "compute_throughput_pct": "sm__throughput.avg.pct_of_peak_sustained_elapsed",
    "memory_throughput_pct": "gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed",
    "dram_throughput_pct": "gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed",
    "registers_per_thread": "launch__registers_per_thread",
}
STALL = re.compile(r"^smsp__average_warps_issue_stalled_(\w+?)_per_issue_active\.ratio$")
SECTIONS = ["SpeedOfLight", "Occupancy", "LaunchStats", "WarpStateStats", "MemoryWorkloadAnalysis"]


def shapes() -> list[tuple[str, str, int, int, int]]:
    """(model, projection, rows, in_features, out_features), card-derived."""
    out = []
    for name in sorted(VERIFIED):
        cfg = ModelConfig.from_card(name)
        kv = cfg.num_key_value_heads * cfg.head_dim
        projections = {
            "q_proj": (cfg.hidden_size, cfg.hidden_size),
            "k_proj": (cfg.hidden_size, kv),
            "gate_proj": (cfg.hidden_size, cfg.intermediate_size),
            "down_proj": (cfg.intermediate_size, cfg.hidden_size),
        }
        for rows in (DECODE_ROWS, PREFILL_ROWS):
            for proj, (k, n) in projections.items():
                out.append((name, proj, rows, k, n))
    return out


def shape_args() -> list[str]:
    return [f"{rows},{k},{n}" for _, _, rows, k, n in shapes()]


def build() -> None:
    if not (BUILD / "CMakeCache.txt").is_file():
        pybind11_dir = subprocess.run([sys.executable, "-m", "pybind11", "--cmakedir"],
                                      capture_output=True, text=True, check=True).stdout.strip()
        subprocess.run(["cmake", "-S", STUDY, "-B", BUILD, f"-Dpybind11_DIR={pybind11_dir}",
                        f"-DPython_EXECUTABLE={sys.executable}"], check=True)
    subprocess.run(["cmake", "--build", BUILD, "-j", str(os.cpu_count() or 1)], check=True)


def git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=REPO, capture_output=True, text=True,
                          check=True).stdout.strip()


def environment() -> dict:
    """Everything a number needs to be defended later (#13)."""
    smi = ET.fromstring(subprocess.run(["nvidia-smi", "-q", "-x"], capture_output=True,
                                       text=True, check=True).stdout)
    gpu = smi.find("gpu")
    processes = gpu.find("processes")
    others = [
        {"pid": int(p.findtext("pid")), "name": p.findtext("process_name"),
         "type": p.findtext("type"), "used_memory": p.findtext("used_memory")}
        for p in (processes if processes is not None else [])
        if int(p.findtext("pid")) != os.getpid()
    ]
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_commit": git("rev-parse", "HEAD"),
        "git_dirty": bool(git("status", "--porcelain", "--", "studies", "src")),
        "gpu": gpu.findtext("product_name"),
        "gpu_total_memory": gpu.findtext("fb_memory_usage/total"),
        "driver_version": smi.findtext("driver_version"),
        "cuda_version": smi.findtext("cuda_version"),
        # Not exclusive on a desktop: the display server, a browser and an
        # editor share the GPU. Recorded rather than pretended away.
        "exclusive_gpu": not others,
        "other_gpu_processes": others,
        "models": sorted(VERIFIED),
        "context_length": None,
        "precision_tiers": None,
    }


def time_shapes(repeat: int) -> list[dict]:
    out = subprocess.run([BENCH, "--repeat", str(repeat), *shape_args()],
                         capture_output=True, text=True, check=True).stdout
    return list(csv.DictReader(io.StringIO(out)))


def write_record(kind: str, record: dict) -> Path:
    RESULTS.mkdir(exist_ok=True)
    stamp = record["environment"]["timestamp"].replace(":", "").replace("+0000", "Z")
    path = RESULTS / f"{stamp}-{record['environment']['git_commit'][:7]}-{kind}.json"
    path.write_text(json.dumps(record, indent=2) + "\n")
    return path


def cmd_time() -> None:
    build()
    timings = time_shapes(repeat=50)
    labels = {(r, k, n): (m, p) for m, p, r, k, n in shapes()}
    for t in timings:
        t["model"], t["projection"] = labels[(int(t["rows"]), int(t["in_features"]),
                                              int(t["out_features"]))]
    path = write_record("timing", {"environment": environment(), "timings": timings})
    print(f"wrote {path.relative_to(REPO)}")


def ncu_command(report: Path) -> list[str]:
    ncu = subprocess.run(["which", "ncu"], capture_output=True, text=True).stdout.strip() or "ncu"
    sections = [arg for s in SECTIONS for arg in ("--section", s)]
    return [ncu, "--profile-from-start", "off", *sections, "-f", "-o", str(report),
            str(BENCH), "--repeat", "1", *shape_args()]


def cmd_ncu_command() -> None:
    build()
    RESULTS.mkdir(exist_ok=True)
    report = RESULTS / f"gemm-{git('rev-parse', '--short', 'HEAD')}"
    print("sudo " + " ".join(ncu_command(report)))
    print(f"\nthen: {sys.executable} {Path(__file__).relative_to(REPO)} summarize {report}.ncu-rep")


def read_report(report: Path) -> list[dict]:
    """One dict per kernel launch, in launch order, from ncu's raw page.

    The raw CSV has a header row, then a row of units, then one row per kernel.
    """
    out = subprocess.run(["ncu", "--import", str(report), "--csv", "--page", "raw"],
                         capture_output=True, text=True, check=True).stdout
    rows = list(csv.reader(io.StringIO(out)))
    header, data = rows[0], rows[2:]
    return [dict(zip(header, row)) for row in data]


def number(value: str) -> float | None:
    try:
        return float(value.replace(",", ""))
    except ValueError:
        return None


def attribute(kernels: list[dict]) -> list[dict]:
    """Group launches into (shape, implementation) calls.

    Launch order is fixed by the bench: for each shape, naive, then tiled, then
    however many kernels one cuBLAS call launches. So a study kernel starts a
    call, and anything else belongs to the cuBLAS call after the tiled one.
    """
    calls = []
    order = iter(shapes())
    for k in kernels:
        name = k["Kernel Name"]
        impl = next((v for key, v in STUDY_KERNELS.items() if name.startswith(key)), "cublas")
        if impl == "naive":
            model, proj, rows, kf, nf = next(order)
            shape = {"model": model, "projection": proj, "rows": rows,
                     "in_features": kf, "out_features": nf}
        if impl == "cublas" and calls and calls[-1]["implementation"] == "cublas":
            calls[-1]["kernels"].append(k)
            continue
        calls.append({**shape, "implementation": impl, "kernels": [k]})
    return calls


def summarise_call(call: dict) -> dict:
    """Durations add up across a call's kernels; every other metric is read
    from its longest kernel, which is the GEMM itself."""
    kernels = call.pop("kernels")
    main = max(kernels, key=lambda k: number(k[METRICS["duration_ns"]]) or 0)
    out = dict(call)
    out["kernel_names"] = [k["Kernel Name"] for k in kernels]
    out["block_size"], out["grid_size"] = main.get("Block Size"), main.get("Grid Size")
    for key, metric in METRICS.items():
        out[key] = number(main.get(metric, ""))
    out["duration_ns"] = sum(number(k[METRICS["duration_ns"]]) or 0 for k in kernels)
    stalls = {m.group(1): number(v) for col, v in main.items()
              if (m := STALL.match(col)) and number(v) is not None}
    out["stalls_per_issue"] = dict(sorted(stalls.items(), key=lambda kv: -kv[1]))
    return out


def cmd_summarize(report: Path) -> None:
    calls = [summarise_call(c) for c in attribute(read_report(report))]
    expected = 3 * len(shapes())
    if len(calls) != expected:
        raise SystemExit(f"{report}: {len(calls)} calls attributed, {expected} expected")
    record = {"environment": environment(), "ncu_report": report.name,
              "ncu_version": subprocess.run(["ncu", "--version"], capture_output=True,
                                            text=True).stdout.strip().splitlines()[-1],
              "clock_control": "ncu default (base clocks locked)", "calls": calls}
    path = write_record("ncu", record)
    print(f"wrote {path.relative_to(REPO)}")
    for c in calls:
        top = ", ".join(f"{k} {v:.1f}" for k, v in list(c["stalls_per_issue"].items())[:3])
        print(f"{c['model'][:13]:13} {c['projection']:9} {c['rows']:4} {c['implementation']:6} "
              f"{c['duration_ns'] / 1000:10.1f} us  occ {c['achieved_occupancy_pct'] or 0:5.1f}%  "
              f"mem {c['memory_throughput_pct'] or 0:5.1f}%  sm {c['compute_throughput_pct'] or 0:5.1f}%  "
              f"stalls: {top}")


def main(argv: list[str]) -> int:
    if argv[:1] == ["time"]:
        cmd_time()
    elif argv[:1] == ["ncu-command"]:
        cmd_ncu_command()
    elif argv[:1] == ["summarize"] and len(argv) == 2:
        cmd_summarize(Path(argv[1]))
    else:
        print(__doc__)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
