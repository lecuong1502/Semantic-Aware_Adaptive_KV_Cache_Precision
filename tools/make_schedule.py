#!/usr/bin/env python3
"""Make a schedule for the contention simulator from a pattern (#59).

    .venv/bin/python tools/make_schedule.py step --mib 300 --hold 30 [--lead 5] --out s.json
    .venv/bin/python tools/make_schedule.py sawtooth --mib 512 --period 10 --cycles 6 --out s.json
    .venv/bin/python tools/make_schedule.py poisson --duration 600 --seed 1 --out s.json
        [--rate 0.05 --mib 100 300 --rise 0.1 --plateau 0.5 2 --fall 0.2]
        [--from-rq1 [--issues 55 56] [--action vlc-2160p]]

microinfer.patterns makes the schedule; tools/simulate_contention.py runs it.
Poisson spikes take each of rate, amplitude, rise, plateau and fall as one
fixed value or a range drawn uniformly, or, with --from-rq1, all of them
from RQ1's spikes in the benchmark log, of one action if --action names
one. A schedule is the same for the same seed.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from microinfer import benchlog, patterns  # noqa: E402

MiB = 2**20


def distribution(values: list[float], scale: float = 1.0):
    """One value is fixed; two are a uniform range."""
    if len(values) == 1:
        return patterns.Fixed(values[0] * scale)
    low, high = values
    return patterns.Uniform(low * scale, high * scale)


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    kinds = parser.add_subparsers(dest="kind", required=True)
    step = kinds.add_parser("step")
    step.add_argument("--mib", type=float, required=True)
    step.add_argument("--hold", type=float, required=True, help="seconds")
    step.add_argument("--lead", type=float, default=0.0, help="seconds taking nothing first")
    saw = kinds.add_parser("sawtooth")
    saw.add_argument("--mib", type=float, required=True, help="the peak")
    saw.add_argument("--period", type=float, required=True, help="seconds")
    saw.add_argument("--cycles", type=int, required=True)
    saw.add_argument("--resolution", type=float, default=0.1, help="seconds per step of a climb")
    poisson = kinds.add_parser("poisson")
    poisson.add_argument("--duration", type=float, required=True, help="seconds of arrivals")
    poisson.add_argument("--seed", type=int, required=True)
    poisson.add_argument("--rate", type=float, help="spikes per second")
    poisson.add_argument("--mib", type=float, nargs="+", help="amplitude: a value, or a range")
    poisson.add_argument("--rise", type=float, nargs="+", help="seconds, or a range")
    poisson.add_argument("--plateau", type=float, nargs="+", help="seconds, or a range")
    poisson.add_argument("--fall", type=float, nargs="+", help="seconds, or a range")
    poisson.add_argument("--from-rq1", action="store_true",
                         help="draw every parameter from RQ1's spikes in the benchmark log")
    poisson.add_argument("--issues", type=int, nargs="+", default=[55, 56])
    poisson.add_argument("--action", help="only this action's spikes")
    poisson.add_argument("--log", type=Path, default=benchlog.DEFAULT_LOG)
    poisson.add_argument("--resolution", type=float, default=0.02)
    for sub in (step, saw, poisson):
        sub.add_argument("--out", type=Path, required=True, help="the schedule, JSON")
    args = parser.parse_args(argv)

    if args.kind == "step":
        schedule = patterns.step(args.mib * MiB, args.hold, args.lead)
    elif args.kind == "sawtooth":
        schedule = patterns.sawtooth(args.mib * MiB, args.period, args.cycles, args.resolution)
    else:
        if args.from_rq1:
            params = patterns.from_rq1(args.log, tuple(args.issues), args.action)
        else:
            given = [args.rate, args.mib, args.rise, args.plateau, args.fall]
            if any(v is None for v in given):
                poisson.error("give --rate, --mib, --rise, --plateau and --fall, or --from-rq1")
            params = patterns.SpikeParameters(
                rate_per_s=args.rate, amplitude_bytes=distribution(args.mib, MiB),
                rise_s=distribution(args.rise), duration_s=distribution(args.plateau),
                fall_s=distribution(args.fall))
        schedule = patterns.poisson_spikes(params, args.duration, args.seed, args.resolution)
    schedule.save(args.out)
    print(f"{args.out}: {len(schedule.points)} points over {schedule.duration_s:.1f} s, "
          f"peak {schedule.peak_bytes / MiB:.0f} MiB")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
