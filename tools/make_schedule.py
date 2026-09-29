#!/usr/bin/env python3
"""Make a schedule for the contention simulator from a pattern (#59).

    .venv/bin/python tools/make_schedule.py step --mib 300 --for 30 [--lead 5] --out s.json
    .venv/bin/python tools/make_schedule.py sawtooth --mib 512 --period 10 --cycles 6 --out s.json
    .venv/bin/python tools/make_schedule.py trapezoid --mib 400 --rise 0.5 --plateau 2 --fall 0.3
        [--count 5 --gap 10] [--lead 5] --out s.json
    .venv/bin/python tools/make_schedule.py poisson --length 600 --seed 1 --out s.json
        --rate 0.05 --mib 100 300 --rise 0.1 --plateau 0.5 2 --fall 0.2
    .venv/bin/python tools/make_schedule.py poisson --length 600 --seed 1 --out s.json
        --from-rq1 spikes|drops [--action vlc-2160p] [--kept 60 | --kept 30 120]

microinfer.patterns makes the schedule; tools/simulate_contention.py runs it.
Poisson arrivals take each of amplitude, rise, plateau and fall as one fixed
value or a range drawn uniformly, and a fixed rate; or all of them from
RQ1's spikes (recovered) or lasting drops, of one action if --action names
one. RQ1 cannot say how long a lasting drop was kept, so --kept gives it. A
schedule is the same for the same seed.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from microinfer import patterns  # noqa: E402

MiB = 2**20
SHAPE = ("mib", "rise", "plateau", "fall")


def distribution(values: list[float], scale: float = 1.0) -> patterns.Distribution:
    """One value is fixed; two are a uniform range."""
    if len(values) == 1:
        return patterns.Fixed(values[0] * scale)
    return patterns.Uniform(values[0] * scale, values[1] * scale)


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    kinds = parser.add_subparsers(dest="kind", required=True)
    step = kinds.add_parser("step")
    step.add_argument("--mib", type=float, required=True)
    step.add_argument("--for", dest="for_s", type=float, required=True, help="seconds taken")
    step.add_argument("--lead", type=float, default=0.0, help="seconds taking nothing first")
    saw = kinds.add_parser("sawtooth")
    saw.add_argument("--mib", type=float, required=True, help="the peak")
    saw.add_argument("--period", type=float, required=True, help="seconds")
    saw.add_argument("--cycles", type=int, required=True)
    saw.add_argument("--resolution", type=float, default=0.1, help="seconds per step of a climb")
    trap = kinds.add_parser("trapezoid")
    for name in SHAPE:
        trap.add_argument(f"--{name}", type=float, required=True,
                          help="MiB" if name == "mib" else "seconds")
    trap.add_argument("--count", type=int, default=1)
    trap.add_argument("--gap", type=float, default=0.0, help="seconds between one and the next")
    trap.add_argument("--lead", type=float, default=0.0)
    poisson = kinds.add_parser("poisson")
    poisson.add_argument("--length", type=float, required=True, help="seconds of arrivals")
    poisson.add_argument("--seed", type=int, required=True)
    poisson.add_argument("--rate", type=float, help="arrivals per second")
    for name in SHAPE:
        poisson.add_argument(f"--{name}", type=float, nargs="+",
                             help=("MiB" if name == "mib" else "seconds") + ": a value, or a range")
    poisson.add_argument("--from-rq1", choices=("spikes", "drops"),
                         help="draw from RQ1's recovered spikes, or its lasting drops")
    poisson.add_argument("--action", help="only this action's")
    poisson.add_argument("--kept", type=float, nargs="+",
                         help="seconds a lasting drop is kept: a value, or a range")
    for sub in (trap, poisson):
        sub.add_argument("--resolution", type=float, default=0.02,
                         help="seconds per sample of a ramp")
    for sub in (step, saw, trap, poisson):
        sub.add_argument("--out", type=Path, required=True, help="the schedule, JSON")
    args = parser.parse_args(argv)

    if args.kind == "step":
        schedule = patterns.step(args.mib * MiB, args.for_s, args.lead)
    elif args.kind == "sawtooth":
        schedule = patterns.sawtooth(args.mib * MiB, args.period, args.cycles, args.resolution)
    elif args.kind == "trapezoid":
        shape = patterns.Trapezoid(args.mib * MiB, args.rise, args.plateau, args.fall)
        schedule = patterns.trapezoids(shape, args.count, args.gap, args.lead, args.resolution)
    else:
        given = {name: getattr(args, name) for name in ("rate", *SHAPE)}
        for name in SHAPE + ("kept",):
            values = getattr(args, name)
            if values is not None and len(values) > 2:
                poisson.error(f"--{name} is one value or a range of two")
        if args.from_rq1:
            if any(v is not None for v in given.values()):
                poisson.error("--from-rq1 draws every parameter: drop --rate, --mib, --rise, "
                              "--plateau and --fall")
            if (args.from_rq1 == "drops") != (args.kept is not None):
                poisson.error("--kept goes with --from-rq1 drops, and only with it")
            params = patterns.from_rq1(args.from_rq1, args.action,
                                       kept_s=args.kept and distribution(args.kept))
        else:
            if any(v is None for v in given.values()):
                poisson.error("give --rate, --mib, --rise, --plateau and --fall, or --from-rq1")
            if args.action or args.kept:
                poisson.error("--action and --kept go with --from-rq1")
            params = patterns.ShapeParameters(
                rate_per_s=args.rate, amplitude_bytes=distribution(args.mib, MiB),
                rise_s=distribution(args.rise), plateau_s=distribution(args.plateau),
                fall_s=distribution(args.fall))
        schedule = patterns.poisson(params, args.length, args.seed, args.resolution)
    schedule.save(args.out)
    print(f"{args.out}: {len(schedule.points)} points over {schedule.duration_s:.1f} s, "
          f"peak {schedule.peak_bytes / MiB:.0f} MiB")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
