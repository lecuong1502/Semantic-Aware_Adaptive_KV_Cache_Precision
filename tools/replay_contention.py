#!/usr/bin/env python3
"""Replay an RQ1 recording on the contention simulator (#60).

    .venv/bin/python tools/replay_contention.py convert --recording data/rq1/x.csv.gz
        [--from 120 --to 300] [--engine-pid P] --out schedule.json
    .venv/bin/python tools/replay_contention.py run --recording data/rq1/x.csv.gz
        [--from 120 --to 300] [--engine-pid P] --out replay.csv.gz --issue 60 [--lead 3]

microinfer.replay makes the schedule: the device's used memory at 50 Hz, less
the engine's where the recording ran beside it (its pid from the hold's
status beside the recording, or --engine-pid), above its least over the
window from --from to --to seconds after the recording's first sample.

`convert` writes that schedule, for tools/simulate_contention.py.

`run` replays it and measures the replay. The recorder records the device to
--out, as RQ1 did; the simulator, a process of its own, takes nothing for
--lead seconds and then follows the schedule. The replay's recording is
compared with the original, sample by sample and spike by spike, against
the bounds microinfer.replay states; from the replay's processes stream, it
also says how far the simulator itself strayed and how far the rest of the
desktop moved. The comparison is logged to the benchmark log with the sha256
of the recording replayed and of every file the replay wrote, and how far
the engine's subtraction strays from the processes stream's own sum
(microinfer.replay.engine_residual).

The original's spikes are sought from a baseline window (5 s) into the
window replayed, so start --from 5 s before the first spike to compare; it
warns of any spike it cannot. Whatever the desktop itself does during the
replay counts as error, so replay a few minutes around the actions of
interest rather than a whole recording.

Run it on an idle desktop: close every application first, as for RQ1's
scenarios (docs/rq1-protocol.md). What else was on the GPU is in the entry.
Refuses a dirty tree, and an --out that exists.
"""

from __future__ import annotations

import argparse
import json
import signal
import sys
import threading
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from microinfer import benchlog, recorder, replay, simulator, spikes  # noqa: E402


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    modes = parser.add_subparsers(dest="mode", required=True)
    convert = modes.add_parser("convert")
    run = modes.add_parser("run")
    for sub in (convert, run):
        sub.add_argument("--recording", type=Path, required=True, help="an RQ1 device trace")
        sub.add_argument("--from", dest="start_s", type=float, default=0.0,
                         help="seconds after the recording's first sample")
        sub.add_argument("--to", dest="end_s", type=float, help="the same; its end if omitted")
        sub.add_argument("--engine-pid", type=int,
                         help="the engine's pid, if the hold's status is not beside it")
    convert.add_argument("--out", type=Path, required=True, help="the schedule, JSON")
    run.add_argument("--out", type=Path, required=True, help="the replay's recording")
    run.add_argument("--issue", type=int, required=True, help="the ticket the replay is logged for")
    run.add_argument("--lead", type=float, default=replay.LEAD_S,
                     help="seconds of idle desktop recorded before the schedule")
    run.add_argument("--log", type=Path, default=benchlog.DEFAULT_LOG)
    args = parser.parse_args(argv)

    window = {"start_s": args.start_s, "end_s": args.end_s, "engine_pid": args.engine_pid}
    if args.mode == "convert":
        schedule = replay.from_recording(args.recording, **window)
        schedule.save(args.out)
        print(f"{args.out}: {len(schedule.points)} points over {schedule.duration_s:.1f} s, "
              f"peak {schedule.peak_bytes / 2**20:.0f} MiB")
        return 0

    if args.lead < replay.IDLE_S:
        run.error(f"--lead is at least {replay.IDLE_S} s, the idle level's measure")
    if args.out.exists():
        run.error(f"{args.out} exists; a replay does not overwrite a recording")
    if benchlog.environment(args.log)["git_dirty"]:
        run.error("tracked files have uncommitted changes; commit first, so that the entry "
                  "names the code that produced it")
    original = replay.held(args.recording, **window)
    schedule = replay.to_schedule(original)
    whole = replay.held(args.recording, engine_pid=args.engine_pid)
    hidden = [round(x.start_ns / 1e9 - args.start_s, 2)
              for x in spikes.find_spikes((whole.t_s * 1e9).astype("int64"), -whole.bytes)
              if 0 <= x.start_ns / 1e9 - args.start_s < spikes.DEFAULT_WINDOW_S]
    if hidden:
        print(f"warning: spikes begin {hidden} s into the window, before the "
              f"{spikes.DEFAULT_WINDOW_S:g} s baseline window the comparison needs; start "
              f"--from earlier to compare them", file=sys.stderr, flush=True)

    stop = threading.Event()
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: stop.set())
    print(f"replaying {schedule.duration_s:.1f} s after a {args.lead:g} s lead", flush=True)
    started = replay.replay(schedule, args.out, lead_s=args.lead, stop=stop)

    _, samples = recorder.read(args.out)
    _, states, procs = recorder.read_processes(recorder.processes_path(args.out))
    pid = json.loads(recorder.companion(args.out, ".replay.json").read_text())["simulator_pid"]
    results = replay.compare(original, samples, started, processes=(states, procs),
                             simulator_pid=pid)
    events_path = recorder.companion(args.out, ".events.jsonl")
    results["simulator"] = simulator.summarise(
        [json.loads(line) for line in events_path.read_text().splitlines()])
    results["engine_residual_mib"] = replay.engine_residual(args.recording, **window)
    results["hidden_spikes_s"] = hidden
    files = replay.files(args.out)
    benchlog.append("contention-replay", model=None, context_length=None, precision_tiers=None,
                    config={"issue": args.issue, "recording": args.recording.name,
                            "recording_sha256": benchlog.file_sha256(args.recording),
                            "from_s": args.start_s, "to_s": args.end_s,
                            "engine_pid": original.engine_pid, "lead_s": args.lead,
                            "idle_s": replay.IDLE_S,
                            "files": {f.name: benchlog.file_sha256(f)
                                      for f in files if f.exists()}},
                    results=results, log=args.log)

    error, found = results["error_mib"], results["spikes"]
    def p90(key: str) -> str:
        value = results.get(key)
        return "-" if value is None else f"{value['p90']:.1f}"

    print(f"error {error['median']:.1f} MiB median, {error['p90']:.1f} P90; "
          f"{found['original']} spikes, {found['unmeasured']} unmeasured; the simulator "
          f"{p90('simulator_error_mib')} MiB P90 off its schedule, the desktop moved "
          f"{p90('desktop_moved_mib')} MiB P90; within: "
          + ", ".join(f"{k} {'yes' if v else 'NO'}" for k, v in results["within"].items()))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
