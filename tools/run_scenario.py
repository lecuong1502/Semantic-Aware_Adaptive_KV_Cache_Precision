#!/usr/bin/env python3
"""Run RQ1's scripted scenario while the recorder records it (#53).

    .venv/bin/python tools/run_scenario.py --out trace.csv.gz
        [--browser chrome|firefox] [--actions 1,2,3,4,6,7,8,9] [--with-engine]
        [--hold 60] [--gap 10] [--idle 120]

It starts the recorder (tools/record_contention.py) on --out, labels each
action's start and end through the recorder's FIFO, and runs the actions of
microinfer.scenario on a fixed schedule: by default the whole scenario in
Chrome, or with --browser firefox the Firefox pass over actions 2 to 4 that
#45 compares browsers with. With --with-engine it first starts the engine
holding Qwen2.5-1.5B at its 32K window (tools/hold_engine.py), labels its
prefill, and begins the actions once it decodes; the hold's status file
lands beside the trace.

Everything it starts, it stops: on SIGINT or SIGTERM the action under way is
cut short with its end label, every application it opened is closed, the
engine is stopped, and the recorder finishes its files. Its children run in
sessions of their own, so the terminal's Ctrl+C reaches only the driver,
which then stops them in order.

Beside the trace it writes trace.scenario.json: the configuration, each
step's times on the recorder's clock, what it opened (the browser's version,
the YouTube resolution asked for and the one played), any error, and the
engine's outcome. VLC plays a 4K clip from Blender's Big Buck Bunny, fetched
once into data/ and checked against a pinned hash.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
import urllib.request
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from microinfer import recorder, scenario  # noqa: E402

VIDEO_URL = ("https://download.blender.org/demo/movies/BBB/"
             "bbb_sunflower_2160p_30fps_normal.mp4.zip")
VIDEO_ZIP = REPO / "data" / "scenario" / "bbb_sunflower_2160p_30fps_normal.mp4.zip"
VIDEO_SHA256 = "750b255c6d9fee1e2a03a6716d4f358bca56e9115bf3e06a66162fc5272ae151"
VIDEO = VIDEO_ZIP.with_suffix("")  # the .mp4 inside
ENGINE_READY_TIMEOUT_S = 1800.0  # a 32K prefill on 1.5B takes minutes


def fetch_video() -> Path:
    """The 4K clip, fetched and checked the first time."""
    if not VIDEO.exists():
        if not VIDEO_ZIP.exists():
            VIDEO_ZIP.parent.mkdir(parents=True, exist_ok=True)
            print(f"fetching {VIDEO_URL} (632 MB)", flush=True)
            urllib.request.urlretrieve(VIDEO_URL, VIDEO_ZIP)
        h = hashlib.sha256()
        with open(VIDEO_ZIP, "rb") as f:
            for block in iter(lambda: f.read(1 << 20), b""):
                h.update(block)
        if h.hexdigest() != VIDEO_SHA256:
            raise SystemExit(f"{VIDEO_ZIP} has sha256 {h.hexdigest()}, not {VIDEO_SHA256}; "
                             "delete it to fetch again")
        with zipfile.ZipFile(VIDEO_ZIP) as z:
            z.extract(VIDEO.name, VIDEO.parent)
    return VIDEO


def child(argv: list[str]) -> subprocess.Popen:
    return subprocess.Popen(argv, start_new_session=True, stdout=subprocess.DEVNULL)


def stop_child(proc: subprocess.Popen, sig: int, timeout: float = 60.0) -> int:
    if proc.poll() is None:
        proc.send_signal(sig)
        try:
            proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
    return proc.returncode


def read_status(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, required=True, help="the trace to record")
    parser.add_argument("--browser", choices=sorted(scenario.DEFAULT_ACTIONS), default="chrome")
    parser.add_argument("--actions", help="the actions of #45 to run, by number, in order; "
                                          "by default all but 5 in Chrome, 2 to 4 and 9 in "
                                          "Firefox")
    parser.add_argument("--with-engine", action="store_true",
                        help="hold the engine at its 32K window through the scenario")
    parser.add_argument("--model", default="qwen2.5-1.5b-instruct")
    parser.add_argument("--hold", type=float, default=scenario.Timing.hold_s)
    parser.add_argument("--gap", type=float, default=scenario.Timing.gap_s)
    parser.add_argument("--idle", type=float, default=scenario.Timing.idle_s)
    parser.add_argument("--video", type=Path, help="the 4K file VLC plays; Big Buck Bunny, "
                                                   "fetched once, by default")
    parser.add_argument("--youtube-video", default=scenario.YOUTUBE_VIDEO)
    args = parser.parse_args(argv)

    actions = (tuple(int(a) for a in args.actions.split(",")) if args.actions
               else scenario.DEFAULT_ACTIONS[args.browser])
    timing = scenario.Timing(hold_s=args.hold, gap_s=args.gap, idle_s=args.idle)
    video = args.video
    if 7 in actions and video is None:
        video = fetch_video()

    stop = threading.Event()
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: stop.set())

    fifo = Path(tempfile.mkdtemp(prefix="microinfer-scenario-")) / "labels.fifo"
    rec = child([sys.executable, str(REPO / "tools" / "record_contention.py"),
                 "--out", str(args.out), "--labels", str(fifo)])
    engine = None
    status_path = args.out.with_name(scenario.log_path(args.out).name.replace(
        ".scenario.json", ".hold.json"))
    log: dict = {"config": {"out": str(args.out), "browser": args.browser,
                            "actions": list(actions), "timing": vars(timing),
                            "with_engine": args.with_engine, "model": args.model,
                            "video": None if video is None else str(video),
                            "youtube_video": args.youtube_video},
                 "t_mono_ns": time.monotonic_ns(), "t_wall": time.time(), "steps": []}

    def send(event: str, action: str) -> None:
        recorder.send_label(fifo, event, action, wait=10.0)

    try:
        if args.with_engine:
            engine = child([sys.executable, str(REPO / "tools" / "hold_engine.py"),
                            "--model", args.model, "--status", str(status_path)])
            send(recorder.START, "engine-prefill")
            end = time.monotonic() + ENGINE_READY_TIMEOUT_S
            status = None
            while not stop.is_set() and engine.poll() is None and time.monotonic() < end:
                status = read_status(status_path)
                if status is not None and status["state"] == "decoding":
                    break
                stop.wait(0.5)
            send(recorder.END, "engine-prefill")
            log["engine_ready"] = status
            if status is None or status["state"] != "decoding":
                raise SystemExit(f"the engine did not start decoding: {status}")
        desktop = scenario.Desktop(args.browser, video, args.youtube_video)
        steps = scenario.steps_for(actions, desktop, timing)
        log["steps"] = scenario.run(steps, desktop, send, stop, timing.gap_s)
    finally:
        log["interrupted"] = stop.is_set()
        if engine is not None:
            log["engine"] = {"exit_code": stop_child(engine, signal.SIGTERM),
                             "status": read_status(status_path)}
        log["recorder_exit_code"] = stop_child(rec, signal.SIGINT)
        scenario.log_path(args.out).write_text(json.dumps(log, indent=1) + "\n")
        shutil.rmtree(fifo.parent, ignore_errors=True)  # the recorder removed its FIFO
    print(f"{len(log['steps'])} steps recorded in {args.out}"
          + (", interrupted" if log["interrupted"] else ""), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
