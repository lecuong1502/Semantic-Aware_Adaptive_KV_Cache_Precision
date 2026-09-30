#!/usr/bin/env python3
"""Run RQ1's scripted scenario while the recorder records it (#53).

    .venv/bin/python tools/run_scenario.py --out trace.csv.gz
        [--browser chrome|firefox] [--actions 1,2,3,4,6,7,8,9] [--with-engine]
        [--hold 60] [--gap 10] [--idle 120] [--call-url URL] [--call-timeout 600]

The collection protocol, what to prepare and how many runs of which kind,
is docs/rq1-protocol.md.

It starts the recorder (tools/record_contention.py) on --out, labels each
span's start and end through the recorder's FIFO, and runs the actions of
microinfer.scenario on a fixed schedule: by default the whole scenario in
Chrome, or with --browser firefox the Firefox pass over actions 2 to 4 that
#45 compares browsers with. With --with-engine it first starts the engine
holding Qwen2.5-1.5B at its 32K window (tools/hold_engine.py), labels its
prefill, and begins the actions once it decodes. Action 5, the video call,
runs only when --actions names it: it opens the call's page and waits at
the terminal for the owner to start the call and press Enter, for
--call-timeout seconds at most; the hold's status file
lands beside the trace. If the engine exits during the scenario, out of
memory as #52 records it, the moment is labelled "engine-exited" and the
scenario goes on without it.

Everything it starts, it stops. On SIGINT, SIGTERM or SIGHUP the span under
way is cut short with its end label, every application it opened is
closed, the engine is stopped, and the recorder finishes its files. A
second signal ends everything at once, without grace. Its children run in
sessions of their own, so the terminal's Ctrl+C reaches only the driver,
which then stops them in order.

Beside the trace it writes trace.scenario.json: the configuration, each
span's times on the recorder's clock, what it opened (the browser's version,
the YouTube resolution asked for and the one played), any error, and the
engine's outcome. VLC plays a 4K clip from Blender's Big Buck Bunny, fetched
once into data/ and checked against a pinned hash.

Exits 0 once the scenario ran or was interrupted cleanly, and 1 if the
recorder failed, since the trace is then not whole.
"""

from __future__ import annotations

import argparse
import json
import os
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

from microinfer import benchlog, recorder, scenario  # noqa: E402
from microinfer.browser import end_process  # noqa: E402
from microinfer.engine import DECODING  # noqa: E402

VIDEO_URL = ("https://download.blender.org/demo/movies/BBB/"
             "bbb_sunflower_2160p_30fps_normal.mp4.zip")
VIDEO_ZIP = REPO / "data" / "scenario" / "bbb_sunflower_2160p_30fps_normal.mp4.zip"
VIDEO_SHA256 = "750b255c6d9fee1e2a03a6716d4f358bca56e9115bf3e06a66162fc5272ae151"
VIDEO = VIDEO_ZIP.with_suffix("")  # the .mp4 inside
ENGINE_READY_TIMEOUT_S = 1800.0  # a 32K prefill on 1.5B takes minutes


def _part(path: Path) -> Path:
    return path.with_name(path.name + ".part")


def fetch_video() -> Path:
    """The 4K clip, fetched, checked and unpacked the first time. Each file
    is written beside itself and renamed once whole, so an interrupted
    fetch or unpacking leaves nothing that passes for done."""
    if VIDEO.exists():
        return VIDEO
    if not VIDEO_ZIP.exists():
        VIDEO_ZIP.parent.mkdir(parents=True, exist_ok=True)
        print(f"fetching {VIDEO_URL} (632 MB)", flush=True)
        urllib.request.urlretrieve(VIDEO_URL, _part(VIDEO_ZIP))
        os.replace(_part(VIDEO_ZIP), VIDEO_ZIP)
    digest = benchlog.file_sha256(VIDEO_ZIP)
    if digest != VIDEO_SHA256:
        raise SystemExit(f"{VIDEO_ZIP} has sha256 {digest}, not {VIDEO_SHA256}; "
                         "delete it to fetch again")
    with zipfile.ZipFile(VIDEO_ZIP) as z, z.open(VIDEO.name) as src, \
            open(_part(VIDEO), "wb") as dst:
        shutil.copyfileobj(src, dst, 1 << 20)
    os.replace(_part(VIDEO), VIDEO)
    return VIDEO


def child(argv: list[str]) -> subprocess.Popen:
    return subprocess.Popen(argv, start_new_session=True, stdout=subprocess.DEVNULL)


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
                                          "by default all but 5 in Chrome, 2 to 4 and 9 "
                                          "in Firefox")
    parser.add_argument("--with-engine", action="store_true",
                        help="hold the engine at its 32K window through the scenario")
    parser.add_argument("--model", default="qwen2.5-1.5b-instruct")
    parser.add_argument("--hold", type=float, default=scenario.Timing.hold_s)
    parser.add_argument("--gap", type=float, default=scenario.Timing.gap_s)
    parser.add_argument("--idle", type=float, default=scenario.Timing.idle_s)
    parser.add_argument("--video", type=Path, help="the 4K file VLC plays; Big Buck Bunny, "
                                                   "fetched once, by default")
    parser.add_argument("--youtube-video", default=scenario.YOUTUBE_VIDEO)
    parser.add_argument("--call-url", default=scenario.CALL_URL,
                        help="the page the video call is started in")
    parser.add_argument("--call-timeout", type=float, default=scenario.Timing.call_timeout_s,
                        help="seconds to wait for the owner at the video call")
    args = parser.parse_args(argv)

    actions = (tuple(int(a) for a in args.actions.split(",")) if args.actions
               else scenario.DEFAULT_ACTIONS[args.browser])
    timing = scenario.Timing(hold_s=args.hold, gap_s=args.gap, idle_s=args.idle,
                             call_timeout_s=args.call_timeout)
    desktop = scenario.Desktop(args.browser, args.video, args.youtube_video, args.call_url)
    spans = scenario.spans_for(actions, desktop, timing)  # refuses an unknown action first
    if 7 in actions and args.video is None:
        desktop.video = fetch_video()

    stop = threading.Event()
    children: list[subprocess.Popen] = []

    def on_signal(*_):
        if stop.is_set():  # the second: end everything now
            desktop.kill_all()
            for proc in children:
                end_process(proc, signal.SIGKILL, grace=0.0)
            os._exit(130)
        stop.set()

    for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        signal.signal(sig, on_signal)

    fifo = Path(tempfile.mkdtemp(prefix="microinfer-scenario-")) / "labels.fifo"
    rec = child([sys.executable, str(REPO / "tools" / "record_contention.py"),
                 "--out", str(args.out), "--labels", str(fifo)])
    children.append(rec)
    status_path = recorder.companion(args.out, ".hold.json")
    engine = None
    log: dict = {"config": {"out": str(args.out), "browser": args.browser,
                            "actions": list(actions), "timing": vars(timing),
                            "with_engine": args.with_engine, "model": args.model,
                            "video": None if desktop.video is None else str(desktop.video),
                            "youtube_video": args.youtube_video},
                 "t_mono_ns": time.monotonic_ns(), "t_wall": time.time(), "spans": []}

    def send(event: str, action: str) -> None:
        recorder.send_label(fifo, event, action, wait=10.0)

    def watch_engine(done: threading.Event) -> None:
        """Label the moment the engine exits, if it does before the end."""
        while not done.wait(1.0):
            if engine.poll() is not None:
                log["engine_exited"] = {"t_mono_ns": time.monotonic_ns(),
                                        "exit_code": engine.returncode,
                                        "status": read_status(status_path)}
                for event in (recorder.START, recorder.END):
                    send(event, "engine-exited")
                return

    watching = threading.Event()
    try:
        if args.with_engine:
            engine = child([sys.executable, str(REPO / "tools" / "hold_engine.py"),
                            "--model", args.model, "--status", str(status_path)])
            children.append(engine)
            send(recorder.START, "engine-prefill")
            end = time.monotonic() + ENGINE_READY_TIMEOUT_S
            status = None
            while not stop.is_set() and engine.poll() is None and time.monotonic() < end:
                status = read_status(status_path)
                if status is not None and status["state"] == DECODING:
                    break
                stop.wait(0.5)
            send(recorder.END, "engine-prefill")
            log["engine_ready"] = status
            if not stop.is_set() and (status is None or status["state"] != DECODING):
                raise SystemExit(f"the engine did not start decoding: {status}")
            threading.Thread(target=watch_engine, args=(watching,), daemon=True).start()
        if not stop.is_set():
            log["spans"] = scenario.run(spans, desktop, send, stop, timing.gap_s)
    finally:
        watching.set()
        desktop.close_all()  # run closes it; this covers an engine that never started
        log["interrupted"] = stop.is_set()
        if engine is not None:
            log["engine"] = {"exit_code": end_process(engine, grace=60.0),
                             "status": read_status(status_path)}
        log["recorder_exit_code"] = end_process(rec, signal.SIGINT, grace=60.0)
        scenario.log_path(args.out).write_text(json.dumps(log, indent=1) + "\n")
        shutil.rmtree(fifo.parent, ignore_errors=True)  # the recorder removed its FIFO
    print(f"{len(log['spans'])} spans recorded in {args.out}"
          + (", interrupted" if log["interrupted"] else ""), flush=True)
    return 0 if log["recorder_exit_code"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
