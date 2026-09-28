"""The scenario driver: RQ1's desktop actions on a fixed schedule, labelled
in the recorder, and every application closed however the run ends (#53).

The schedule and its cleanup are tested on a desktop that only records what
it is asked, each span of it (CONTEXT.md) where the schedule puts it; the browsers are tested for real, headless, on a page served
here; and the driver end to end, with the recorder, on the actions that
open nothing.
"""

import contextlib
import json
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

from microinfer import recorder, scenario
from microinfer.browser import BROWSERS, sweep

REPO = Path(__file__).resolve().parent.parent
TOOL = REPO / "tools" / "run_scenario.py"


class FakeDesktop:
    """Records what it is asked to do; `fail` names a call that raises, and
    `slow` how long a call takes the first time."""

    browser_name = "chrome"

    def __init__(self, fail=None, slow=None):
        self.calls, self.fail, self.slow = [], fail, slow or {}

    def __getattr__(self, name):
        def call(*args):
            self.calls.append((name, *args))
            time.sleep(self.slow.pop(name, 0.0))
            if name == self.fail:
                raise RuntimeError(f"{name} failed")
            return {"did": name}
        return call


def test_the_schedule_labels_every_span_on_time_and_closes_everything_however_it_ends():
    """Each span between its start and end labels, in order, each starting
    where the schedule puts it; what the actions open is left open until
    action 9 closes it, and the run closes the desktop again as it ends. A
    span slow to open overruns without moving the spans after it. Stopped
    partway, the span under way still gets its end label and nothing after
    it starts. An action that fails is logged and the run goes on; a label
    that cannot be sent stops it, the log kept."""
    desktop = FakeDesktop()
    timing = scenario.Timing(hold_s=0.05, gap_s=0.02, idle_s=0.1)
    # The call runs only when asked for, between the videos and the WebGL
    # page; the rest of this runs the default, without it.
    assert [s.label for s in scenario.spans_for((4, 5, 6), desktop, timing)] == [
        "chrome-youtube-2160p", "chrome-video-call-setup", "chrome-video-call", "chrome-webgl"]
    assert 5 not in scenario.DEFAULT_ACTIONS["chrome"]
    spans = scenario.spans_for(scenario.DEFAULT_ACTIONS["chrome"], desktop, timing)
    labels = []
    log = scenario.run(spans, desktop, lambda e, a: labels.append((e, a)), threading.Event(),
                       gap_s=timing.gap_s)

    names = ["idle", "chrome-tabs-1", "chrome-tabs-5", "chrome-tabs-10",
             "chrome-youtube-1080p", "chrome-youtube-2160p", "chrome-webgl", "vlc-2160p",
             "vscode", "close-all"]
    assert labels == [(e, n) for n in names for e in (recorder.START, recorder.END)]
    assert [entry["label"] for entry in log] == names
    starts = [(entry["start_ns"] - log[0]["start_ns"]) / 1e9 for entry in log]
    planned = [0.0] + [0.1 + 0.02 + k * 0.07 for k in range(9)]
    assert all(abs(a - b) < 0.03 for a, b in zip(starts, planned)), starts
    assert not any(entry["overran"] or entry["cut_short"] for entry in log)
    assert desktop.calls == [
        ("tabs", 1), ("tabs", 5), ("tabs", 10), ("youtube", "hd1080"),
        ("youtube_playing", "hd1080"), ("youtube", "hd2160"), ("youtube_playing", "hd2160"),
        ("webgl",), ("vlc",), ("vscode",), ("close_vscode",), ("close_all",), ("close_all",)]
    assert log[4]["begin"] == {"did": "youtube"} and log[4]["end"] == {"did": "youtube_playing"}

    firefox = FakeDesktop()
    firefox.browser_name = "firefox"
    assert [s.label for s in scenario.spans_for(scenario.DEFAULT_ACTIONS["firefox"], firefox,
                                                scenario.Timing())] == [
        "firefox-tabs-1", "firefox-tabs-5", "firefox-tabs-10", "firefox-youtube-1080p",
        "firefox-youtube-2160p", "close-all"]
    with pytest.raises(ValueError, match="the actions are"):
        scenario.spans_for((10,), desktop, scenario.Timing())

    # Opening 1 tab takes three times its span: it overruns, and so does the
    # next span, which can only start after its own end; the one after is
    # back on the schedule, which nothing moved.
    slow = FakeDesktop(slow={"tabs": 0.3})
    log = scenario.run(scenario.spans_for((2,), slow, scenario.Timing(hold_s=0.1, gap_s=0.05)),
                       slow, lambda e, a: None, threading.Event(), gap_s=0.05)
    assert [entry["overran"] for entry in log] == [True, True, False]
    assert log[1]["late_s"] > 0.1 and log[2]["late_s"] < 0.03

    # Stopped during the third span, the second having failed.
    desktop, labels, stop = FakeDesktop(fail="vlc"), [], threading.Event()
    spans = scenario.spans_for((1, 7, 6, 8, 9), desktop,
                               scenario.Timing(hold_s=0.5, gap_s=0.01, idle_s=0.1))
    threading.Timer(0.85, stop.set).start()  # webgl runs from 0.62 s to 1.12 s
    log = scenario.run(spans, desktop, lambda e, a: labels.append((e, a)), stop, gap_s=0.01)
    assert labels == [(recorder.START, "idle"), (recorder.END, "idle"),
                      (recorder.START, "vlc-2160p"), (recorder.END, "vlc-2160p"),
                      (recorder.START, "chrome-webgl"), (recorder.END, "chrome-webgl")]
    assert "vlc failed" in log[1]["error"]  # recorded, and the run went on
    assert log[2]["cut_short"] and desktop.calls[-1] == ("close_all",)

    # The recorder gone: the label fails, the run stops, the log stays.
    def gone(event, action):
        if action == "chrome-webgl":
            raise ConnectionError("no recorder")

    desktop = FakeDesktop()
    log = scenario.run(scenario.spans_for((1, 6, 9), desktop, scenario.Timing(0.01, 0.01, 0.01)),
                       desktop, gone, threading.Event(), gap_s=0.01)
    assert [entry["label"] for entry in log] == ["idle", "chrome-webgl"]
    assert "no recorder" in log[1]["label_error"] and desktop.calls[-1] == ("close_all",)


def test_the_video_call_waits_for_a_person_and_labels_the_wait():
    """Action 5 opens the call's page and asks; the wait is a span of its
    own, and the call's span begins at the answer, the schedule after it
    counted from there. No one answering, or a page that did not open,
    skips the call; a stop while waiting ends the run. At the terminal,
    only Enter pressed after the prompt answers, and the end of the input,
    or the time running out, means no one is there."""
    def answering(after, confirmed=True):
        def ask(stop):
            time.sleep(after)
            return {"confirmed": confirmed}
        return ask

    timing = scenario.Timing(hold_s=0.1, gap_s=0.02, idle_s=0.05)
    desktop, labels = FakeDesktop(), []
    spans = scenario.spans_for((1, 5, 6), desktop, timing, ask=answering(0.3))
    log = scenario.run(spans, desktop, lambda e, a: labels.append((e, a)), threading.Event(),
                       gap_s=timing.gap_s)
    names = ["idle", "chrome-video-call-setup", "chrome-video-call", "chrome-webgl"]
    assert labels == [(e, n) for n in names for e in (recorder.START, recorder.END)]
    assert log[1]["answer"] == {"confirmed": True} and desktop.calls[0] == ("video_call",)
    waited = (log[1]["end_ns"] - log[1]["start_ns"]) / 1e9
    assert waited >= 0.3 and (log[2]["start_ns"] - log[1]["end_ns"]) / 1e9 < 0.02
    assert abs((log[3]["start_ns"] - log[2]["start_ns"]) / 1e9 - 0.12) < 0.06

    desktop, labels = FakeDesktop(), []
    log = scenario.run(scenario.spans_for((5, 6), desktop, timing, ask=answering(0, False)),
                       desktop, lambda e, a: labels.append((e, a)), threading.Event(), gap_s=0.02)
    assert [a for _, a in labels] == ["chrome-video-call-setup"] * 2 + ["chrome-webgl"] * 2
    assert "skipped" in log[1] and log[1]["label"] == "chrome-video-call"

    asked = []
    desktop = FakeDesktop(fail="video_call")
    log = scenario.run(scenario.spans_for((5, 6), desktop, timing, ask=asked.append),
                       desktop, lambda e, a: None, threading.Event(), gap_s=0.02)
    assert asked == [] and log[0]["answer"]["reason"] == "preparing it failed"
    assert "skipped" in log[1]

    stop = threading.Event()
    threading.Timer(0.1, stop.set).start()
    desktop, labels = FakeDesktop(), []
    log = scenario.run(scenario.spans_for((5, 6), desktop, timing,
                                          ask=lambda s: None if s.wait(5) else {}),
                       desktop, lambda e, a: labels.append((e, a)), stop, gap_s=0.02)
    assert [a for _, a in labels] == ["chrome-video-call-setup"] * 2
    assert log[0]["cut_short"] and desktop.calls[-1] == ("close_all",)

    def ask(stream, stop=None, timeout_s=10.0):
        with open(os.devnull, "w") as out:
            return scenario.ask_at_terminal("start the call", stop or threading.Event(),
                                            timeout_s, stream, out)

    # A terminal: an Enter typed before the prompt does not answer it.
    parent, child = os.openpty()
    with os.fdopen(child) as tty:
        os.write(parent, b"\n")
        time.sleep(0.1)
        assert ask(tty, timeout_s=0.5)["reason"] == "no answer in time"
        threading.Timer(0.2, lambda: os.write(parent, b"\n")).start()
        assert ask(tty)["confirmed"]
    os.close(parent)

    read, write = os.pipe()
    with os.fdopen(read) as stream:
        os.close(write)
        assert ask(stream)["reason"] == "the end of the input: no one there"
        stop = threading.Event()
        stop.set()
        assert ask(stream, stop) is None


@pytest.mark.parametrize("name", sorted(BROWSERS))
def test_a_browser_opens_pages_evaluates_in_their_frames_and_leaves_nothing(name):
    """Each browser, headless on a page this test serves: it opens a window,
    finds the frame inside it, evaluates there, and closing it ends every
    process that named its profile."""
    import http.server
    import socketserver

    class Page(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            port = self.server.server_address[1]
            body = (b"<title>inner</title><p>frame</p>" if self.path == "/inner" else
                    f'<title>outer</title><iframe src="http://localhost:{port}/inner">'
                    f"</iframe>".encode())
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *_):
            pass

    server = socketserver.TCPServer(("127.0.0.1", 0), Page)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    b = BROWSERS[name](headless=True)
    try:
        assert b.version.lower().startswith(name) or name in b.version.lower()
        page = b.open(f"http://127.0.0.1:{server.server_address[1]}/")
        assert b.evaluate(page, "Promise.resolve({title: document.title, n: 6 * 7})") == {
            "title": "outer", "n": 42}
        frame = b.frame(page, "/inner")
        assert b.evaluate(frame, "document.title") == "inner"
        # A second page with the same frame: each page finds its own.
        other = b.open(f"http://127.0.0.1:{server.server_address[1]}/")
        other_frame = b.frame(other, "/inner")
        assert other_frame != frame
        b.evaluate(other_frame, "(document.title = 'mine')")
        assert b.evaluate(frame, "document.title") == "inner"
    finally:
        profile = b.app.directory
        b.close()
        server.shutdown()
    assert sweep(profile) == 0
    assert not Path(profile).exists()


def test_the_driver_records_a_labelled_scenario_and_stops_cleanly(tmp_path):
    """The driver as the owner runs it, on the actions that open nothing:
    the recorder takes its labels, the scenario's log lands beside the
    trace, and an interrupt ends the action under way, with its label."""
    out = tmp_path / "trace.csv.gz"
    proc = subprocess.Popen([sys.executable, str(TOOL), "--out", str(out), "--actions", "1,9",
                             "--idle", "30", "--hold", "1", "--gap", "0.2"],
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        end = time.monotonic() + 60
        while time.monotonic() < end:
            if recorder.labels_path(out).exists():
                with contextlib.suppress(ValueError):  # its header not flushed yet
                    _, labels = recorder.read_labels(recorder.labels_path(out))
                    if len(labels):
                        break
            time.sleep(0.2)
        time.sleep(1.0)
        proc.send_signal(signal.SIGINT)  # during the idle desktop
        assert proc.wait(timeout=60) == 0
    finally:
        if proc.poll() is None:
            proc.terminate()  # the driver stops its recorder, as the owner's Ctrl+C would
            try:
                proc.wait(timeout=60)
            except subprocess.TimeoutExpired:
                proc.kill()
        # A driver killed outright leaves its recorder holding the stderr pipe.
        with contextlib.suppress(subprocess.TimeoutExpired):
            _, err = proc.communicate(timeout=10)
            if proc.returncode != 0:
                print(err, file=sys.stderr)

    meta, labels = recorder.read_labels(recorder.labels_path(out))
    assert meta["complete"]
    assert [(e, a) for e, a in zip(labels["event"], labels["action"])] == [
        (recorder.START, "idle"), (recorder.END, "idle")]
    assert recorder.read(out)[0]["complete"]
    log = json.loads(scenario.log_path(out).read_text())
    assert log["interrupted"] and [s["label"] for s in log["spans"]] == ["idle"]
    assert log["spans"][0]["cut_short"] and log["config"]["actions"] == [1, 9]
