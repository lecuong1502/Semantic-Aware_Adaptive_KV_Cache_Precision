"""The scenario driver: RQ1's desktop actions on a fixed schedule, labelled
in the recorder, and every application closed however the run ends (#53).

The schedule and its cleanup are tested on a desktop that only records what
it is asked; the browsers are tested for real, headless, on a page served
here; and the driver end to end, with the recorder, on the actions that
open nothing.
"""

import json
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
    browser_name = "chrome"

    def __init__(self, fail=None):
        self.calls, self.fail = [], fail

    def __getattr__(self, name):
        def call(*args):
            self.calls.append((name, *args))
            if name == self.fail:
                raise RuntimeError(f"{name} failed")
            return {"did": name}
        return call


def test_the_schedule_labels_every_action_and_closes_everything_however_it_ends():
    """Each action between its start and end labels, in order, holding and
    pausing as timed; what the actions open is left open until action 9
    closes it, and the run closes the desktop again as it ends. Stopped
    partway, the action under way still gets its end label, none after it
    starts, and everything is closed. An action that fails is logged, and
    the run goes on."""
    desktop = FakeDesktop()
    steps = scenario.steps_for(scenario.DEFAULT_ACTIONS["chrome"], desktop,
                               scenario.Timing(hold_s=0.05, gap_s=0.02, idle_s=0.1))
    labels = []
    started = time.monotonic()
    log = scenario.run(steps, desktop, lambda e, a: labels.append((e, a)), threading.Event(),
                       gap_s=0.02)
    took = time.monotonic() - started

    names = ["idle", "chrome-tabs-1", "chrome-tabs-5", "chrome-tabs-10",
             "chrome-youtube-1080p", "chrome-youtube-2160p", "chrome-webgl", "vlc-2160p",
             "vscode", "close-all"]
    assert labels == [(e, n) for n in names for e in (recorder.START, recorder.END)]
    assert [entry["label"] for entry in log] == names
    assert all(entry["start_ns"] < entry["end_ns"] for entry in log)
    assert took >= 0.1 + 9 * 0.05 + 9 * 0.02
    assert desktop.calls == [
        ("tabs", 1), ("tabs", 5), ("tabs", 10), ("youtube", "hd1080"),
        ("youtube_playing", "hd1080"), ("youtube", "hd2160"), ("youtube_playing", "hd2160"),
        ("webgl",), ("vlc",), ("vscode",), ("close_vscode",), ("close_all",), ("close_all",)]
    assert log[4]["begin"] == {"did": "youtube"} and log[4]["end"] == {"did": "youtube_playing"}

    firefox = FakeDesktop()
    firefox.browser_name = "firefox"
    assert [s.label for s in scenario.steps_for(scenario.DEFAULT_ACTIONS["firefox"], firefox,
                                                scenario.Timing())] == [
        "firefox-tabs-1", "firefox-tabs-5", "firefox-tabs-10", "firefox-youtube-1080p",
        "firefox-youtube-2160p", "close-all"]
    with pytest.raises(ValueError, match="#54"):
        scenario.steps_for((5,), desktop, scenario.Timing())

    # Stopped during the third step.
    desktop, labels, stop = FakeDesktop(fail="vlc"), [], threading.Event()
    steps = scenario.steps_for((1, 7, 6, 8, 9), desktop,
                               scenario.Timing(hold_s=0.3, gap_s=0.01, idle_s=0.01))
    threading.Timer(0.45, stop.set).start()
    log = scenario.run(steps, desktop, lambda e, a: labels.append((e, a)), stop, gap_s=0.01)
    assert labels == [(recorder.START, "idle"), (recorder.END, "idle"),
                      (recorder.START, "vlc-2160p"), (recorder.END, "vlc-2160p"),
                      (recorder.START, "chrome-webgl"), (recorder.END, "chrome-webgl")]
    assert "vlc failed" in log[1]["error"]  # recorded, and the run went on
    assert log[2]["cut_short"] and desktop.calls[-1] == ("close_all",)


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
                _, labels = recorder.read_labels(recorder.labels_path(out))
                if len(labels):
                    break
            time.sleep(0.2)
        time.sleep(1.0)
        proc.send_signal(signal.SIGINT)  # during the idle desktop
        assert proc.wait(timeout=60) == 0
    finally:
        if proc.poll() is None:
            proc.kill()
        _, err = proc.communicate(timeout=10)
        if proc.returncode != 0:
            print(err, file=sys.stderr)

    meta, labels = recorder.read_labels(recorder.labels_path(out))
    assert meta["complete"]
    assert [(e, a) for e, a in zip(labels["event"], labels["action"])] == [
        (recorder.START, "idle"), (recorder.END, "idle")]
    assert recorder.read(out)[0]["complete"]
    log = json.loads(scenario.log_path(out).read_text())
    assert log["interrupted"] and [s["label"] for s in log["steps"]] == ["idle"]
    assert log["steps"][0]["cut_short"] and log["config"]["actions"] == [1, 9]
