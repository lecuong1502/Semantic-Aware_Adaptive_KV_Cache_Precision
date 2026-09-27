"""RQ1's scripted scenario: the desktop actions, on a fixed schedule, each
marked in the recording with a start and an end label (#53).

The Milestone 1 spec (#45) names nine actions; this runs the eight that need
no person, the video call (5) being #54's:

1. the idle desktop;
2. a browser with 1, then 5, then 10 ordinary tabs, three labelled steps;
3. YouTube at 1080p;
4. YouTube at 4K;
6. a heavy WebGL page, standing in for a game;
7. VLC playing a local 4K video;
8. VS Code, opened, held, and closed again;
9. closing everything, to see whether and how fast memory returns.

**What opens stays open** until action 9 closes it, as a desktop's
applications accumulate; only VS Code closes within its own action, as the
spec has it. The spike each application brings as it opens still falls in
its action.

**YouTube plays at the resolution asked for.** YouTube takes no resolution
from its URL, and a watch page may be challenged as bot traffic, so the
video is embedded in a page served here, and the driver asks the player
inside the frame for the resolution, then records the one it got and the
height of the video it decodes (browser.py).

**Everything it opens is closed**, when action 9 comes or however the run
ends: `run` closes the desktop in its `finally`, and every application is
started so that closing it ends every process it started (browser.App).

An action that fails, a page that will not load, is recorded in the
scenario's log with its error, holds its span of the schedule all the same,
and the run goes on: one missing action leaves the rest of the recording
usable, and on the schedule of every other repeat.
"""

from __future__ import annotations

import http.server
import shutil
import socketserver
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from . import browser as browsers
from .recorder import END, START

#: The actions of #45 this driver runs, by number.
ACTIONS = {1: "idle", 2: "tabs", 3: "youtube-1080p", 4: "youtube-2160p", 6: "webgl",
           7: "vlc-2160p", 8: "vscode", 9: "close-all"}
#: What a run covers by default: the whole scenario in Chrome, and #45's
#: Firefox pass over actions 2 to 4, closing everything after them.
DEFAULT_ACTIONS = {"chrome": (1, 2, 3, 4, 6, 7, 8, 9), "firefox": (2, 3, 4, 9)}
BROWSER_ACTIONS = {2, 3, 4, 6}

#: Ordinary pages, opened as tabs: text, images and scripts, nothing that plays.
TABS = ("https://en.wikipedia.org/wiki/Graphics_processing_unit",
        "https://edition.cnn.com/",
        "https://github.com/trending",
        "https://news.ycombinator.com/",
        "https://docs.python.org/3/",
        "https://en.wikipedia.org/wiki/Video_random-access_memory",
        "https://arxiv.org/list/cs.LG/recent",
        "https://www.theguardian.com/international",
        "https://stackoverflow.com/questions",
        "https://www.nasa.gov/")
TAB_COUNTS = (1, 5, 10)
#: A video published in 4K and down to 144p.
YOUTUBE_VIDEO = "LXb3EKWsInQ"
#: The WebGL Aquarium: thousands of fish, rendered every frame.
WEBGL_URL = "https://webglsamples.org/aquarium/aquarium.html"


@dataclass(frozen=True)
class Timing:
    """How long each action is held, the gap between actions, and how long
    the idle desktop is recorded (#45: 60 s, with gaps, and 2 minutes)."""

    hold_s: float = 60.0
    gap_s: float = 10.0
    idle_s: float = 120.0


@dataclass
class Step:
    """One labelled span of a scenario: `begin` runs as it starts, `end`
    before its end label; each may return details for the log."""

    label: str
    hold_s: float
    begin: Callable[[], dict | None] = lambda: None
    end: Callable[[], dict | None] = lambda: None


def run(steps: list[Step], desktop, send_label: Callable[[str, str], None],
        stop: threading.Event, gap_s: float) -> list[dict]:
    """Run `steps` in order, each between its start and its end label, with
    `gap_s` between them, until they are done or `stop` is set; then close
    the desktop, however the run ended. Returns the log: one entry per step
    begun, with its times on the monotonic clock, its details, and its
    error if it had one. A step under way when `stop` is set is cut short
    and still ends with its label."""
    log: list[dict] = []
    try:
        for i, step in enumerate(steps):
            if stop.is_set() or (i and stop.wait(gap_s)):
                break
            entry: dict = {"label": step.label, "start_ns": time.monotonic_ns()}
            send_label(START, step.label)
            try:
                for part, action in (("begin", step.begin), ("end", step.end)):
                    try:
                        entry[part] = action()
                    except Exception as exc:  # noqa: BLE001 - recorded; the run goes on
                        entry.setdefault("error", f"{part}: {type(exc).__name__}: {exc}")
                    if part == "begin":
                        # The step spans hold_s from its start label, however
                        # long opening took, failed or not: the schedule is
                        # the same in every repeat.
                        held = (time.monotonic_ns() - entry["start_ns"]) / 1e9
                        stop.wait(max(step.hold_s - held, 0.0))
            finally:
                send_label(END, step.label)
                entry["end_ns"] = time.monotonic_ns()
                entry["cut_short"] = stop.is_set()
                log.append(entry)
    finally:
        desktop.close_all()
    return log


def steps_for(actions: tuple[int, ...], desktop, timing: Timing) -> list[Step]:
    """The steps of the numbered actions, in the order given."""
    unknown = set(actions) - set(ACTIONS)
    if unknown:
        raise ValueError(f"no action {sorted(unknown)}: the actions are {sorted(ACTIONS)}, "
                         f"5, the video call, being #54's")
    b = desktop.browser_name
    steps: list[Step] = []
    for n in actions:
        if n == 1:
            steps.append(Step("idle", timing.idle_s))
        elif n == 2:
            for count in TAB_COUNTS:
                steps.append(Step(f"{b}-tabs-{count}", timing.hold_s,
                                  begin=lambda count=count: desktop.tabs(count)))
        elif n in (3, 4):
            quality = "hd1080" if n == 3 else "hd2160"
            steps.append(Step(f"{b}-{ACTIONS[n]}", timing.hold_s,
                              begin=lambda q=quality: desktop.youtube(q),
                              end=lambda q=quality: desktop.youtube_playing(q)))
        elif n == 6:
            steps.append(Step(f"{b}-webgl", timing.hold_s, begin=desktop.webgl))
        elif n == 7:
            steps.append(Step("vlc-2160p", timing.hold_s, begin=desktop.vlc))
        elif n == 8:
            steps.append(Step("vscode", timing.hold_s, begin=desktop.vscode,
                              end=desktop.close_vscode))
        elif n == 9:
            steps.append(Step("close-all", timing.hold_s, begin=desktop.close_all))
    return steps


def log_path(trace: str | Path) -> Path:
    """Where a scenario's log is written: beside its recording, trace.csv.gz
    becoming trace.scenario.json."""
    trace = Path(trace)
    name = trace.name
    for suffix in (".csv.gz", ".gz", ".csv"):
        if name.endswith(suffix):
            name = name[: -len(suffix)]
            break
    return trace.with_name(name + ".scenario.json")


# -- the desktop ------------------------------------------------------------------------


class _Pages(socketserver.ThreadingMixIn, http.server.HTTPServer):
    """The pages the driver serves itself, on localhost: a YouTube video
    embedded to fill the window, which the player plays only inside a page
    of its own origin."""

    daemon_threads = True

    def __init__(self):
        super().__init__(("127.0.0.1", 0), _PageHandler)
        self.origin = f"http://127.0.0.1:{self.server_address[1]}"
        threading.Thread(target=self.serve_forever, name="scenario-pages",
                         daemon=True).start()


class _PageHandler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):  # noqa: N802 - the name http.server calls
        video = self.path.rsplit("/", 1)[-1]
        if not self.path.startswith("/youtube/") or not video.replace("-", "").replace(
                "_", "").isalnum():
            self.send_error(404)
            return
        origin = self.server.origin
        body = (
            '<!doctype html><html><head><meta name="referrer" '
            'content="strict-origin-when-cross-origin"><style>html,body,iframe{margin:0;'
            'border:0;width:100%;height:100%;overflow:hidden;background:#000}</style>'
            f'</head><body><iframe src="https://www.youtube.com/embed/{video}?autoplay=1'
            f'&mute=1&loop=1&playlist={video}&enablejsapi=1&origin={origin}" '
            'allow="autoplay; fullscreen" referrerpolicy="strict-origin-when-cross-origin">'
            '</iframe></body></html>').encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *_):
        pass


#: Asks the player for `want`, until it plays at it or 30 s pass, and says
#: what it plays: the player's quality, and the height of the video decoded.
_FORCE_QUALITY = """(async () => {
  const want = %s;
  let p = null;
  for (let i = 0; i < 60; i++) {
    p = document.getElementById("movie_player");
    if (p && p.getAvailableQualityLevels && p.getAvailableQualityLevels().length) {
      p.mute(); p.setPlaybackQualityRange(want, want); p.playVideo();
      if (p.getPlaybackQuality() === want) break;
    }
    await new Promise(r => setTimeout(r, 500));
  }
  return %s;
})()"""
_PLAYING = """({
  quality: p && p.getPlaybackQuality ? p.getPlaybackQuality() : null,
  state: p && p.getPlayerState ? p.getPlayerState() : null,
  video_height: (document.querySelector("video") || {}).videoHeight || null,
  error: (document.querySelector(".ytp-error") || {}).innerText || null})"""


class Desktop:
    """The applications a scenario opens, and the closing of them all.

    One browser serves the browser actions, opened on the first; VLC and VS
    Code are applications of their own. `video` is the local 4K file VLC
    plays."""

    def __init__(self, browser_name: str = "chrome", video: str | Path | None = None,
                 youtube_video: str = YOUTUBE_VIDEO):
        if browser_name not in browsers.BROWSERS:
            raise ValueError(f"the browsers are {sorted(browsers.BROWSERS)}")
        self.browser_name, self.video, self.youtube_video = browser_name, video, youtube_video
        self._browser = None
        self._tabs = 0
        self._players: dict[str, str] = {}  # quality -> the frame playing it
        self._apps: dict[str, browsers.App] = {}
        self._pages: _Pages | None = None

    def _open_browser(self):
        if self._browser is None:
            self._browser = browsers.BROWSERS[self.browser_name]()
        return self._browser

    def tabs(self, count: int) -> dict:
        """Ordinary tabs up to `count`, in one window: the first opens it.
        A page that will not load is noted and the tab stays, showing its
        error, as it would for anyone browsing."""
        b = self._open_browser()
        opened, failed = [], {}
        while self._tabs < count:
            url = TABS[self._tabs]
            try:
                page = b.open(url, new_window=self._tabs == 0)
                where = b.evaluate(page, "location.href")
                if not where.startswith(("http://", "https://")):
                    failed[url] = f"showed {where}"  # the browser's own error page
            except browsers.BrowserError as exc:
                failed[url] = str(exc)
            opened.append(url)
            self._tabs += 1
        return {"browser": b.version, "tabs": self._tabs, "opened": opened, "failed": failed}

    def youtube(self, quality: str) -> dict:
        """The video in a window of its own, playing at `quality`."""
        b = self._open_browser()
        if self._pages is None:
            self._pages = _Pages()
        page = b.open(f"{self._pages.origin}/youtube/{self.youtube_video}")
        frame = b.frame(page, "youtube.com/embed/")
        self._players[quality] = frame
        playing = b.evaluate(frame, _FORCE_QUALITY % (f'"{quality}"', _PLAYING), timeout=60)
        return {"browser": b.version, "asked": quality, **playing}

    def youtube_playing(self, quality: str) -> dict:
        """What the player opened for `quality` plays now."""
        frame = self._players[quality]
        return self._browser.evaluate(
            frame, f'(() => {{ const p = document.getElementById("movie_player"); '
                   f'return {_PLAYING}; }})()')

    def webgl(self) -> dict:
        b = self._open_browser()
        b.open(WEBGL_URL)
        return {"browser": b.version, "url": WEBGL_URL}

    def vlc(self) -> dict:
        if self.video is None or not Path(self.video).is_file():
            raise FileNotFoundError(f"no video to play: {self.video}")
        self._apps["vlc"] = browsers.App("vlc", lambda d: [
            "vlc", "--fullscreen", "--loop", "--no-video-title-show", "--no-qt-privacy-ask",
            "--config", f"{d}/vlcrc", str(self.video)])
        return {"video": str(self.video)}

    def vscode(self) -> dict:
        """VS Code on this repository, as its own instance: the Electron
        binary, not the `code` launcher, which hands the window to a running
        instance and exits."""
        electron = Path(shutil.which("code") or "/usr/bin/code").resolve().parent.parent / "code"
        repo = Path(__file__).resolve().parents[2]
        self._apps["vscode"] = browsers.App("vscode", lambda d: [
            str(electron), "--user-data-dir", f"{d}/user", "--extensions-dir", f"{d}/ext",
            "--disable-extensions", "--new-window", str(repo)])
        return {"executable": str(electron)}

    def close_vscode(self) -> dict | None:
        app = self._apps.pop("vscode", None)
        if app is not None:
            app.close()
        return None

    def close_all(self) -> dict:
        """Close every application open, the browser included."""
        closed = list(self._apps)
        for app in list(self._apps.values()):
            app.close()
        self._apps.clear()
        if self._browser is not None:
            closed.append(self.browser_name)
            self._browser.close()
            self._browser, self._tabs = None, 0
            self._players.clear()
        if self._pages is not None:
            self._pages.shutdown()
            self._pages.server_close()
            self._pages = None
        return {"closed": closed}
