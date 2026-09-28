"""RQ1's scripted scenario: the desktop actions, on a fixed schedule, each
marked in the recording with a start and an end label (#53).

The Milestone 1 spec (#45) names nine actions:

1. the idle desktop;
2. a browser with 1, then 5, then 10 ordinary tabs, recorded as three spans;
3. YouTube at 1080p;
4. YouTube at 4K;
5. a browser video call with camera and screen share, which needs a person;
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

**The video call waits for a person** (#54). Signing in and granting the
camera and the screen are the owner's, so the driver opens the call's page
and asks, and the wait is a span of its own, "<browser>-video-call-setup",
from the prompt to the owner's answer: what signing in and starting a
camera cost is still labelled. The call's own span begins at the answer,
and the schedule after it is counted from there. With no one to answer,
in ten minutes or at the end of the input, the call is skipped and the run
goes on.

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
import select
import shutil
import socketserver
import sys
import termios
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from . import browser as browsers
from .recorder import END, START, companion

#: The actions of #45 this driver runs, by number, with the label of each.
#: A browser action's label names the browser too.
ACTIONS = {1: "idle", 2: "tabs", 3: "youtube-1080p", 4: "youtube-2160p", 5: "video-call",
           6: "webgl", 7: "vlc-2160p", 8: "vscode", 9: "close-all"}
BROWSER_ACTIONS = (2, 3, 4, 5, 6)
#: What a run covers by default: the scenario in Chrome, and #45's Firefox
#: pass over actions 2 to 4, closing everything after them. Action 5 runs
#: only when asked for: Google refused the owner's sign-in in the driver's
#: fresh, remotely controlled profile (#55), so the call could not start.
DEFAULT_ACTIONS = {"chrome": (1, 2, 3, 4, 6, 7, 8, 9), "firefox": (2, 3, 4, 9)}
#: A call anyone signed in can start, with no link to share first.
CALL_URL = "https://meet.google.com/new"
CALL_PROMPT = ("Action 5, the video call. In the browser window just opened: sign in if "
               "asked, start the call with the camera on, and share the entire screen.")

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
#: How long before its end label a span's closing work begins: time for VS
#: Code to close, or a player to be read, inside the span's own time.
END_LEAD_S = 5.0


@dataclass(frozen=True)
class Timing:
    """How long each span is held, the gap between spans, how long the idle
    desktop is recorded (#45: 60 s, with gaps, and 2 minutes), and how long
    a person is waited for."""

    hold_s: float = 60.0
    gap_s: float = 10.0
    idle_s: float = 120.0
    #: How long the video call waits for the owner before it is skipped, so
    #: that a run no one attends still ends.
    call_timeout_s: float = 600.0


@dataclass
class Span:
    """One labelled span of a scenario (CONTEXT.md): an action, or one of
    the stages an action is recorded in, such as 5 tabs of 1, 5 and 10.
    `begin` runs at its start label, `end` shortly before its end label, and
    each may return details for the log."""

    label: str
    hold_s: float
    begin: Callable[[], dict | None] = lambda: None
    end: Callable[[], dict | None] = lambda: None


@dataclass
class WaitSpan:
    """A span that waits for a person, before the span it prepares: it has
    no fixed length, and the schedule after it is counted from its end.
    `prepare` runs at its start label; if that works, `ask(stop)` waits for
    the person, returning {"confirmed": ...} with details, or None if the
    run was stopped meanwhile. Unless someone confirms, the span after it is
    skipped: there is nothing to hold."""

    label: str
    prepare: Callable[[], dict | None]
    ask: Callable[[threading.Event], dict | None]


#: How often a wait for a person checks whether the run was stopped.
_POLL_S = 0.2


def ask_at_terminal(message: str, stop: threading.Event, timeout_s: float,
                    stream=None, out=None) -> dict | None:
    """Print `message` and wait for Enter on `stream` (stdin), checking
    `stop` as it waits. A terminal's input typed before the prompt, an Enter
    pressed during the prefill, is discarded first, so only an answer to
    this prompt counts. Returns {"confirmed": True} with the wait in seconds;
    {"confirmed": False, "reason": ...} at the end of the input, no one being
    there, or after `timeout_s`, no one answering; None if stopped first."""
    stream, out = stream or sys.stdin, out or sys.stdout
    if stream.isatty():
        termios.tcflush(stream, termios.TCIFLUSH)
    started = time.monotonic()
    print(f"\n>>> {message}\n>>> Press Enter when it is done "
          f"(within {timeout_s / 60:g} minutes).", file=out, flush=True)

    def answer(confirmed: bool, reason: str | None = None) -> dict:
        return {"confirmed": confirmed, "waited_s": round(time.monotonic() - started, 3),
                **({"reason": reason} if reason else {})}

    while not stop.is_set():
        if time.monotonic() - started > timeout_s:
            return answer(False, "no answer in time")
        ready, _, _ = select.select([stream], [], [], _POLL_S)
        if ready:
            if stream.readline() == "":
                return answer(False, "the end of the input: no one there")
            return answer(True)
    return None


def run(spans: list[Span | WaitSpan], desktop, send_label: Callable[[str, str], None],
        stop: threading.Event, gap_s: float) -> list[dict]:
    """Run `spans` on a fixed schedule until they are done or `stop` is set,
    then close the desktop, however the run ended.

    Span k starts at the sum of the spans and gaps before it, counted from
    the first start: an action slow to open does not move the ones after
    it. `end` runs END_LEAD_S before the span's end, so its work falls
    inside the span. A span whose opening or closing overran its time is
    marked so, and one that starts late says by how much. A WaitSpan
    lasts as long as the person takes, and the span it prepares follows at
    once, without a gap: the schedule after it is counted from the answer.

    Returns the log: one entry per span begun, with its times on the
    monotonic clock, its details and any error, which is recorded and the
    run goes on. A span under way when `stop` is set is cut short and still
    ends with its label. A label that cannot be sent, the recorder gone,
    stops the run: nothing after it would be labelled."""
    log: list[dict] = []

    def label(entry: dict, event: str) -> None:
        try:
            send_label(event, entry["label"])
        except Exception as exc:  # noqa: BLE001 - recorded, and the run stops
            entry.setdefault("label_error", f"{event}: {type(exc).__name__}: {exc}")
            stop.set()

    def attempt(entry: dict, part: str, work: Callable[[], dict | None]) -> None:
        try:
            entry[part] = work()
        except Exception as exc:  # noqa: BLE001 - recorded; the run goes on
            entry.setdefault("error", f"{part}: {type(exc).__name__}: {exc}")

    def until(t: float) -> bool:
        """Wait until monotonic time `t`; True if stopped first."""
        return stop.wait(max(t - time.monotonic(), 0.0))

    def finish(entry: dict) -> None:
        entry["cut_short"] = stop.is_set()
        label(entry, END)
        entry["end_ns"] = time.monotonic_ns()

    try:
        planned = time.monotonic()
        skip_next = False
        for span in spans:
            if until(planned):
                break
            entry: dict = {"label": span.label, "start_ns": time.monotonic_ns(),
                           "late_s": round(time.monotonic() - planned, 3)}
            log.append(entry)
            if skip_next:
                entry["skipped"], skip_next = "no one confirmed the span before it", False
                entry["end_ns"] = entry["start_ns"]
                continue  # the next span takes its place in the schedule
            label(entry, START)
            if isinstance(span, WaitSpan):
                attempt(entry, "prepare", span.prepare)
                # Nothing to ask the person to do if preparing it failed.
                answer = (span.ask(stop) if "error" not in entry
                          else {"confirmed": False, "reason": "preparing it failed"})
                entry["answer"] = answer
                finish(entry)
                skip_next = answer is not None and not answer.get("confirmed")
                planned = time.monotonic()  # the rest is counted from the answer
                continue
            end_at = planned + span.hold_s
            if not stop.is_set():
                attempt(entry, "begin", span.begin)
                entry["overran"] = time.monotonic() > end_at
                until(end_at - min(END_LEAD_S, span.hold_s / 2))
                attempt(entry, "end", span.end)
                entry["overran"] = entry["overran"] or time.monotonic() > end_at
                until(end_at)
            finish(entry)
            planned = end_at + gap_s
    finally:
        desktop.close_all()
    return log


def spans_for(actions: tuple[int, ...], desktop, timing: Timing,
              ask: Callable[[threading.Event], dict | None] | None = None
              ) -> list[Span | WaitSpan]:
    """The spans of the numbered actions, in the order given. `ask` waits
    for the person at the video call; by default, at the terminal."""
    ask = ask or (lambda stop: ask_at_terminal(CALL_PROMPT, stop, timing.call_timeout_s))
    unknown = set(actions) - set(ACTIONS)
    if unknown:
        raise ValueError(f"no action {sorted(unknown)}: the actions are {sorted(ACTIONS)}")
    browser = desktop.browser_name

    def labelled(n: int) -> str:
        return f"{browser}-{ACTIONS[n]}" if n in BROWSER_ACTIONS else ACTIONS[n]

    quality = {3: "hd1080", 4: "hd2160"}
    spans_of = {
        1: lambda: [Span(labelled(1), timing.idle_s)],
        2: lambda: [Span(f"{labelled(2)}-{count}", timing.hold_s,
                         begin=lambda count=count: desktop.tabs(count))
                    for count in TAB_COUNTS],
        **{n: (lambda q=q, n=n: [Span(labelled(n), timing.hold_s,
                                      begin=lambda: desktop.youtube(q),
                                      end=lambda: desktop.youtube_playing(q))])
           for n, q in quality.items()},
        5: lambda: [WaitSpan(f"{labelled(5)}-setup", desktop.video_call, ask),
                    Span(labelled(5), timing.hold_s)],
        6: lambda: [Span(labelled(6), timing.hold_s, begin=desktop.webgl)],
        7: lambda: [Span(labelled(7), timing.hold_s, begin=desktop.vlc)],
        8: lambda: [Span(labelled(8), timing.hold_s, begin=desktop.vscode,
                         end=desktop.close_vscode)],
        9: lambda: [Span(labelled(9), timing.hold_s, begin=desktop.close_all)],
    }
    return [span for n in actions for span in spans_of[n]()]


def log_path(trace: str | Path) -> Path:
    """Where a scenario's log is written: beside its recording, trace.csv.gz
    becoming trace.scenario.json."""
    return companion(trace, ".scenario.json")


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


#: What the player in the frame plays: its quality and state, the height of
#: the video it decodes, and any error it shows.
_PLAYING = """(() => {
  const p = document.getElementById("movie_player");
  return {
    quality: p && p.getPlaybackQuality ? p.getPlaybackQuality() : null,
    state: p && p.getPlayerState ? p.getPlayerState() : null,
    video_height: (document.querySelector("video") || {}).videoHeight || null,
    error: (document.querySelector(".ytp-error") || {}).innerText || null};
})()"""

#: Asks the player for a quality until it plays at it or 30 s pass, then
#: says what it plays.
_FORCE_QUALITY = """(async () => {
  const want = %s;
  for (let i = 0; i < 60; i++) {
    const p = document.getElementById("movie_player");
    if (p && p.getAvailableQualityLevels && p.getAvailableQualityLevels().length) {
      p.mute(); p.setPlaybackQualityRange(want, want); p.playVideo();
      if (p.getPlaybackQuality() === want) break;
    }
    await new Promise(r => setTimeout(r, 500));
  }
  return %s;
})()"""


class Desktop:
    """The applications a scenario opens, and the closing of them all.

    One browser serves the browser actions, opened on the first; VLC and VS
    Code are applications of their own. `video` is the local 4K file VLC
    plays."""

    def __init__(self, browser_name: str = "chrome", video: str | Path | None = None,
                 youtube_video: str = YOUTUBE_VIDEO, call_url: str = CALL_URL):
        if browser_name not in browsers.BROWSERS:
            raise ValueError(f"the browsers are {sorted(browsers.BROWSERS)}")
        self.browser_name, self.video, self.youtube_video = browser_name, video, youtube_video
        self.call_url = call_url
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
            except (browsers.BrowserError, OSError) as exc:
                failed[url] = f"{type(exc).__name__}: {exc}"
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
        return self._open_browser().evaluate(self._players[quality], _PLAYING)

    def video_call(self) -> dict:
        """The call's page, in a window of its own, for the person to start
        the call in. It stays open, the call with it, until action 9."""
        b = self._open_browser()
        b.open(self.call_url)
        return {"browser": b.version, "url": self.call_url}

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

    def _everything(self) -> list[tuple[str, object]]:
        return [*self._apps.items(),
                *([(self.browser_name, self._browser)] if self._browser is not None else [])]

    def close_all(self) -> dict:
        """Close every application open, the browser included: each on its
        own, so that one that will not close does not keep the rest open."""
        closed, errors = [], {}
        for name, app in self._everything():
            try:
                app.close()
                closed.append(name)
            except Exception as exc:  # noqa: BLE001 - reported; the rest still close
                errors[name] = f"{type(exc).__name__}: {exc}"
        self._forget()
        return {"closed": closed, **({"errors": errors} if errors else {})}

    def kill_all(self) -> None:
        """End every application at once, without grace: the driver forced
        to stop."""
        for _, app in self._everything():
            try:
                app.kill()
            except Exception:  # noqa: BLE001, S110 - the rest must still die
                pass
        self._forget()

    def _forget(self) -> None:
        self._apps.clear()
        self._browser, self._tabs = None, 0
        self._players.clear()
        if self._pages is not None:
            self._pages.shutdown()
            self._pages.server_close()
            self._pages = None
