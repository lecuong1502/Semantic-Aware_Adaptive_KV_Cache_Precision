"""Browsers driven by the scenario driver (#53): Chrome over the DevTools
Protocol, Firefox over WebDriver BiDi, both over a WebSocket client written
here on the standard library. The environment is pinned, and gains no
dependency for this (pyproject.toml).

RQ1's scenarios need what a browser's own controls do: open a window, play a
video at a chosen resolution, and say what it is playing. YouTube no longer
takes a resolution from its URL, and left to choose it picks one from the
network and the window. So the driver asks the page's player for a
resolution, and reads back the one it got.

Each browser runs on a profile of its own, in a fresh temporary directory:
it cannot join a browser already open, it starts from nothing each repeat,
and its directory marks every process it starts. Closing it ends its process
group and then any process whose command line still names that directory,
since a browser starts helpers outside its group.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import shutil
import signal
import socket
import struct
import subprocess
import tempfile
import time
import urllib.parse
from pathlib import Path
from typing import Callable


class BrowserError(RuntimeError):
    """The browser refused a command, or could not be reached."""


# -- a WebSocket client, RFC 6455, the little of it a browser's protocol needs --


_GUID = b"258EAFA5-E914-47DA-95CA-C5AB0DC85B11"


class WebSocket:
    """A client connection that sends and receives text messages.

    Every read has a deadline. One that passes mid-frame leaves the stream
    at an unknown point, so the connection is closed then, and every later
    call raises."""

    def __init__(self, url: str, timeout: float = 30.0):
        u = urllib.parse.urlparse(url)
        path = (u.path or "/") + (f"?{u.query}" if u.query else "")
        self._sock = socket.create_connection((u.hostname, u.port or 80), timeout=timeout)
        self._buffer = b""
        self._deadline = time.monotonic() + timeout
        key = base64.b64encode(os.urandom(16)).decode()
        self._sock.sendall(
            f"GET {path} HTTP/1.1\r\nHost: {u.hostname}:{u.port}\r\nUpgrade: websocket\r\n"
            f"Connection: Upgrade\r\nSec-WebSocket-Key: {key}\r\n"
            f"Sec-WebSocket-Version: 13\r\n\r\n".encode())
        head = self._take(self._headers_end).decode("latin-1")
        accept = base64.b64encode(hashlib.sha1(key.encode() + _GUID).digest()).decode()
        if not head.startswith("HTTP/1.1 101") or accept not in head:
            self.close()
            raise BrowserError(f"{url} refused the WebSocket handshake: {head.splitlines()[0]}")

    def _take(self, ready: Callable[[], int]) -> bytes:
        """The first `ready()` bytes of the stream, reading until `ready`
        says how many there are (not 0), or the deadline passes."""
        while not (n := ready()):
            left = self._deadline - time.monotonic()
            try:
                if left <= 0:
                    raise TimeoutError
                self._sock.settimeout(left)
                chunk = self._sock.recv(65536)
            except OSError as exc:  # a timeout among them
                self.close()
                raise BrowserError("the browser did not answer in time; connection "
                                   "closed") from exc
            if not chunk:
                self.close()
                raise BrowserError("the browser closed the connection")
            self._buffer += chunk
        out, self._buffer = self._buffer[:n], self._buffer[n:]
        return out

    def _headers_end(self) -> int:
        at = self._buffer.find(b"\r\n\r\n")
        return at + 4 if at >= 0 else 0

    def _read(self, n: int) -> bytes:
        return self._take(lambda: n if len(self._buffer) >= n else 0)

    def _frame(self, opcode: int, payload: bytes) -> None:
        # A client masks every frame it sends (RFC 6455, 5.3).
        n = len(payload)
        head = bytes([0x80 | opcode])
        if n < 126:
            head += bytes([0x80 | n])
        elif n < 1 << 16:
            head += bytes([0x80 | 126]) + struct.pack("!H", n)
        else:
            head += bytes([0x80 | 127]) + struct.pack("!Q", n)
        mask = os.urandom(4)
        body = bytes(b ^ mask[i % 4] for i, b in enumerate(payload))
        self._sock.sendall(head + mask + body)

    def send(self, text: str) -> None:
        self._frame(0x1, text.encode())

    def recv(self, deadline: float) -> str:
        """The next text message, by `deadline` on the monotonic clock,
        answering pings on the way."""
        self._deadline = deadline
        parts: list[bytes] = []
        while True:
            b0, b1 = self._read(2)
            opcode, n = b0 & 0x0F, b1 & 0x7F
            if n == 126:
                n = struct.unpack("!H", self._read(2))[0]
            elif n == 127:
                n = struct.unpack("!Q", self._read(8))[0]
            mask = self._read(4) if b1 & 0x80 else None
            payload = self._read(n)
            if mask is not None:
                payload = bytes(b ^ mask[i % 4] for i, b in enumerate(payload))
            if opcode == 0x8:
                self.close()
                raise BrowserError("the browser closed the connection")
            if opcode == 0x9:
                self._frame(0xA, payload)
                continue
            if opcode in (0x1, 0x0):  # text, and its continuations
                parts.append(payload)
                if b0 & 0x80:
                    return b"".join(parts).decode()

    def close(self) -> None:
        try:
            self._frame(0x8, b"")
        except OSError:
            pass
        self._sock.close()


class _Rpc:
    """Commands and their answers over a WebSocket. The DevTools Protocol
    and WebDriver BiDi both send {"id", "method", "params"} and answer with
    the same id; everything else on the line is an event, and ignored."""

    def __init__(self, ws: WebSocket):
        self._ws, self._next = ws, 0

    def call(self, method: str, params: dict | None = None, *, timeout: float = 30.0,
             **extra) -> dict:
        """The answer to `method`, within `timeout` seconds of asking,
        however many events arrive meanwhile."""
        deadline = time.monotonic() + timeout
        self._next += 1
        wanted = self._next
        self._ws.send(json.dumps({"id": wanted, "method": method, "params": params or {},
                                  **extra}))
        while True:
            message = json.loads(self._ws.recv(deadline))
            if message.get("id") != wanted:
                continue
            if "error" in message:
                error = message["error"]
                detail = error.get("message") if isinstance(error, dict) else message.get(
                    "message", error)
                raise BrowserError(f"{method}: {detail}")
            return message.get("result", {})

    def close(self) -> None:
        self._ws.close()


# -- processes -------------------------------------------------------------------------


def _signal(kill: Callable[[int, int], None], target: int, sig: int) -> None:
    """os.kill or os.killpg, for a process that may already be gone, or be a
    setuid launcher's (a snap's) that is not ours to signal."""
    try:
        kill(target, sig)
    except (ProcessLookupError, PermissionError):
        pass


def sweep(marker: str, grace: float = 3.0) -> int:
    """End every process whose command line names `marker`: SIGTERM, then
    SIGKILL for any left after `grace` seconds. Returns how many there were.
    The marker is a directory made for one application, so no other process
    names it."""
    def matching() -> list[int]:
        found = []
        for entry in Path("/proc").iterdir():
            if not entry.name.isdigit() or int(entry.name) == os.getpid():
                continue
            try:
                if marker.encode() in (entry / "cmdline").read_bytes():
                    found.append(int(entry.name))
            except OSError:
                pass  # it exited, or is not ours to read
        return found

    pids = matching()
    for pid in pids:
        _signal(os.kill, pid, signal.SIGTERM)
    end = time.monotonic() + grace
    while time.monotonic() < end and matching():
        time.sleep(0.1)
    for pid in matching():
        _signal(os.kill, pid, signal.SIGKILL)
    return len(pids)


def end_process(proc: subprocess.Popen, sig: int = signal.SIGTERM, grace: float = 5.0,
                group: bool = True) -> int | None:
    """Send `sig` to a process, or to its process group, wait `grace`
    seconds for it, then kill it. Returns its exit code, None if even SIGKILL
    left it running."""
    kill = os.killpg if group else os.kill
    if proc.poll() is None:
        _signal(kill, proc.pid, sig)
        try:
            proc.wait(timeout=grace)
        except subprocess.TimeoutExpired:
            _signal(kill, proc.pid, signal.SIGKILL)
            try:
                proc.wait(timeout=grace)
            except subprocess.TimeoutExpired:
                return None
    return proc.returncode


class App:
    """An application started for a scenario, in a session of its own, with
    a temporary directory that marks its processes. Closing it ends them
    all, however it was started."""

    def __init__(self, name: str, argv_for: Callable[[str], list[str]],
                 under: str | Path | None = None):
        """`argv_for(directory)` is the command line, given the directory,
        which is made under `under`, or the system's temporary directory."""
        self.name = name
        self.directory = tempfile.mkdtemp(prefix=f"microinfer-{name}-",
                                          dir=None if under is None else str(under))
        self.argv = argv_for(self.directory)
        self.process = subprocess.Popen(self.argv, start_new_session=True,
                                        stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                                        stderr=subprocess.DEVNULL)

    def close(self, grace: float = 5.0) -> None:
        end_process(self.process, grace=grace)
        sweep(self.directory)
        shutil.rmtree(self.directory, ignore_errors=True)

    def kill(self) -> None:
        """End it at once, SIGKILL without grace, as when the driver is forced."""
        _signal(os.killpg, self.process.pid, signal.SIGKILL)
        sweep(self.directory, grace=0.0)


def _wait_for(what: str, probe, timeout: float):
    end = time.monotonic() + timeout
    while True:
        try:
            result = probe()
            if result:
                return result
        except (OSError, BrowserError, ValueError):
            pass
        if time.monotonic() > end:
            raise BrowserError(f"{what} did not come up in {timeout:g} s")
        time.sleep(0.2)


# -- the browsers ---------------------------------------------------------------------


def _json_of(expression: str) -> str:
    """An expression whose value, awaited, comes back as JSON text: both
    protocols then return one string, whatever the page computed."""
    return f"(async () => JSON.stringify(await ({expression})))()"


class _Browser:
    """What both browsers share: the application and its protocol
    connection, settling a page just opened, and closing."""

    name: str
    _close_command: str
    app: App
    _rpc: _Rpc
    version: str

    def evaluate(self, page: str, expression: str, timeout: float = 60.0):
        raise NotImplementedError

    def _settle(self, page: str, timeout: float) -> None:
        """Wait until `page` has left the blank page it opens on and has a
        parsed document to evaluate in: a page just opened, or redirected,
        may not."""
        _wait_for("the page's document",
                  lambda: self.evaluate(page, "location.href !== 'about:blank' && "
                                              "document.readyState !== 'loading'"),
                  timeout)

    def close(self) -> None:
        try:
            self._rpc.call(self._close_command, timeout=10.0)
            self._rpc.close()
        except (OSError, BrowserError):
            pass
        self.app.close()

    def kill(self) -> None:
        self.app.kill()


class Chrome(_Browser):
    """Google Chrome over the DevTools Protocol."""

    name, _close_command = "chrome", "Browser.close"

    def __init__(self, headless: bool = False, timeout: float = 30.0):
        def argv(profile: str) -> list[str]:
            return ["google-chrome", f"--user-data-dir={profile}", "--remote-debugging-port=0",
                    "--no-first-run", "--no-default-browser-check", "--start-maximized",
                    "--autoplay-policy=no-user-gesture-required", "--disable-sync",
                    *(["--headless=new"] if headless else []), "about:blank"]

        self.app = App(self.name, argv)
        try:
            active = Path(self.app.directory) / "DevToolsActivePort"
            port, path = _wait_for("Chrome's DevTools port",
                                   lambda: active.read_text().split("\n")[:2], timeout)
            self._rpc = _Rpc(WebSocket(f"ws://127.0.0.1:{port}{path}", timeout))
            self.version = self._rpc.call("Browser.getVersion")["product"]
        except BaseException:
            self.app.close()
            raise
        self._sessions: dict[str, str] = {}

    def open(self, url: str, new_window: bool = True, timeout: float = 60.0) -> str:
        """Open `url`, in a window of its own by default, where a background
        tab would not render, and wait until its document is parsed.
        Returns the page's handle."""
        page = self._rpc.call("Target.createTarget", {"url": url, "newWindow": new_window})[
            "targetId"]
        self._settle(page, timeout)
        return page

    def frame(self, page: str, url_part: str, timeout: float = 60.0) -> str:
        """The handle of the frame in `page` whose URL contains `url_part`,
        to evaluate in: a cross-site frame is a target of its own."""
        def within(target: str, parents: dict[str, str]) -> bool:
            while target in parents:
                target = parents[target]
                if target == page:
                    return True
            return False

        def find():
            targets = self._rpc.call("Target.getTargets")["targetInfos"]
            parents = {t["targetId"]: t["parentId"] for t in targets if t.get("parentId")}
            return next((t["targetId"] for t in targets
                         if t["type"] == "iframe" and url_part in t["url"]
                         and within(t["targetId"], parents)), None)
        return _wait_for(f"a frame of {url_part}", find, timeout)

    def evaluate(self, page: str, expression: str, timeout: float = 60.0):
        """The value of `expression`, awaited, in `page`, through JSON."""
        if page not in self._sessions:
            self._sessions[page] = self._rpc.call(
                "Target.attachToTarget", {"targetId": page, "flatten": True})["sessionId"]
        result = self._rpc.call("Runtime.evaluate",
                                {"expression": _json_of(expression), "awaitPromise": True,
                                 "returnByValue": True, "timeout": timeout * 1000},
                                sessionId=self._sessions[page], timeout=timeout + 10)
        if "exceptionDetails" in result:
            raise BrowserError(result["exceptionDetails"].get("text", "the page threw"))
        value = result["result"].get("value")
        return None if value is None else json.loads(value)


_FIREFOX_PREFS = {
    "browser.shell.checkDefaultBrowser": False,
    "browser.aboutwelcome.enabled": False,
    "browser.startup.homepage_override.mstone": "ignore",
    "datareporting.policy.dataSubmissionEnabled": False,
    "toolkit.telemetry.reportingpolicy.firstRun": False,
    "media.autoplay.default": 0,
    "media.autoplay.blocking_policy": 0,
    "browser.tabs.warnOnClose": False,
    "browser.warnOnQuit": False,
}

#: A snap's Firefox has a /tmp of its own and cannot read another; the one
#: directory it and its user share is its own common directory.
_SNAP_FIREFOX = Path.home() / "snap" / "firefox" / "common"


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class Firefox(_Browser):
    """Mozilla Firefox over WebDriver BiDi."""

    name, _close_command = "firefox", "browser.close"

    def __init__(self, headless: bool = False, timeout: float = 60.0):
        # Chosen here, as Firefox cannot report a port it chose. Another
        # process taking it first would show as a connection that times out.
        port = _free_port()

        def argv(profile: str) -> list[str]:
            prefs = "".join(f"user_pref({json.dumps(k)}, {json.dumps(v)});\n"
                            for k, v in _FIREFOX_PREFS.items())
            (Path(profile) / "user.js").write_text(prefs)
            return ["firefox", "--profile", profile, "--no-remote", "--new-instance",
                    f"--remote-debugging-port={port}", *(["--headless"] if headless else []),
                    "about:blank"]

        self.app = App(self.name, argv, under=_SNAP_FIREFOX if _SNAP_FIREFOX.is_dir() else None)
        try:
            ws = _wait_for("Firefox's WebDriver BiDi port",
                           lambda: WebSocket(f"ws://127.0.0.1:{port}/session", timeout), timeout)
            self._rpc = _Rpc(ws)
            capabilities = self._rpc.call("session.new", {"capabilities": {}})["capabilities"]
            self.version = f"Firefox/{capabilities.get('browserVersion', '?')}"
        except BaseException:
            self.app.close()
            raise

    def open(self, url: str, new_window: bool = True, timeout: float = 60.0) -> str:
        """As Chrome.open."""
        context = self._rpc.call("browsingContext.create",
                                 {"type": "window" if new_window else "tab"})["context"]
        self._rpc.call("browsingContext.navigate",
                       {"context": context, "url": url, "wait": "interactive"}, timeout=timeout)
        self._settle(context, timeout)
        return context

    def frame(self, page: str, url_part: str, timeout: float = 60.0) -> str:
        """As Chrome.frame."""
        def find():
            pending = self._rpc.call("browsingContext.getTree", {"root": page})["contexts"]
            while pending:
                context = pending.pop()
                if url_part in context.get("url", ""):
                    return context["context"]
                pending.extend(context.get("children") or [])
            return None
        return _wait_for(f"a frame of {url_part}", find, timeout)

    def evaluate(self, page: str, expression: str, timeout: float = 60.0):
        """As Chrome.evaluate."""
        result = self._rpc.call("script.evaluate",
                                {"expression": _json_of(expression), "awaitPromise": True,
                                 "target": {"context": page}, "resultOwnership": "none"},
                                timeout=timeout + 10)
        if result.get("type") == "exception":
            raise BrowserError(result.get("exceptionDetails", {}).get("text", "the page threw"))
        value = result.get("result", {}).get("value")
        return None if value is None else json.loads(value)


BROWSERS = {"chrome": Chrome, "firefox": Firefox}
