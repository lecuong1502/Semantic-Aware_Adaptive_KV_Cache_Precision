"""llama.cpp, run as an external baseline (#211): its server and its log.

Every other condition runs inside MicroInfer, so a reader cannot tell how an
engine people deploy fares under the same contention. llama.cpp is that
engine. It is a process of its own that nothing of MicroInfer's links or
imports (ADR-0001's amendment): this module starts llama-server, sends it
token ids over HTTP, and reads its log.

**What is run.** One slot at the whole context, every layer on the device,
the cache's keys and values at one ggml type (CACHE_TYPES), flash attention,
which a quantised value cache needs and every type is therefore given, and
MicroInfer's own prefill chunk as its batch. --fit is off, so that it holds
what it is asked for rather than what it finds room for, and its cache of
prompts in host memory is off.

**What it holds** is in its log at verbosity 4: the model's, the cache's and
the compute buffers on the device, all allocated as it loads. The cache is
its whole context's from the start, where MicroInfer's grows with its
positions.

**How it ends.** A completion streams its progress, so that a server lost
part way has said how far it got (`Lost`). Whether it was lost for device
memory is read from its log (`out_of_memory`): ggml's allocations that failed, and CUDA's
own error, which ends the process. An allocation of the host's that failed
is not.

The build and the model file are made by tools/build_llamacpp.py, from a
pinned release and from the checkpoint MicroInfer itself runs, and logged
(`logged_build`): a run names them by their hashes.
"""

from __future__ import annotations

import http.client
import json
import re
import socket
import subprocess
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from types import TracebackType

import numpy as np

from . import benchlog

REPO = Path(__file__).resolve().parents[2]
#: Where the build and its model files are kept, outside git (.gitignore).
ROOT = REPO / "data" / "llamacpp"
BINARY = ROOT / "build" / "bin" / "llama-server"

BUILD_KIND = "llamacpp-build"
#: The ggml types its cache is run at, with the bits each holds an element in:
#: 32 elements of q8_0 are 34 bytes and of q4_0 18.
CACHE_TYPES = {"f16": 16.0, "q8_0": 8.5, "q4_0": 4.5}
#: MicroInfer's default prefill chunk, llama.cpp's logical and physical batch.
BATCH = 512
#: Verbose enough to say what its buffers hold.
VERBOSITY = 4
LOAD_TIMEOUT_S = 120.0
#: The longest a completion goes without a word before its server is lost.
SILENCE_TIMEOUT_S = 120.0
EXIT_WAIT_S = 30.0
POLL_S = 0.05
HOST = "127.0.0.1"


class Lost(RuntimeError):
    """The server did not answer, or stopped answering: it died, refused the
    completion, or went silent."""


def gguf_path(model: str) -> Path:
    """The model file a build keeps for `model`, its weights at FP16."""
    return ROOT / f"{model}-f16.gguf"


def command(binary: str | Path, gguf: str | Path, *, port: int, context: int,
            cache_type: str) -> list[str]:
    """llama-server's command line for a run."""
    if cache_type not in CACHE_TYPES:
        raise ValueError(f"a cache is run at one of {', '.join(CACHE_TYPES)}; got {cache_type}")
    return [str(binary), "-m", str(gguf), "-c", str(context), "-np", "1", "-ngl", "99",
            "-b", str(BATCH), "-ub", str(BATCH), "-fa", "on",
            "-ctk", cache_type, "-ctv", cache_type, "--fit", "off", "--cache-ram", "0",
            "-lv", str(VERBOSITY), "--host", HOST, "--port", str(port)]


_BUFFER = re.compile(r"\bCUDA\d+\s+(\w+) buffer size =\s*([\d.]+) MiB")


def device_buffers_mib(log: str) -> dict[str, float]:
    """What the server's log says it holds on the device, by buffer; the
    host's buffers (CPU_Mapped, CUDA_Host) are left out. ValueError if the
    log names no cache on the device: the model was not offloaded."""
    found: dict[str, float] = {}
    for kind, mib in _BUFFER.findall(log):
        found[kind] = round(found.get(kind, 0.0) + float(mib), 2)
    if "KV" not in found:
        raise ValueError("the server's log names no KV buffer on the device")
    return found


_FAILED = re.compile(r"out of memory|failed to allocate")
_HOST = re.compile(r"pinned|prompt cache|host", re.IGNORECASE)


def out_of_memory(log: str) -> bool:
    """Whether the server's log says an allocation on the device failed."""
    return any(_FAILED.search(line) and not _HOST.search(line) for line in log.splitlines())


def complete(port: int, ids: Sequence[int] | np.ndarray, new_tokens: int,
             report: Callable[[int], None]) -> dict:
    """A completion of `new_tokens` after `ids`, greedy and past any end of
    sequence, over the server's one slot, whose cache it continues where the
    two share a prefix. `report` is called with the positions the cache holds,
    after each batch of the prompt and after each token decoded. Returns the
    server's last event, with its timings. Raises Lost."""
    body = json.dumps({"prompt": [int(t) for t in ids], "n_predict": int(new_tokens),
                       "temperature": 0.0, "ignore_eos": True, "cache_prompt": True,
                       "id_slot": 0, "stream": True, "return_progress": True,
                       "return_tokens": True})
    connection = http.client.HTTPConnection(HOST, port, timeout=SILENCE_TIMEOUT_S)
    try:
        connection.request("POST", "/completion", body, {"Content-Type": "application/json"})
        response = connection.getresponse()
        if response.status != 200:
            raise Lost(f"the server answered {response.status}: "
                       f"{response.read().decode(errors='replace').strip()}")
        decoded = 0
        for line in response:
            if line.startswith(b"error:"):
                line = b"data: {\"error\":" + line.partition(b":")[2].strip() + b"}"
            if not line.startswith(b"data:"):
                continue
            event = json.loads(line.partition(b":")[2])
            if "error" in event:
                raise Lost(f"the server answered an error: {event['error']}")
            if event.get("stop"):
                return event
            if "prompt_progress" in event:
                report(int(event["prompt_progress"]["processed"]))
            else:
                # the first token comes with the prompt's last batch, unfed
                decoded += len(event.get("tokens") or ())
                report(len(ids) + decoded)
    except (OSError, http.client.HTTPException, json.JSONDecodeError) as exc:
        raise Lost(f"the server did not answer: {exc!r}") from exc
    finally:
        connection.close()
    raise Lost("the stream broke off before the completion ended")


def free_port() -> int:
    with socket.socket() as s:
        s.bind((HOST, 0))
        return s.getsockname()[1]


class Server:
    """llama-server as a process, from loaded to stopped; its log is written
    to `log`, and `buffers_mib` is what it said it holds on the device."""

    def __init__(self, binary: str | Path, gguf: str | Path, *, context: int, cache_type: str,
                 log: str | Path):
        self.port = free_port()
        self.command = command(binary, gguf, port=self.port, context=context,
                               cache_type=cache_type)
        self.log_path = Path(log)
        self.buffers_mib: dict[str, float] = {}
        self.load_seconds: float | None = None
        self._proc: subprocess.Popen | None = None

    def __enter__(self) -> Server:
        began = time.monotonic()
        with open(self.log_path, "wb") as out:
            self._proc = subprocess.Popen(self.command, stdout=out, stderr=subprocess.STDOUT)
        try:
            while not self._healthy():
                if self._proc.poll() is not None:
                    raise Lost(f"the server ended as it loaded, with {self._proc.returncode}; "
                               f"its log is {self.log_path}")
                if time.monotonic() - began > LOAD_TIMEOUT_S:
                    raise Lost(f"the server did not load in {LOAD_TIMEOUT_S:.0f} s; its log is "
                               f"{self.log_path}")
                time.sleep(POLL_S)
            self.load_seconds = time.monotonic() - began
            self.buffers_mib = device_buffers_mib(self.log())
        except BaseException:
            self._stop()
            raise
        return self

    def __exit__(self, kind: type[BaseException] | None, exc: BaseException | None,
                 trace: TracebackType | None) -> None:
        self._stop()

    @property
    def returncode(self) -> int | None:
        """What the process ended with; None while it runs."""
        return None if self._proc is None else self._proc.poll()

    def log(self) -> str:
        return self.log_path.read_text(errors="replace")

    def complete(self, ids: Sequence[int] | np.ndarray, new_tokens: int,
                 report: Callable[[int], None]) -> dict:
        return complete(self.port, ids, new_tokens, report)

    def _healthy(self) -> bool:
        connection = http.client.HTTPConnection(HOST, self.port, timeout=1.0)
        try:
            connection.request("GET", "/health")
            return connection.getresponse().status == 200
        except (OSError, http.client.HTTPException):
            return False
        finally:
            connection.close()

    def _stop(self) -> None:
        if self._proc is None or self._proc.poll() is not None:
            return
        self._proc.terminate()
        try:
            self._proc.wait(timeout=EXIT_WAIT_S)
        except subprocess.TimeoutExpired:
            self._proc.kill()
            self._proc.wait()


def logged_build(log: str | Path, binary: str | Path = BINARY,
                 gguf: str | Path | None = None, model: str | None = None) -> dict:
    """The last build logged, its binary and model file the ones on disk by
    their hashes. ValueError if none is, or another is on disk."""
    gguf = gguf_path(model) if gguf is None else Path(gguf)
    builds = [e for e in benchlog.read(log) if e["kind"] == BUILD_KIND] \
        if Path(log).exists() else []
    if not builds:
        raise ValueError("no llama.cpp build is logged; run tools/build_llamacpp.py")
    build = builds[-1]["results"]
    for path, key in ((binary, "binary_sha256"), (gguf, "gguf_sha256")):
        if not Path(path).is_file() or benchlog.file_sha256(Path(path)) != build[key]:
            raise ValueError(f"{path} is not the one logged, but another build's, or missing; "
                             f"run tools/build_llamacpp.py")
    return build
