"""The VRAM pressure monitor: polling, levels, hysteresis and events (#61).

RQ2 asks whether a runtime that cooperates with no scheduler can see
pressure coming in time to act. This is what sees it, as #45 decided:

- **A background thread polling device headroom every 50 ms,** the driver's
  free memory by NVML. NVML is called through ctypes, which releases the
  GIL, so the thread does not hold up decoding. Polls keep absolute
  deadlines (recorder.every): a slow read delays only itself.
- **Headroom classified in MiB, not percent,** because spikes are absolute:
  RED below `red_below_bytes`, YELLOW below `yellow_below_bytes`, GREEN
  otherwise. A pressure level is the machine's state (CONTEXT.md), whoever
  took the memory.
- **Hysteresis over K polls.** A level other than the current one must be
  read on K polls in a row before it becomes current, so a dip shorter than
  K polls is never reported and a clean step is reported on its Kth poll,
  within K + 1 polls of its first reading whatever the phase. The first
  poll's level is current at once.
- **Events go to a queue** that the decode loop drains between steps without
  blocking: each transition once, with when it happened, the level before
  and after, and the headroom that settled it.

**The thresholds and K are provisional.** #63 sets them from RQ1's data by a
rule recorded in an ADR; until then PROVISIONAL says so, in its name and its
`provisional` flag, and every event from a monitor using it inherits the
mark through `Monitor.thresholds`. The provisional values follow the rule
#45 proposes, from RQ1's 48 uncensored spikes (#55, #56):

- RED below 512 MiB: a spike at RQ1's P90 amplitude, 506 MiB, would leave
  nothing;
- YELLOW below 1024 MiB: room for two such spikes, time to downgrade
  gradually;
- K = 3 polls, 150 ms: shorter than RQ1's P10 rise time, 242 ms.

Attribution (how much of a change was the engine's own) and the engine's
integration are #62's.
"""

from __future__ import annotations

import enum
import queue
import threading
import time
from dataclasses import dataclass
from typing import Callable

from . import nvml, recorder

MiB = 2**20
POLL_S = 0.05


class Level(str, enum.Enum):
    """A pressure level: the machine's state, never a page's."""

    GREEN = "GREEN"
    YELLOW = "YELLOW"
    RED = "RED"


GREEN, YELLOW, RED = Level.GREEN, Level.YELLOW, Level.RED


@dataclass(frozen=True)
class Thresholds:
    """Where headroom turns YELLOW and RED, in bytes, and K, the polls a new
    level must hold for. `provisional` until #63's ADR sets them."""

    red_below_bytes: int
    yellow_below_bytes: int
    persist_polls: int
    provisional: bool = True

    def __post_init__(self):
        if not 0 <= self.red_below_bytes < self.yellow_below_bytes:
            raise ValueError(f"thresholds nest, 0 <= red < yellow; got red below "
                             f"{self.red_below_bytes}, yellow below {self.yellow_below_bytes}")
        if self.persist_polls < 1:
            raise ValueError(f"a level holds for at least one poll; got {self.persist_polls}")

    def classify(self, headroom_bytes: int) -> Level:
        if headroom_bytes < self.red_below_bytes:
            return RED
        if headroom_bytes < self.yellow_below_bytes:
            return YELLOW
        return GREEN


#: PROVISIONAL until the thresholds ADR (#63); see the module's docstring.
PROVISIONAL = Thresholds(red_below_bytes=512 * MiB, yellow_below_bytes=1024 * MiB,
                         persist_polls=3, provisional=True)


@dataclass(frozen=True)
class PressureEvent:
    """A transition: at poll `poll` (from 0) and `t_mono_ns`, the level went
    from `previous`, None for the first, to `level`, on a reading of
    `headroom_bytes`."""

    t_mono_ns: int
    poll: int
    previous: Level | None
    level: Level
    headroom_bytes: int


def nvml_headroom() -> int:
    """The device's free memory, by the driver's account."""
    return nvml.memory().free


class Monitor:
    """Polls `reader`, headroom in bytes, every `poll_s` seconds on a thread
    of its own, from start() until stop(), or as a context manager."""

    def __init__(self, reader: Callable[[], int] | None = None,
                 thresholds: Thresholds = PROVISIONAL, poll_s: float = POLL_S):
        if poll_s <= 0:
            raise ValueError(f"a poll period is > 0; got {poll_s}")
        self.thresholds, self._reader, self._poll_s = thresholds, reader or nvml_headroom, poll_s
        self._events: queue.SimpleQueue[PressureEvent] = queue.SimpleQueue()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.error: BaseException | None = None
        self.level: Level | None = None
        self.missed = 0

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(self) -> Monitor:
        if self._thread is not None:
            raise RuntimeError("a monitor starts once")
        self._thread = threading.Thread(target=self._run, name="pressure-monitor", daemon=True)
        self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join()

    def __enter__(self) -> Monitor:
        return self.start()

    def __exit__(self, *exc) -> None:
        self.stop()

    def drain(self) -> list[PressureEvent]:
        """Every event not yet drained, oldest first, without blocking. If the
        thread stopped on an error, raises it once the events it left are
        drained."""
        events = []
        while True:
            try:
                events.append(self._events.get_nowait())
            except queue.Empty:
                break
        if not events and self.error is not None:
            raise RuntimeError("the pressure monitor stopped: its reader failed") from self.error
        return events

    def _run(self) -> None:
        polls = 0
        candidate, held = None, 0  # a level other than the current, and its polls in a row

        def poll() -> None:
            nonlocal polls, candidate, held
            t = time.monotonic_ns()
            headroom = self._reader()
            level = self.thresholds.classify(headroom)
            if self.level is None or level == self.level:
                candidate, held = None, 0
                if self.level is None:
                    self._emit(t, polls, level, headroom)
            else:
                held = held + 1 if level == candidate else 1
                candidate = level
                if held >= self.thresholds.persist_polls:
                    self._emit(t, polls, level, headroom)
                    candidate, held = None, 0
            polls += 1

        try:
            self.missed = recorder.every(1 / self._poll_s, None, self._stop, poll)
        except Exception as exc:  # noqa: BLE001 - raised by drain, on the caller's thread
            self.error = exc

    def _emit(self, t: int, poll: int, level: Level, headroom: int) -> None:
        self._events.put(PressureEvent(t, poll, self.level, level, headroom))
        self.level = level
