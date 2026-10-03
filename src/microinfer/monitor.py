"""The VRAM pressure monitor: polling, levels, hysteresis and events (#61).

RQ2 asks whether a runtime that cooperates with no scheduler can see
pressure coming in time to act. This is what sees it, as #45 decided:

- **A background thread polling device headroom every 50 ms,** the driver's
  free memory by NVML. Polls keep absolute deadlines (recorder.every): a slow
  read delays only itself. The thread runs beside decoding because the
  extension releases the GIL while the device works (ADR-0012); ctypes
  releasing it during an NVML call is only the other half.
- **Headroom classified in MiB, not percent,** because spikes are absolute:
  RED below `red_below_bytes`, YELLOW below `yellow_below_bytes`, GREEN
  otherwise. A pressure level is the machine's state (CONTEXT.md), whoever
  took the memory.
- **Hysteresis over K polls** (Hysteresis). A new level is reported only
  once K polls in a row have read the same side of the current one, so a
  drop shorter than K polls is never reported and a clean level change is
  reported on its Kth poll, within K + 1 polls of its first reading whatever
  the phase. It becomes the level of those K polls nearest the current one,
  the level headroom kept throughout: readings that flicker between YELLOW
  and RED from GREEN report YELLOW, and RED only once RED alone holds for K
  more. The first poll's level is current at once, and reported, so that a
  consumer knows where the machine started.
- **Events go to a queue** that the decode loop drains between steps without
  blocking: each transition once, with when it happened, the level before
  and after, and the headroom that settled it. If the reader fails, the
  thread stops and drain raises its error once the events before it are
  drained: a monitor that died quietly would read as a machine at GREEN.

**The thresholds and K are ADR-0013's**, DEFAULT: set from RQ1's data by
the rule pressure_rule applies, which a test holds them to. RED below 512
MiB, where a spike at RQ1's P90 amplitude could run the machine out; YELLOW
below 1024 MiB, room for what a fast spike takes while it is detected and a
plan applied, here the whole spike; K = 3 polls, detection within RQ1's P10
rise time.

**Each transition carries its own/others split** (#62, MemorySplit): this
process's device memory and every other process's, by the driver's account
of each (nvml.own_used_bytes, others_used_bytes), and how much each changed
since the transition before, beside the headroom's own change. The monitor
runs in the engine's process, so "own" is everything that process holds:
the engine's weights, cache and workspace, and its CUDA context. The split
is read at transitions only, after the poll that completed one: a process
query costs far more than a headroom reading, and a poll must stay cheap.
So its change runs from one transition to the next, not from when the
change began. Headroom also moves by the driver's own memory, so the two
changes need not sum to the headroom's. A split the driver cannot give is
None; it never stops the monitor.

The engine starts a monitor, drains its events between steps and records
them (Engine.start_monitor); an adaptive engine also plans and downgrades
on YELLOW and RED (#105).
"""

from __future__ import annotations

import enum
import queue
import threading
import time
from dataclasses import dataclass, replace
from typing import Callable

from . import nvml, recorder
from .footprint import MIB

POLL_S = 0.05


class Level(str, enum.Enum):
    """A pressure level: the machine's state, never a page's."""

    GREEN = "GREEN"
    YELLOW = "YELLOW"
    RED = "RED"


GREEN, YELLOW, RED = Level.GREEN, Level.YELLOW, Level.RED
_SEVERITY = {GREEN: 0, YELLOW: 1, RED: 2}


@dataclass(frozen=True)
class Thresholds:
    """Where headroom turns YELLOW and RED, in bytes, and K, the polls a new
    level must hold for."""

    red_below_bytes: int
    yellow_below_bytes: int
    persist_polls: int

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


#: ADR-0013's, the rule of pressure_rule applied to RQ1's log entries.
DEFAULT = Thresholds(red_below_bytes=512 * MIB, yellow_below_bytes=1024 * MIB, persist_polls=3)


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
    #: Read at the transition; None if the driver could not give it.
    split: MemorySplit | None = None
    #: Since the transition before; None for the first.
    split_change: MemorySplit | None = None
    headroom_change_bytes: int | None = None


@dataclass(frozen=True)
class MemorySplit:
    """Device memory split between this process, the engine's, and every
    other process; either None where the driver did not report it."""

    own_bytes: int | None
    others_bytes: int | None

    def minus(self, before: MemorySplit | None) -> MemorySplit | None:
        if before is None:
            return None

        def change(now: int | None, then: int | None) -> int | None:
            return None if now is None or then is None else now - then

        return MemorySplit(change(self.own_bytes, before.own_bytes),
                           change(self.others_bytes, before.others_bytes))


def nvml_headroom() -> int:
    """The device's free memory, by the driver's account."""
    return nvml.memory().free


def nvml_split() -> MemorySplit:
    """This process's device memory and every other process's. Own is 0 if
    the driver does not list this process, which then holds nothing on the
    device; None if it lists it without its memory, or a query fails."""
    def read(query: Callable[[], int]) -> int | None:
        try:
            return query()
        except Exception:  # noqa: BLE001 - a split the driver cannot give is None
            return None

    return MemorySplit(read(nvml.own_used_bytes), read(nvml.others_used_bytes))


class Hysteresis:
    """The levels a series of headroom readings settles on, one poll at a
    time: feed() returns the transition a reading completes, if any."""

    def __init__(self, thresholds: Thresholds):
        self.thresholds = thresholds
        self.level: Level | None = None
        self._polls = 0
        self._side = 0  # +1 worse than the current level, -1 better, 0 neither
        self._run = 0  # the polls in a row on that side
        self._nearest: Level | None = None  # of those, the level nearest the current

    def feed(self, t_mono_ns: int, headroom_bytes: int) -> PressureEvent | None:
        poll, self._polls = self._polls, self._polls + 1
        level = self.thresholds.classify(headroom_bytes)
        if self.level is None:
            return self._settle(t_mono_ns, poll, level, headroom_bytes)
        change = _SEVERITY[level] - _SEVERITY[self.level]
        side = (change > 0) - (change < 0)
        if side == 0 or side != self._side:
            self._side, self._run, self._nearest = side, int(side != 0), level
        else:
            self._run += 1
            self._nearest = min(self._nearest, level,
                                key=lambda x: abs(_SEVERITY[x] - _SEVERITY[self.level]))
        if self._run >= self.thresholds.persist_polls:
            return self._settle(t_mono_ns, poll, self._nearest, headroom_bytes)
        return None

    def _settle(self, t: int, poll: int, level: Level, headroom: int) -> PressureEvent:
        event = PressureEvent(t, poll, self.level, level, headroom)
        self.level, self._side, self._run, self._nearest = level, 0, 0, None
        return event


class Monitor:
    """Polls `reader`, headroom in bytes, every `poll_s` seconds on a thread
    of its own, from start() until stop(), or as a context manager; at each
    transition, reads `split`, this process's and the others' memory."""

    def __init__(self, reader: Callable[[], int] | None = None,
                 thresholds: Thresholds = DEFAULT, poll_s: float = POLL_S,
                 split: Callable[[], MemorySplit] | None = None):
        if poll_s <= 0:
            raise ValueError(f"a poll period is > 0; got {poll_s}")
        self.thresholds, self._reader, self._poll_s = thresholds, reader or nvml_headroom, poll_s
        self._split = split or nvml_split
        self._events: queue.SimpleQueue[PressureEvent] = queue.SimpleQueue()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.error: BaseException | None = None
        self.level: Level | None = None
        self.missed = 0  # the polls whose deadlines passed unread, once stopped

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
            raise RuntimeError("the pressure monitor stopped: its headroom reader "
                               "failed") from self.error
        return events

    def _run(self) -> None:
        hysteresis = Hysteresis(self.thresholds)
        last: PressureEvent | None = None

        def poll() -> None:
            nonlocal last
            event = hysteresis.feed(time.monotonic_ns(), self._reader())
            if event is None:
                return
            try:
                split = self._split()
            except Exception:  # noqa: BLE001 - a split is information; pressure goes on
                split = None
            event = replace(event, split=split)
            if last is not None:
                event = replace(
                    event, split_change=None if split is None else split.minus(last.split),
                    headroom_change_bytes=event.headroom_bytes - last.headroom_bytes)
            last = event
            self.level = event.level
            self._events.put(event)

        try:
            self.missed = recorder.every(1 / self._poll_s, duration=None, stop=self._stop,
                                         sample=poll)
        except Exception as exc:  # noqa: BLE001 - raised by drain, on the caller's thread
            self.error = exc
