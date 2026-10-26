"""The survival experiment (#108): the run, and the logic it is made of.

A generation fills a long context at FP16 while the contention simulator,
a process of its own, takes memory part way through and keeps it, as most of
RQ1's spikes kept theirs (ADR-0014). The simulator takes enough that a
static engine runs out before its cache is full; an adaptive one is to
downgrade, give the memory back to the driver, and finish.

**How much the simulator takes** is set by a rule, not a number, so that a
static run and an adaptive run face the same contention whatever else the
desktop holds that day. When the cache reaches the position the experiment
chooses, the engine still has the rest of its cache to grow; the simulator
takes what leaves it `shortfall` short of that.

**What a run comes to** is whether it survived, how it ended, and every plan
the engine made: the memory each returned, by the cache's allocator and by
the driver's account of the engine's process, whether the two agree to a
granule (#88's gate), and how long it took; and how long each pressure
event waited for the step boundary at which the engine drained it (#135).

**The run** pauses the generation at the chosen position until the
simulator has taken its memory, so that contention arrives at the same place
in every run however fast the engine and the simulator start.

**A pulse and release** (#155) is the run with the simulator giving its
memory back `pulse_s` after it took it, the generation going on: pressure
passes, and once the cooldown has run the engine's upgrades restore the
pages from their shadows (#106). A run samples, at every step, what its
cache holds, its effective bits and its pages below FP16, and `recovery`
gives what that comes to: the time from the end of pressure to every page
back at FP16, the effective bits over time, and any thrash, downgrades and
upgrades alternating.

**A scheduled scenario** (#190) is the run with the simulator playing a
schedule in S5's step's place, S1 to S4 of the survival matrix
(survival_matrix): made from the headroom at the take, the run ending with
the generation and cutting the schedule there. A static INT4 or INT2
engine's headroom is counted as an FP16 engine's would be, so that every
condition faces the same bytes taken. A run's monitor is scored against
the recorder's trace (`detection`), as RQ2 scored it (evaluation).
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import NamedTuple

import numpy as np

from . import _microinfer, evaluation, nvml, recorder, replay
from .config import ModelConfig
from .contention import spread
from .engine import COMPLETE, Ending, Engine, PlanRecord, PressureRecord, paged_cache_bytes
from .footprint import MIB
from .monitor import PressureEvent, Thresholds
from .schedule import Schedule

#: How a static run that ran out ends: generate raises, it does not end.
OUT_OF_MEMORY = "out of memory"

#: How long the simulator waits, its context in place, before it takes.
LEAD_S = 0.2
#: The longest a generation waits at the chosen position for the take.
TAKE_TIMEOUT_S = 60.0
#: The longest the simulator keeps what it took, the generation ending
#: first: a 32K prefill and its decoding take minutes, not a day.
KEEP_S = 24 * 3600.0


def contention_bytes(cfg: ModelConfig, *, headroom_bytes: int, positions: int, context: int,
                     simulator_context_bytes: int, shortfall_bytes: int) -> int:
    """What the simulator takes, beyond its own CUDA context, so that an
    engine with `positions` of a `context`-position FP16 cache and
    `headroom_bytes` free would end `shortfall_bytes` short of its full
    cache: headroom, less the simulator's context, still to come, less what
    the cache has still to grow, plus the shortfall. Rounded up to a
    granule, the unit the simulator takes memory in, so that the shortfall
    is never less than asked. Raises ValueError if the engine is short
    already, with nothing taken."""
    still_to_grow = paged_cache_bytes(cfg, context) - paged_cache_bytes(cfg, positions)
    room = headroom_bytes - simulator_context_bytes - still_to_grow
    if room < 0:
        raise ValueError(f"the engine is short already: {room / MIB:.0f} MiB with nothing "
                         f"taken")
    granule = _microinfer.granule_bytes()
    return -(-(room + shortfall_bytes) // granule) * granule


def as_fp16(cfg: ModelConfig, headroom_bytes: int, *, positions: int, kv_tier: str) -> int:
    """The headroom an FP16 engine would leave where one at `kv_tier`
    leaves `headroom_bytes`, its cache at `positions`: less by what the
    FP16 cache would hold beyond it."""
    return headroom_bytes - (paged_cache_bytes(cfg, positions)
                             - paged_cache_bytes(cfg, positions, kv_tier))


def step(nbytes: int, pulse_s: float | None = None) -> Schedule:
    """S5's step: `nbytes` taken, and kept, or given back `pulse_s` after."""
    return Schedule([(0.0, nbytes)] + ([] if pulse_s is None else [(pulse_s, 0)]))


def summarise(plans: Sequence[PlanRecord], *, ending: Ending | None = None,
              error: str | None = None, positions_reached: int | None = None) -> dict:
    """What a run comes to. An adaptive run, or a static one that finished,
    passes its `ending`; a static run that ran out passes the `error` it
    raised and the positions its cache last reported. It survived if it
    ended COMPLETE."""
    if ending is not None:
        ended = {"state": ending.state, "positions": ending.positions,
                 "tokens": len(ending.tokens), "reason": ending.reason}
    else:
        ended = {"state": OUT_OF_MEMORY, "positions": positions_reached, "error": error}
    granule = _microinfer.granule_bytes()
    rows = [_plan(record, granule) for record in plans]
    seen = [r["seen_mib"] for r in rows]
    agreed = [r["seen_within_granule"] for r in rows]
    totals = {"plans": len(rows),
              "emergency_plans": sum(r["emergency"] is not None for r in rows),
              "downgrades_applied": sum(r["applied"] for r in rows if not r["upgrades"]),
              "upgrades_applied": sum(r["applied"] for r in rows if r["upgrades"]),
              "returned_mib": sum(r["returned_mib"] for r in rows),
              "seen_mib": None if None in seen else sum(seen),
              "every_plan_seen_within_granule": None if not rows or None in agreed
              else all(agreed)}
    return {"survived": ending is not None and ending.state == COMPLETE, "ending": ended,
            "plans": rows, "totals": totals}


def _plan(record: PlanRecord, granule: int) -> dict:
    """One plan: what it was for, what became of it, the memory it returned
    and the time it took. The driver saw what the cache returned if, batch
    by batch, the two differ by a granule at most. Its time from the event to its last batch is
    given for a plan made on its event, not for one made because pressure
    persisted or an allocation failed."""
    batches = record.batches
    returned = sum(b.cache_bytes_before - b.cache_bytes_after for b in batches)
    measured = all(b.own_bytes_before is not None and b.own_bytes_after is not None
                   for b in batches)
    seen = sum(b.own_bytes_before - b.own_bytes_after for b in batches) if measured else None
    agreed = all(abs((b.own_bytes_before - b.own_bytes_after)
                     - (b.cache_bytes_before - b.cache_bytes_after)) <= granule
                 for b in batches) if measured else None
    on_event = (record.pressure is not None and batches and not record.persisting
                and record.emergency is None)
    plan = record.plan
    return {"level": plan.level.value, "emergency": record.emergency,
            "persisting": record.persisting, "upgrades": record.upgrades,
            "ended": record.ended, "headroom_mib": record.headroom_bytes / MIB,
            "target_mib": getattr(plan, "target_bytes", 0) / MIB,
            "short": getattr(plan, "short", False), "moves": len(plan),
            "applied": record.applied, "skipped": record.skipped, "batches": len(batches),
            "planning_ms": round(record.planning_seconds * 1e3, 3),
            "applying_ms": round(sum(b.seconds for b in batches) * 1e3, 3),
            "returned_mib": returned / MIB, "seen_mib": None if seen is None else seen / MIB,
            "seen_within_granule": agreed,
            "event_to_applied_ms": round((batches[-1].at_seconds
                                          - record.pressure.event.t_mono_ns / 1e9) * 1e3, 3)
            if on_event else None}


def waits(pressure_events: Sequence[PressureRecord], plans: Sequence[PlanRecord]) -> dict:
    """Each pressure event, with how long it waited from the monitor's
    transition to the step boundary at which the engine drained it, and its
    time to its plan: that wait and the plan's making, for an event a plan
    was made on (#135). With the median, P90 and max of each, over the
    YELLOW and RED events, the ones a plan answers. A persisting or an
    emergency plan is made on no event."""
    made = {id(p.pressure): p.planning_seconds for p in plans
            if p.pressure is not None and not p.persisting and p.emergency is None}
    rows = []
    for r in pressure_events:
        wait_ms = round(r.waited_ns / 1e6, 3)
        planning = made.get(id(r))
        event = r.event
        rows.append({"level": event.level.value,
                     "previous": None if event.previous is None else event.previous.value,
                     "headroom_mib": event.headroom_bytes / MIB, "positions": r.positions_held,
                     "wait_ms": wait_ms,
                     "to_plan_ms": None if planning is None
                     else round(wait_ms + planning * 1e3, 3)})
    pressed = [r for r in rows if r["level"] in ("YELLOW", "RED")]
    return {"events": rows, "pressure_wait_ms": spread([r["wait_ms"] for r in pressed]),
            "to_plan_ms": spread([r["to_plan_ms"] for r in pressed
                                  if r["to_plan_ms"] is not None])}


@dataclass
class Run:
    """A run of the experiment: what it came to, the contention it met, and
    the pressure events the engine drained."""

    summary: dict
    #: The position the cache had reached when the simulator took memory.
    contention_position: int
    #: The headroom then, before the simulator started, and what it took.
    headroom_bytes: int
    taken_bytes: int
    #: When the take began, on the recorder's clock (replay's return).
    started_ns: int
    pressure_events: list[PressureRecord] = field(default_factory=list)
    #: What the cache held at each step, from the generation's first report
    #: (`sample`), and when the simulator had taken its memory, on
    #: time.monotonic(), the clock of the engine's plans.
    timeline: list[Step] = field(default_factory=list)
    took_at_seconds: float | None = None
    #: When pressure last ended, GREEN after YELLOW or RED, on that clock.
    pressure_ended_seconds: float | None = None
    #: How long the simulator kept what it took, or None, to the end.
    pulse_s: float | None = None
    #: What the simulator played: S5's step, or a scenario's schedule.
    schedule: Schedule | None = None


class Step(NamedTuple):
    """What a generation's cache held at a step, when it reported one: the
    time, on time.monotonic(), the state and position reported, and the
    cache's effective bits and (layer, page)s below FP16 (Engine.held_bits),
    None before it holds a full page."""

    t: float
    state: str
    position: int
    effective_bits: float | None
    pages_below_fp16: int | None


def sample(engine: Engine, state: str, position: int) -> Step:
    bits = engine.held_bits() or {}
    return Step(time.monotonic(), state, position, bits.get("effective_bits"),
                bits.get("pages_below_fp16"))


def run(engine: Engine, prompt: np.ndarray, new_tokens: int, *, contention_at: int,
        out: str | Path, simulator_context_bytes: int, shortfall_bytes: int | None = None,
        pulse_s: float | None = None,
        schedule: Callable[[int], Schedule] | None = None) -> Run:
    """Generate `new_tokens` after `prompt` with the engine's monitor on and
    plans measured, while the recorder records the device to `out`. When
    the cache first reports `contention_at` positions or more, the
    simulator takes contention_bytes for `shortfall_bytes` and keeps it
    until the generation ends, or, with `pulse_s`, for that long, and then
    gives it back; or, given a `schedule` in their place, plays what it
    makes of the headroom then, before the simulator's context, until the
    generation ends. Either way the headroom is as an FP16 engine's would
    be (as_fp16), and the generation waits there until the simulator has
    taken its first level. Every step is sampled into the run's timeline.
    A static engine that runs out raises OutOfMemory, which is caught and
    summarised; an adaptive one ends as its `ending` says."""
    if (schedule is None) == (shortfall_bytes is None) or (
            schedule is not None and pulse_s is not None):
        raise ValueError("a run's contention is a shortfall, pulsed or kept, or a schedule")
    context = len(prompt) + new_tokens - 1
    simulator = Simulator(out)
    took: dict = {}
    reached = [0]
    timeline: list[Step] = []

    def report(state: str, position: int, token: int | None) -> None:
        reached[0] = position
        timeline.append(sample(engine, state, position))
        if took or position < contention_at:
            return
        took["position"], took["headroom"] = position, nvml.memory().free
        headroom = as_fp16(engine.config, took["headroom"], positions=position,
                           kv_tier=engine.kv_tier)
        if schedule is None:
            nbytes = contention_bytes(engine.config, headroom_bytes=headroom,
                                      positions=position, context=context,
                                      simulator_context_bytes=simulator_context_bytes,
                                      shortfall_bytes=shortfall_bytes)
            played = step(nbytes, pulse_s)
        else:
            played = schedule(headroom)
        took["schedule"] = played
        simulator.play(played)
        took["at_seconds"] = time.monotonic()

    engine.measure_plans = True
    engine.start_monitor()
    ending = error = None
    try:
        engine.generate(prompt, new_tokens, stop_at_eos=False, report=report)
        ending = engine.ending
    except _microinfer.OutOfMemory as exc:
        if engine.kv_adaptive:
            raise
        error = str(exc)
    finally:
        engine.stop_monitor()
        simulator.close()
    if not took:
        raise RuntimeError(f"the cache never reached {contention_at} positions: no contention")
    return Run(summarise(engine.plans, ending=ending, error=error, positions_reached=reached[0]),
               took["position"], took["headroom"], took["schedule"].points[0][1],
               simulator.started_ns, list(engine.pressure_events), timeline,
               took["at_seconds"], engine.pressure_ended_seconds, pulse_s, took["schedule"])


def detection(out: str | Path, events: Sequence[PressureEvent], thresholds: Thresholds,
              started_ns: int) -> dict:
    """The monitor's RED `events` scored against the true RED episodes of
    the trace at `out`, from when the simulator began to the trace's end,
    as RQ2 scored it (evaluation.score): the engine's own growth moves
    headroom too, and counts as the simulator's does."""
    _, samples = recorder.read(out)
    t = samples["t_mono_ns"]
    episodes = evaluation.true_red(t, samples["free_bytes"], thresholds)
    return evaluation.summary(evaluation.score(episodes, list(events), thresholds,
                                               within=(started_ns, int(t[-1]) + 1)))


def recovery(run: Run, plans: Sequence[PlanRecord], *, cooldown_seconds: float) -> dict:
    """What a pulse and release comes to, its times in seconds from when
    the simulator had taken its memory:
    - when it gave it back, `pulse_s` later, as it was scheduled, and when
      pressure last ended, the monitor's GREEN after it, which lags the
      release by the monitor's polls;
    - the last downgrade applied, and the first and last upgrade applied;
      whether the first came the cooldown after the end of pressure or the
      last downgrade before it, the later, as the engine's cooldown runs
      (for one pulse: under several, pressure that ended before the last
      is not known);
    - the **time to restore**: from the end of pressure to the first step
      after it whose cache holds no page below FP16, and from the release;
      None if no step did;
    - the **effective bits**, each step's held until the next, averaged over
      time from the take to the last step, under pressure and after it, and
      the lowest;
    - **thrash**: how often the applied plans, in the order they were first
      applied, turn from downgrades to upgrades or back (one, for a pulse
      restored), the downgrade plans applied after the first upgrade, and
      the upgrade plans pressure cancelled."""
    took = run.took_at_seconds

    def since(t: float | None) -> float | None:
        return None if t is None else t - took

    applied = sorted((p.batches[0].at_seconds, p.upgrades, p) for p in plans if p.batches)
    downs = [b.at_seconds for p in plans if not p.upgrades for b in p.batches]
    ups = [b.at_seconds for p in plans if p.upgrades for b in p.batches]
    first_up = min(ups, default=None)
    ended = run.pressure_ended_seconds
    released = None if run.pulse_s is None else took + run.pulse_s
    restored_at = None if ended is None else next(
        (step.t for step in run.timeline if step.t >= ended and step.pages_below_fp16 == 0),
        None)
    cooldown_from = None
    if first_up is not None:
        cooldown_from = max([t for t in (ended, *downs) if t is not None and t <= first_up],
                            default=None)
    kinds = [up for _, up, _ in applied]
    held = [step for step in run.timeline if step.effective_bits is not None]
    # From the step the take came at, whose bits it held: the steps after it.
    first = max([i for i, step in enumerate(held) if step.t <= took], default=0)
    steps = held[first:]
    return {
        "released_s": run.pulse_s, "pressure_ended_s": since(ended),
        "last_downgrade_s": since(max(downs, default=None)),
        "first_upgrade_s": since(first_up), "last_upgrade_s": since(max(ups, default=None)),
        "cooldown_s": cooldown_seconds,
        "first_upgrade_after_cooldown": None if cooldown_from is None
        else first_up - cooldown_from >= cooldown_seconds,
        "restored": restored_at is not None,
        "time_to_restore_s": None if restored_at is None else restored_at - ended,
        "time_to_restore_from_release_s": None if restored_at is None or released is None
        else restored_at - released,
        "effective_bits": {
            "mean": _mean_over_time(steps, took, None),
            "mean_under_pressure": None if ended is None else _mean_over_time(steps, took, ended),
            "mean_after_pressure": None if ended is None else _mean_over_time(steps, ended, None),
            "lowest": min((step.effective_bits for step in steps), default=None)},
        "thrash": {
            "switches": sum(a != b for a, b in zip(kinds, kinds[1:])),
            "downgrade_plans_after_first_upgrade": 0 if first_up is None else sum(
                1 for t, up, _ in applied if not up and t > first_up),
            "upgrade_plans_cancelled_by_pressure": sum(
                1 for p in plans if p.upgrades and p.ended == "cancelled by pressure")}}


def _mean_over_time(steps: Sequence[Step], start: float, end: float | None) -> float | None:
    """The effective bits of `steps`, each held from its time to the next
    step's, averaged over [start, end), end the last step's time if None."""
    times = [step.t for step in steps]
    end = times[-1] if end is None and times else end
    total = weight = 0.0
    for i, step in enumerate(steps):
        held_from = max(step.t, start)
        held_to = min(times[i + 1] if i + 1 < len(steps) else end, end)
        if held_to > held_from:
            total += step.effective_bits * (held_to - held_from)
            weight += held_to - held_from
    return None if weight == 0 else total / weight


class Simulator:
    """The contention simulator, a process of its own, run on a thread of
    this one while the recorder records the device to `out` (replay):
    `take` starts it and returns once it has taken its memory, which it
    keeps until `close`, or, with a pulse, gives back that long after;
    `play` plays a schedule the same way, until it ends or `close`."""

    def __init__(self, out: str | Path):
        self.out = Path(out)
        #: When the take began, on the recorder's clock; None before it.
        self.started_ns: int | None = None
        self._done = threading.Event()
        self._failure: list[BaseException] = []
        self._thread: threading.Thread | None = None

    def take(self, nbytes: int, *, pulse_s: float | None = None) -> None:
        """Take `nbytes` beyond the simulator's own context, waiting until
        it has; with `pulse_s`, give them back that long after."""
        self.play(step(nbytes, pulse_s))

    def play(self, schedule: Schedule) -> None:
        """Play `schedule`, beyond the simulator's own context, waiting until
        it has taken its first level; `close` cuts it short."""
        if self._thread is not None:
            raise RuntimeError("a simulator takes once")

        def contend() -> None:
            try:
                self.started_ns = replay.replay(schedule, self.out, lead_s=LEAD_S,
                                                tail_s=KEEP_S, stop=self._done,
                                                cut_by_stop=True)
            except BaseException as exc:  # noqa: BLE001 - raised by the caller's thread
                self._failure.append(exc)

        self._thread = threading.Thread(target=contend, name="contention-simulator")
        self._thread.start()
        _wait_for_take(recorder.companion(self.out, ".events.jsonl"), self._thread,
                       self._failure)

    def close(self) -> None:
        """Stop the simulator, giving back what it holds, and raise what
        stopped it, if anything did."""
        self._done.set()
        if self._thread is not None:
            self._thread.join()
        if self._failure:
            raise self._failure[0]


def _wait_for_take(events: Path, simulating: threading.Thread,
                   failure: list[BaseException]) -> None:
    """Until the simulator has made both of its changes, nothing during its
    lead and then the take (replay's schedule), as its events say."""
    deadline = time.monotonic() + TAKE_TIMEOUT_S
    while not (events.exists() and len(events.read_text().splitlines()) >= 2):
        if failure:
            raise failure[0]
        if not simulating.is_alive() or time.monotonic() > deadline:
            raise RuntimeError("the simulator did not take its memory")
        time.sleep(replay.POLL_S)
