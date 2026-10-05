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
"""

from __future__ import annotations

import threading
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import _microinfer, nvml, recorder, replay
from .config import ModelConfig
from .contention import spread
from .engine import COMPLETE, Ending, Engine, PlanRecord, PressureRecord, paged_cache_bytes
from .footprint import MIB
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


def run(engine: Engine, prompt: np.ndarray, new_tokens: int, *, contention_at: int,
        shortfall_bytes: int, out: str | Path, simulator_context_bytes: int) -> Run:
    """Generate `new_tokens` after `prompt` with the engine's monitor on and
    plans measured, while the recorder records the device to `out`. When
    the cache first reports `contention_at` positions or more, the
    simulator takes contention_bytes for `shortfall_bytes` and keeps it
    until the generation ends; the generation waits there until it has.
    A static engine that runs out raises OutOfMemory, which is caught and
    summarised; an adaptive one ends as its `ending` says."""
    out = Path(out)
    context = len(prompt) + new_tokens - 1
    done = threading.Event()
    failure: list[BaseException] = []
    took: dict = {}
    reached = [0]

    def contend() -> None:
        try:
            took["started_ns"] = replay.replay(Schedule([(0.0, took["bytes"])]), out,
                                               lead_s=LEAD_S, tail_s=KEEP_S, stop=done)
        except BaseException as exc:  # noqa: BLE001 - raised by the caller's thread
            failure.append(exc)

    simulating = threading.Thread(target=contend, name="survival-simulator")

    def report(state: str, position: int, token: int | None) -> None:
        reached[0] = position
        if took or position < contention_at:
            return
        took["position"], took["headroom"] = position, nvml.memory().free
        took["bytes"] = contention_bytes(engine.config, headroom_bytes=took["headroom"],
                                         positions=position, context=context,
                                         simulator_context_bytes=simulator_context_bytes,
                                         shortfall_bytes=shortfall_bytes)
        simulating.start()
        _wait_for_take(recorder.companion(out, ".events.jsonl"), simulating, failure)

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
        done.set()
        if simulating.is_alive():
            simulating.join()
    if failure:
        raise failure[0]
    if not took:
        raise RuntimeError(f"the cache never reached {contention_at} positions: no contention")
    return Run(summarise(engine.plans, ending=ending, error=error, positions_reached=reached[0]),
               took["position"], took["headroom"], took["bytes"], took["started_ns"],
               list(engine.pressure_events))


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
