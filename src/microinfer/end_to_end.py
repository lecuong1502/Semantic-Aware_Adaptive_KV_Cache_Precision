"""One end-to-end run under simulated contention: RQ3's confirmation (#157).

ADR-0017 measures RQ3 with controlled plans, made between turns for a byte
target. This confirms the result survives real pressure: the screened
needle set's question-blind task, with the plan made by the monitor and
the controller, as under contention on a desktop. A run is a sample's turn
1, the document and the request for a summary, during which the
contention simulator takes memory and keeps it, then turn 2, the question,
under the semantic or the uniform source at the 80% level.

**When and how much** is a rule, not a number, so that both sources face
the same contention whatever else the desktop holds that day. After
TAKE_AFTER of turn 1's tokens the generation waits while the simulator
takes what leaves the engine's headroom the 80% level's bytes below
YELLOW: the bytes a controlled plan at 80% reclaims (share_target_bytes),
from the cache as it is then, every page at FP16, the same for both
sources. The monitor sees YELLOW, and the generation goes on once it has,
so that the engine plans at the next step from the scores turn 1 has given
so far; it downgrades within its step budget, and is back at GREEN, its
plan cancelled, once it has given back what was taken from the headroom
beyond YELLOW. The rest of turn 1 is decoded meanwhile; a turn 1 that ends
before TAKE_AFTER tokens is refused, since its contention would come with
the question.

A run logs, beside its answer, the contention it met, what the cache held
when the question was asked, its effective bits, its tiers and the bytes
its pages gave back since the take, and whether every plan was done by
then, the same once the answer was decoded, and each plan the engine made,
with the memory it returned to the driver. A plan for an 8K cache takes
more steps at the budget than turn 1 has left after the take: on
Qwen2.5-1.5B about 60 (#157), so it goes on through the question and its
answer, from the scores it was made from. Back at GREEN at the take's
level, the engine's own growth can bring YELLOW again, and a plan from the
scores of then.

The summary gives each source's accuracy, with bootstrap and Wilson
intervals, the bytes reclaimed and taken, how often the target was met and
the plans done before the question and by the answer's end, and semantic paired against uniform,
sample by sample, by McNemar's exact test.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from . import benchlog, controller, needles, nvml, rq3, rq3_summary, survival
from ._microinfer import granule_bytes
from .engine import DECODING
from .footprint import MIB
from .monitor import GREEN, POLL_S, Monitor, Thresholds

KIND = "rq3-end-to-end"
SUMMARY_KIND = "end-to-end-summary"
SOURCES = ("semantic", "uniform")
SHARE = rq3_summary.CLOSING_SHARE
EXPERIMENT = "question-blind"
#: Turn 1's tokens decoded before the simulator takes: the scores a plan
#: can come from, and steps enough after for it to be applied (#157).
TAKE_AFTER = 16
#: The longest a generation waits for the monitor to see the take.
LEVEL_TIMEOUT_S = 10.0
#: The summary's bootstrap seed.
SEED = 157
CONTENTION_RULE = (f"after {TAKE_AFTER} of turn 1's tokens, the simulator takes what leaves "
                   "the engine's headroom below YELLOW by the bytes a controlled plan at "
                   f"{round(SHARE * 100)}% would reclaim, and keeps it to the end")


def take_bytes(headroom_bytes: int, target_bytes: int, thresholds: Thresholds, *,
               simulator_context_bytes: int) -> int:
    """What the simulator takes, beyond its own CUDA context, from an
    engine with `headroom_bytes` free, to leave it `target_bytes` below
    YELLOW: rounded up to a granule, the unit it takes memory in, so that
    the target is never less than asked. Raises ValueError for an engine
    there already."""
    room = headroom_bytes - simulator_context_bytes - (thresholds.yellow_below_bytes
                                                       - target_bytes)
    if room <= 0:
        raise ValueError(f"the engine is below the level already: {room / MIB:.0f} MiB "
                         f"with nothing taken")
    granule = granule_bytes()
    return -(-room // granule) * granule


def run(engine, sample, source: str, *, out: str | Path, simulator_context_bytes: int,
        take_after: int = TAKE_AFTER) -> dict:
    """One sample's question-blind turns under `source` at the 80% level,
    the simulator taking memory after `take_after` of turn 1's tokens and
    recording the device to `out`, as the module describes: what a log
    entry's results hold. The engine is set for the run and put back
    after, its monitor stopped."""
    cond = rq3.source_condition(source, SHARE)
    rq3.check_engine(engine)
    simulator = survival.Simulator(out)
    end_of_turn = engine.tokenizer.token_to_id(rq3.END_OF_TURN)
    measuring = engine.measure_plans
    took: dict = {}
    with rq3.set_for(engine, cond), engine.conversation() as chat:
        engine.measure_plans = True
        watching = engine.start_monitor()
        # The prefill gives turn 1's first token, which is not reported.
        decoded = [1]

        def report(state: str, position: int, token: int | None) -> None:
            if state != DECODING or took:
                return
            decoded[0] += 1
            if decoded[0] == take_after:
                took.update(_contend(engine, chat, watching, simulator, sample.page_tokens,
                                     simulator_context_bytes))

        try:
            summary = chat.turn(sample.turn_1, rq3.SUMMARY_TOKENS, report=report)
            if not took:
                raise RuntimeError(f"turn 1 ended after {len(summary)} tokens, before the "
                                   f"take after {take_after}: lower take_after")
            stopped = int(summary[-1]) == end_of_turn
            taken_tiers = took.pop("tiers")
            at_question = _held(engine, chat, sample, taken_tiers, took["target_bytes"])
            answer = chat.turn(sample.turn_2_after_eos if stopped else sample.turn_2,
                               rq3.ANSWER_TOKENS)
            at_answer = _held(engine, chat, sample, taken_tiers, took["target_bytes"])
        finally:
            engine.stop_monitor()
            engine.measure_plans = measuring
            simulator.close()
        text = engine.decode(answer)
        ran = survival.summarise(engine.plans, ending=engine.ending)
        return {"answer": text, "correct": needles.correct(text, sample.code),
                "answer_tokens": answer.tolist(), "summary": engine.decode(summary),
                "summary_length": len(summary), "summary_stopped": stopped,
                "contention": {"after_tokens": take_after, **took,
                               "started_ns": simulator.started_ns},
                "at_question": at_question, "at_answer": at_answer,
                "plans": ran["plans"], "totals": ran["totals"],
                "pressure_levels": [r.event.level.value for r in engine.pressure_events],
                "pressure": survival.waits(engine.pressure_events, engine.plans)}


def _contend(engine, chat, watching: Monitor, simulator: survival.Simulator, page_tokens: int,
             simulator_context_bytes: int) -> dict:
    """The take, while the generation waits: the target from the cache as
    it is, the simulator's memory taken, and the monitor's sight of it."""
    tiers = chat.tiers()
    target = controller.share_target_bytes(SHARE, tiers, engine.page_bytes,
                                           positions=chat.positions, page_tokens=page_tokens)
    headroom = nvml.memory().free
    nbytes = take_bytes(headroom, target, watching.thresholds,
                        simulator_context_bytes=simulator_context_bytes)
    simulator.take(nbytes)
    deadline = time.monotonic() + LEVEL_TIMEOUT_S
    while watching.level in (None, GREEN):
        if time.monotonic() > deadline:
            raise RuntimeError("the monitor did not see the take")
        time.sleep(POLL_S)
    return {"position": chat.positions, "headroom_bytes": headroom, "target_bytes": target,
            "taken_bytes": nbytes, "tiers": tiers}


def _held(engine, chat, sample, tiers_at_take: np.ndarray, target: int) -> dict:
    """What the cache holds now, at the question or after its answer: its effective bits
    and tiers, the bytes its pages gave back since the take, whether that
    met the target, and whether every plan was done."""
    planned = rq3.Planned(chat.positions, None, chat.effective_bits(), chat.tiers())
    taken_tiers = np.full_like(planned.tiers, -1)
    taken_tiers[:, :tiers_at_take.shape[1]] = tiers_at_take
    reclaimed = controller.bytes_given_back(taken_tiers, planned.tiers, engine.page_bytes)
    return {"positions": planned.positions, "effective_bits": planned.bits["effective_bits"],
            "candidate_effective_bits": planned.bits["candidate_effective_bits"],
            "tier_shares": rq3.tier_shares(planned.tiers[:, :planned.positions
                                                         // sample.page_tokens]),
            "needle_page_tiers": planned.page_tiers(sample.needle_pages),
            "reclaimed_bytes": reclaimed, "target_met": reclaimed >= target,
            "plans_done": all(p.ended for p in engine.plans)}


def config(sample, source: str, of: rq3.NeedleSet, **more) -> dict:
    """What a log entry's config names of a run."""
    return {**rq3.config(sample, rq3.source_condition(source, SHARE), EXPERIMENT,
                         **of.config()),
            "source": source, "take_after": TAKE_AFTER, "contention_rule": CONTENTION_RULE,
            **more}


def logged(of: rq3.NeedleSet, commit: str, log: str | Path = benchlog.DEFAULT_LOG
           ) -> dict[tuple[str, str], bool]:
    """The runs already logged at the code of `commit` (benchlog.same_code),
    of the set, by (sample id, source), each with whether it was right."""
    if not Path(log).exists():
        return {}
    same_code = benchlog.same_code_as(commit, log)
    return {(e["config"]["sample_id"], e["config"]["source"]): e["results"]["correct"]
            for e in benchlog.read(log)
            if e["kind"] == KIND and of.holds(e) and same_code(e["git_commit"])}


def summarise(entries: Sequence[dict], of: rq3.NeedleSet, *,
              sources: Sequence[str] = SOURCES) -> dict:
    """The summary of the screened set's runs in `entries`, the latest of
    each sample under each of `sources`, as the module describes. A
    ValueError names what is missing."""
    screening = of.screening(entries)
    kept = screening["results"]["kept"]
    latest = {(e["config"]["sample_id"], e["config"]["source"]): e
              for e in entries if e["kind"] == KIND and of.holds(e)}  # the latest stands
    missing = [(s, source) for source in sources for s in kept if (s, source) not in latest]
    if missing:
        raise ValueError(f"no run of {len(missing)} samples' sources, such as "
                         f"{missing[:3]}; run tools/end_to_end.py for them")
    rng = np.random.default_rng(SEED)
    runs = {source: [latest[(s, source)]["results"] for s in kept] for source in sources}
    right = {source: np.array([r["correct"] for r in runs[source]]) for source in sources}
    by_source = {}
    for source in sources:
        results = runs[source]
        at_question = [r["at_question"] for r in results]
        at_answer = [r["at_answer"] for r in results]
        by_source[source] = {
            **rq3_summary.proportion(right[source], rng),
            "accuracy": float(right[source].mean()),
            "reclaimed_mib_mean": float(np.mean([a["reclaimed_bytes"] for a in at_question]))
            / MIB,
            "returned_mib_mean": float(np.mean([r["totals"]["returned_mib"] for r in results])),
            "target_mib_mean": float(np.mean([r["contention"]["target_bytes"]
                                              for r in results])) / MIB,
            "taken_mib_mean": float(np.mean([r["contention"]["taken_bytes"]
                                             for r in results])) / MIB,
            "target_met": sum(a["target_met"] for a in at_question),
            "plans_done_before_question": sum(a["plans_done"] for a in at_question),
            "reclaimed_mib_mean_at_answer": float(np.mean([a["reclaimed_bytes"]
                                                           for a in at_answer])) / MIB,
            "plans_done_at_answer": sum(a["plans_done"] for a in at_answer),
            "effective_bits": float(np.mean([a["effective_bits"] for a in at_question]))}
    paired = {f"semantic-{other}": rq3_summary.paired(right["semantic"], right[other], rng)
              for other in sources if other != "semantic" and "semantic" in sources}
    return {"model": of.model, **of.config(), "experiment": EXPERIMENT, "share": SHARE,
            "samples": len(kept), "contention_rule": CONTENTION_RULE,
            "bootstrap": {"resamples": rq3_summary.RESAMPLES, "seed": SEED},
            "sources": by_source, "paired": paired, "screening": screening["sha256"],
            "commits": sorted({latest[(s, source)]["git_commit"]
                               for source in sources for s in kept}),
            "entries": [latest[(s, source)]["sha256"] for source in sources for s in kept]}
