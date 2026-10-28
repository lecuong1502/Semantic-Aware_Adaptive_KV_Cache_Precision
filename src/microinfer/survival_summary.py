"""The survival matrix's summary: its logged runs pooled into one entry (#191).

A run's cell is its condition, B1 to B6, and its scenario, S1 to S6:

- S1 to S4 are the survival matrix's (#190), three runs a cell, each entry
  naming its condition, scenario and run. A cell pools the latest entry of
  each of its runs, so that a repeat stands for the run before it.
- S5 and S6 are the survival experiment's, one run a cell: S5 a "survival"
  entry of its step, the first logged on S5's generation, and S6 the first
  "pulse-release" entry (#155). The first, because the survival experiment's
  later S5 runs under B6 measured redesigns of the prefill's chunks (#135),
  not survival again. The runs before the matrix name no condition: it is
  their engine's, static FP16 B1 and adaptive by its score source.

For each cell, the survival rate, the runs that completed every token, with
its Wilson 95% interval: the survival table. For each scenario of the
matrix, the reaction table, B6's reaction by
stage, pooled over its runs: detection, the latency of each detected RED
episode as RQ2 scored it; the wait for a step, of each YELLOW and RED event,
from the monitor's transition to the step boundary at which the engine
drained it; planning, of each downgrade plan; application, of each that
applied a batch; and target met, the downgrade plans whose moves reached
their byte target, not short. An upgrade plan answers no pressure, and is
not a reaction.
"""

from __future__ import annotations

from collections.abc import Sequence

from . import contention, pressure_rule, rq3_summary, survival_matrix as matrix

SUMMARY_KIND = "survival-summary"
MODEL = pressure_rule.MODEL
#: Every scenario of the survival table: the matrix's, then the survival
#: experiment's step and its release.
TABLE_SCENARIOS = (*matrix.SCHEDULED, "S5", "S6")
#: The condition whose reaction the reaction table gives: the proposed policy.
REACTING = "B6"
#: S5's generation and step, the survival experiment's defaults (#108), and
#: S6's (#155): S5 and S6 runs of another are not the experiment's.
S5_GENERATION = {"prompt_positions": 32704, "new_tokens": 64, "contention_at": 16384,
                 "shortfall_mib": 256.0}
S6_GENERATION = {"prompt_positions": 30720, "new_tokens": 2048, "contention_at": 16384,
                 "shortfall_mib": 256.0}

Cell = tuple[str, str]


def cell_of(entry: dict, model: str = MODEL) -> Cell | None:
    """The (condition, scenario) a run of `model` is, None for an entry of no cell."""
    config = entry["config"]
    if entry["model"] != model or entry["kind"] not in ("survival", "pulse-release"):
        return None
    run = matrix.run_of(entry)
    if run is not None and run.scenario in matrix.SCHEDULED:
        return run.condition, run.scenario
    scenario = "S6" if entry["kind"] == "pulse-release" else "S5"
    generation = S6_GENERATION if scenario == "S6" else S5_GENERATION
    if any(config.get(k) != v for k, v in generation.items()):
        return None
    condition = config.get("condition") or matrix.condition_of(
        "FP16", config["engine"] == "adaptive", config["score_source"])
    return condition, scenario


def pool(entries: Sequence[dict], model: str = MODEL) -> dict[Cell, list[dict]]:
    """Each cell's runs, by run: a matrix cell's latest entry of each run,
    an experiment cell's first entry, its run 0."""
    kept: dict[Cell, dict[int, dict]] = {}
    for e in entries:
        cell = cell_of(e, model)
        if cell is None:
            continue
        runs = kept.setdefault(cell, {})
        if cell[1] in matrix.SCHEDULED:
            runs[e["config"]["run"]] = e
        else:
            runs.setdefault(0, e)
    return {cell: [runs[r] for r in sorted(runs)] for cell, runs in kept.items()}


def survival(cells: dict[Cell, list[dict]]) -> dict[str, dict[str, dict]]:
    """Each scenario's survival rate by condition, with its Wilson interval
    and the hashes of the runs it pools."""
    out: dict[str, dict[str, dict]] = {}
    for (condition, scenario), runs in sorted(cells.items(), key=_order):
        survived = sum(bool(e["results"]["survived"]) for e in runs)
        out.setdefault(scenario, {})[condition] = {
            "survived": survived, "runs": len(runs), "rate": survived / len(runs),
            "wilson_95": list(rq3_summary.wilson(survived, len(runs))),
            "entries": [e["sha256"] for e in runs]}
    return out


def reaction(cells: dict[Cell, list[dict]]) -> dict[str, dict]:
    """B6's reaction by stage in each scenario of the matrix,
    pooled over its runs: the median, P90 and max of each stage's times,
    and the downgrade plans whose target was met."""
    out = {}
    for scenario in matrix.SCHEDULED:
        runs = cells.get((REACTING, scenario))
        if not runs:
            continue
        results = [e["results"] for e in runs]
        plans = [p for r in results for p in r["plans"] if not p["upgrades"]]
        out[scenario] = {
            "detection_ms": contention.percentiles(
                [ms for r in results for ms in r["detection"]["latencies_ms"]]),
            "wait_ms": contention.percentiles(
                [ev["wait_ms"] for r in results for ev in r["pressure"]["events"]
                 if ev["level"] in ("YELLOW", "RED")]),
            "planning_ms": contention.percentiles([p["planning_ms"] for p in plans]),
            "applying_ms": contention.percentiles([p["applying_ms"] for p in plans
                                                   if p["batches"]]),
            "target_met": sum(not p["short"] for p in plans), "plans": len(plans),
            "entries": [e["sha256"] for e in runs]}
    return out


def summarise(entries: Sequence[dict], model: str = MODEL) -> dict:
    """The summary of a log's entries: survival, reaction, and every run pooled."""
    cells = pool(entries, model)
    return {"survival": survival(cells), "reaction": reaction(cells),
            "entries": [e["sha256"] for _cell, runs in sorted(cells.items(), key=_order)
                        for e in runs]}


def _order(item: tuple[Cell, list[dict]]) -> tuple[int, str]:
    (condition, scenario), _runs = item
    return TABLE_SCENARIOS.index(scenario), condition
