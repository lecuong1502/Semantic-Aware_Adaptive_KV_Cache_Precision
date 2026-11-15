"""The survival matrix's summary: its logged runs pooled into one entry (#191).

A run's cell is its condition, B1 to B6 or B8, and its scenario, S1 to S6:

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

The held table (#207, ADR-0014's amendment) gives, for each cell, what its
runs' caches held over time and what that cost: the effective bits over
time and the share of the time at FP16, from each run's `results.held`;
and the session-level quality, continuation fidelity's KL and top-1 at
each step's bits, by the rule the amendment fixed (`session_quality`). A
static engine holds its tier for the whole session, so its runs logged
before their timeline are given their tier's by construction. An adaptive
run with no timeline has no figure.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import NamedTuple

from . import contention, pressure_rule, rq3_summary, survival_matrix as matrix
from .survival import BITS_DECIMALS

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


class Point(NamedTuple):
    """A quality curve's point: effective bits, and continuation's KL from
    FP16 and top-1 agreement there."""

    bits: float
    kl: float
    top1: float


#: FP16's point: the reference itself.
FP16_POINT = Point(16.0, 0.0, 1.0)
#: A session with no quality: no curve, or no time.
NO_QUALITY = {"kl": None, "top1": None, "below_measured_fraction": None}
#: The perplexity the static tiers are given beside their quality: #18's.
PERPLEXITY_WINDOW = 2048
#: What the held table pools by condition: the matrix's scenarios.
POOLED = "S1-S4"


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


def quality_curves(by_condition: Mapping[str, dict],
                   tier_bits: Mapping[str, float]) -> dict[str, list[Point] | None]:
    """Each condition's quality curve from a continuation summary's
    `by_condition` (ADR-0014, #207), its points by bits, lowest first: a
    static engine's its tier's alone, B1 FP16's; an adaptive engine's FP16's,
    its source's plans at 70% and 80% at the bits they held, and static
    INT2's at INT2's bits, every page at INT2, once it has a run. None for
    a static tier with no continuation run."""
    def point(name: str, bits: float | None = None) -> Point | None:
        c = by_condition.get(name)
        return None if c is None else Point(
            c["effective_bits"] if bits is None else bits, c["kl_mean"], c["top1"])

    int2_point = point("static-int2", tier_bits["INT2"])
    curves: dict[str, list[Point] | None] = {}
    for name, condition in matrix.CONDITIONS.items():
        if not condition.adaptive:
            tier = FP16_POINT if condition.kv_tier == "FP16" else point(
                f"static-{condition.kv_tier.lower()}")
            curves[name] = None if tier is None else [tier]
            continue
        plans = [point(f"{condition.score_source}-{level}") for level in (70, 80)]
        curves[name] = sorted(p for p in [FP16_POINT, *plans, int2_point] if p is not None)
    return curves


def session_quality(steps: Sequence[Sequence[float]], curve: Sequence[Point]) -> dict:
    """A session's quality (ADR-0014, #207): each step's KL and top-1 at its
    bits, linear in bits between the curve's neighbouring points, held for
    the step's seconds and averaged over the time. A step below the lowest
    point takes that point's quality, a lower bound on its KL and an upper
    bound on its top-1, and the share of the time spent there is given."""
    low = curve[0]
    seconds = kl = top1 = below = 0.0
    for held_s, bits in steps:
        at_kl, at_top1 = _interpolated(bits, curve)
        seconds += held_s
        kl += held_s * at_kl
        top1 += held_s * at_top1
        below += held_s if bits < round(low.bits, BITS_DECIMALS) else 0.0
    if seconds == 0:
        return dict(NO_QUALITY)
    return {"kl": kl / seconds, "top1": top1 / seconds, "below_measured_fraction": below / seconds}


def _interpolated(bits: float, curve: Sequence[Point]) -> tuple[float, float]:
    """KL and top-1 at `bits`, linear between `curve`'s neighbouring
    points, its ends held beyond them."""
    if bits <= curve[0].bits:
        return curve[0].kl, curve[0].top1
    for (b0, kl0, t0), (b1, kl1, t1) in zip(curve, curve[1:]):
        if bits <= b1:
            w = (bits - b0) / (b1 - b0)
            return kl0 + w * (kl1 - kl0), t0 + w * (t1 - t0)
    return curve[-1].kl, curve[-1].top1


def held_table(cells: dict[Cell, list[dict]], tier_bits: Mapping[str, float],
               curves: Mapping[str, list[Point] | None]) -> dict[str, dict[str, dict]]:
    """Each cell's held figures, by scenario and condition, and by condition
    over S1 to S4 (POOLED): the means of its measured runs' effective bits,
    FP16 fraction and session quality. A static run without its timeline
    holds its tier by construction; an adaptive one is not measured."""
    out: dict[str, dict[str, dict]] = {}
    pooled: dict[str, list[tuple[dict | None, dict]]] = {}
    for (condition, scenario), runs in sorted(cells.items(), key=_order):
        pairs = [(_held_figures(e, condition, tier_bits, curves.get(condition)), e)
                 for e in runs]
        out.setdefault(scenario, {})[condition] = _pooled(pairs)
        if scenario in matrix.SCHEDULED:
            pooled.setdefault(condition, []).extend(pairs)
    out[POOLED] = {condition: _pooled(pairs) for condition, pairs in pooled.items()}
    return out


def _held_figures(entry: dict, condition: str, tier_bits: Mapping[str, float],
                  curve: list[Point] | None) -> dict | None:
    """One run's effective bits, FP16 fraction and quality, or None."""
    spec = matrix.CONDITIONS[condition]
    held = entry["results"].get("held")
    if not spec.adaptive:
        bits = tier_bits[spec.kv_tier] if held is None else held["effective_bits"]
        fp16 = float(spec.kv_tier == "FP16") if held is None else held["fp16_fraction"]
        quality = NO_QUALITY if curve is None else {
            "kl": curve[0].kl, "top1": curve[0].top1, "below_measured_fraction": 0.0}
        return {"effective_bits": bits, "fp16_fraction": fp16, "by_construction": held is None,
                **quality}
    if held is None or held["effective_bits"] is None:
        return None
    quality = NO_QUALITY if curve is None else session_quality(held["steps"], curve)
    return {"effective_bits": held["effective_bits"], "fp16_fraction": held["fp16_fraction"],
            "by_construction": False, **quality}


def _pooled(pairs: list[tuple[dict | None, dict]]) -> dict:
    """The means of runs' figures, each run's (figures or None, entry),
    None where none was measured."""
    figures = [f for f, _ in pairs if f is not None]

    def mean(key: str) -> float | None:
        values = [f[key] for f in figures if f[key] is not None]
        return sum(values) / len(values) if values else None

    bits = [f["effective_bits"] for f in figures]
    below = [f["below_measured_fraction"] for f in figures
             if f["below_measured_fraction"] is not None]
    return {"runs": len(pairs), "measured": len(figures),
            "by_construction": bool(figures) and all(f["by_construction"] for f in figures),
            "effective_bits": mean("effective_bits"),
            "effective_bits_range": [min(bits), max(bits)] if bits else None,
            "fp16_fraction": mean("fp16_fraction"), "kl": mean("kl"), "top1": mean("top1"),
            "below_measured_fraction": mean("below_measured_fraction"),
            "kl_lower_bound": any(b > 0 for b in below),
            "entries": [e["sha256"] for _, e in pairs]}


def static_perplexity(entries: Sequence[dict], model: str = MODEL) -> dict[str, float]:
    """Each static condition's WikiText-2 perplexity, its tier's latest
    entry of both halves at it in #18's windows of 2048: beside the
    session figure, not in it."""
    tiers = {spec.kv_tier: name for name, spec in matrix.CONDITIONS.items() if not spec.adaptive}
    out: dict[str, float] = {}
    for e in entries:
        tier = next(iter(e.get("precision_tiers") or {}), None)
        if (e["kind"] == "perplexity" and e["model"] == model and tier in tiers
                and e["config"].get("kv_halves") == "both"
                and e["config"].get("window") == PERPLEXITY_WINDOW):
            out[tiers[tier]] = e["results"]["perplexity"]
    return {name: out[name] for name in tiers.values() if name in out}


def summarise(entries: Sequence[dict], model: str = MODEL, *,
              tier_bits: Mapping[str, float] | None = None) -> dict:
    """The summary of a log's entries: survival, reaction, and every run
    pooled; given the model's `tier_bits`, the held table too, its quality
    from the latest continuation summary of the model (ADR-0014, #207)."""
    cells = pool(entries, model)
    out = {"survival": survival(cells), "reaction": reaction(cells)}
    if tier_bits is not None:
        source = next((e for e in reversed(entries)
                       if e["kind"] == "continuation-summary" and e["model"] == model), None)
        by_condition = {} if source is None else source["results"]["by_condition"]
        out["held"] = held_table(cells, tier_bits, quality_curves(by_condition, tier_bits))
        out["quality_source"] = None if source is None else source["sha256"]
        out["perplexity"] = static_perplexity(entries, model)
    out["entries"] = [e["sha256"] for _cell, runs in sorted(cells.items(), key=_order)
                      for e in runs]
    return out


def _order(item: tuple[Cell, list[dict]]) -> tuple[int, str]:
    (condition, scenario), _runs = item
    return TABLE_SCENARIOS.index(scenario), condition
