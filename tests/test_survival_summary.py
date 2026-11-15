"""The survival matrix's summary (#191): its logged runs pooled into one entry.

The survival table gives each condition's survival rate in each scenario, with
its Wilson interval: S1 to S4 from the matrix's three runs a cell, S5 and S6
from the survival experiment's one run a cell. The reaction table gives B6's
reaction in S1 to S4 by stage, pooled over its runs.

Seams: the pooling and the statistics, pure, over a log's entries; and
tools/summarise_survival.py, which logs the summary with the hashes of what
it pooled.
"""

import pytest

from conftest import load_tool
from microinfer import benchlog, survival_summary as summary

MODEL = "qwen2.5-1.5b-instruct"
S5_CONFIG = summary.S5_GENERATION


def plan(*, upgrades=False, short=False, planning_ms=6.0, applying_ms=60.0, batches=1):
    return {"upgrades": upgrades, "short": short, "planning_ms": planning_ms,
            "applying_ms": applying_ms, "batches": batches}


def results(survived=True, *, plans=(), latencies=(), waits=()):
    events = [{"level": "GREEN", "wait_ms": 500.0}] + [
        {"level": "RED", "wait_ms": w} for w in waits]
    return {"survived": survived, "plans": list(plans),
            "detection": {"latencies_ms": list(latencies)},
            "pressure": {"events": events}}


def matrix_run(condition, scenario, run, survived=True, **kw):
    return {"kind": "survival", "model": MODEL, "sha256": f"{scenario}{condition}{run}",
            "config": {"issue": 191, "condition": condition, "scenario": scenario,
                       "run": run, **({} if scenario in ("S1", "S2", "S3", "S4") else S5_CONFIG)},
            "results": results(survived, **kw)}


def experiment_run(engine, survived, sha, *, source="semantic", kind="survival", **config):
    """A run of the survival experiment from before the matrix: no condition named."""
    return {"kind": kind, "model": MODEL, "sha256": sha,
            "config": {"engine": engine, "score_source": None if engine == "static" else source,
                       **S5_CONFIG, **config},
            "results": {"survived": survived, "plans": []}}


def test_a_run_s_cell_is_its_condition_and_scenario():
    assert summary.cell_of(matrix_run("B4", "S3", 2)) == ("B4", "S3")
    # S5 and S6 runs from before the matrix name no condition: it is their engine's
    assert summary.cell_of(experiment_run("static", False, "a")) == ("B1", "S5")
    assert summary.cell_of(experiment_run("adaptive", True, "b")) == ("B6", "S5")
    assert summary.cell_of(experiment_run("adaptive", True, "c", source="uniform",
                                          kind="pulse-release", pulse_s=30.0,
                                          prompt_positions=30720, new_tokens=2048)) == ("B4", "S6")


@pytest.mark.parametrize("entry", [
    experiment_run("adaptive", True, "x", prompt_positions=8192),   # not S5's generation
    experiment_run("adaptive", True, "x", shortfall_mib=128.0),     # not S5's shortfall
    {**experiment_run("adaptive", True, "x"), "model": "qwen2.5-0.5b-instruct"},
    {"kind": "monitor-evaluation", "model": MODEL, "config": {}, "results": {}},
])
def test_runs_of_no_cell_are_not_pooled(entry):
    assert summary.cell_of(entry, MODEL) is None


def test_a_matrix_cell_pools_its_three_runs_and_an_experiment_cell_its_first():
    entries = [experiment_run("adaptive", True, "first"),
               experiment_run("adaptive", True, "redesign"),        # a later design, #135
               *(matrix_run("B1", "S1", r, survived=r == 2) for r in range(3)),
               matrix_run("B1", "S1", 0, survived=True) | {"sha256": "repeat"}]
    cells = summary.pool(entries, MODEL)
    assert [e["sha256"] for e in cells[("B6", "S5")]] == ["first"]
    # a repeat of a matrix run stands for the run before it
    assert sorted(e["sha256"] for e in cells[("B1", "S1")]) == ["S1B11", "S1B12", "repeat"]


def test_a_survival_rate_is_given_with_its_wilson_interval():
    rates = summary.survival({("B1", "S1"): [matrix_run("B1", "S1", r, survived=r == 0)
                                             for r in range(3)]})
    cell = rates["S1"]["B1"]
    assert (cell["survived"], cell["runs"]) == (1, 3)
    assert cell["rate"] == pytest.approx(1 / 3)
    low, high = cell["wilson_95"]
    assert 0 < low < 1 / 3 < high < 1
    assert cell["entries"] == ["S1B10", "S1B11", "S1B12"]


def test_the_reaction_pools_each_stage_over_a_scenario_s_runs():
    runs = [matrix_run("B6", "S1", 0, latencies=[100.0, 120.0], waits=[10.0],
                       plans=[plan(planning_ms=5.0, applying_ms=50.0), plan(upgrades=True)]),
            matrix_run("B6", "S1", 1, latencies=[140.0], waits=[30.0, 20.0],
                       plans=[plan(short=True, planning_ms=7.0, applying_ms=70.0),
                              plan(batches=0, planning_ms=6.0, applying_ms=0.0)])]
    row = summary.reaction({("B6", "S1"): runs})["S1"]
    assert row["detection_ms"]["median"] == 120.0 and row["detection_ms"]["n"] == 3
    assert row["wait_ms"]["median"] == 20.0          # the YELLOW and RED events only
    # downgrade plans only; a plan of no batch applied nothing, and took no time to
    assert row["planning_ms"]["n"] == 3 and row["applying_ms"]["n"] == 2
    assert (row["target_met"], row["plans"]) == (2, 3)


def test_a_scenario_with_no_red_episode_has_no_detection():
    row = summary.reaction({("B6", "S2"): [matrix_run("B6", "S2", 0, plans=[plan()])]})["S2"]
    assert row["detection_ms"] is None and row["wait_ms"] is None


# -- what a session's cache held over time, and what it cost (#207, ADR-0014) --------------

TIER_BITS = {"FP16": 16.0, "INT8": 8.625, "INT4": 4.625, "INT2": 2.625}


def continuation(**by_condition):
    """A continuation summary's conditions, each (effective bits, KL, top-1)."""
    return {name.replace("_", "-"): {"effective_bits": bits, "kl_mean": kl, "top1": top1}
            for name, (bits, kl, top1) in by_condition.items()}


CONTINUATION = continuation(semantic_70=(5.5, 0.002, 0.99), semantic_80=(4.0, 0.03, 0.94),
                            uniform_70=(5.5, 0.003, 0.97), uniform_80=(4.0, 0.1, 0.86),
                            random_70=(5.5, 0.005, 0.97), random_80=(4.0, 0.08, 0.88),
                            static_int8=(8.625, 2e-5, 0.997), static_int4=(4.625, 0.0055, 0.966))


def held_run(condition, scenario, run, steps):
    """A matrix run that logged its timeline: its steps [seconds, bits]."""
    seconds = sum(d for d, _ in steps)
    entry = matrix_run(condition, scenario, run)
    entry["results"]["held"] = {
        "seconds": seconds, "effective_bits": sum(d * b for d, b in steps) / seconds,
        "fp16_fraction": sum(d for d, b in steps if b == 16.0) / seconds,
        "lowest_bits": min(b for _, b in steps), "steps": [list(s) for s in steps]}
    return entry


def test_quality_is_linear_in_bits_between_the_points_measured_and_held_over_time():
    """16 bits for 10 s, 8 for 10 s, 6, halfway from 4 to 8, for 10 s, 4
    for 10 s: each step's quality from the curve, averaged over time."""
    curve = [summary.Point(4.0, 0.1, 0.8), summary.Point(8.0, 0.02, 0.96), summary.FP16_POINT]
    q = summary.session_quality([[10.0, 16.0], [10.0, 8.0], [10.0, 6.0], [10.0, 4.0]], curve)
    assert q["kl"] == pytest.approx((0 + 0.02 + 0.06 + 0.1) / 4)
    assert q["top1"] == pytest.approx((1 + 0.96 + 0.88 + 0.8) / 4)
    assert q["below_measured_fraction"] == 0.0


def test_below_the_lowest_point_a_step_takes_its_quality_and_the_time_is_reported():
    curve = [summary.Point(4.0, 0.1, 0.8), summary.FP16_POINT]
    q = summary.session_quality([[30.0, 16.0], [10.0, 2.7]], curve)
    assert q["kl"] == pytest.approx(0.1 / 4)  # a lower bound
    assert q["below_measured_fraction"] == pytest.approx(0.25)


def test_each_condition_s_curve_is_its_tier_or_its_source_s_points():
    curves = summary.quality_curves(CONTINUATION, TIER_BITS)
    assert curves["B1"] == [(16.0, 0.0, 1.0)]
    assert curves["B8"] == [(8.625, 2e-5, 0.997)] and curves["B2"] == [(4.625, 0.0055, 0.966)]
    assert curves["B3"] is None                       # static INT2 has no continuation run
    assert curves["B6"] == [(4.0, 0.03, 0.94), (5.5, 0.002, 0.99), (16.0, 0.0, 1.0)]
    assert curves["B4"][0] == (4.0, 0.1, 0.86)
    # once static INT2 has one, it is every source's lowest point, at INT2's bits
    curves = summary.quality_curves({**CONTINUATION, **continuation(
        static_int2=(2.625, 0.3, 0.6))}, TIER_BITS)
    assert curves["B3"] == [(2.625, 0.3, 0.6)]
    assert curves["B5"][0] == (2.625, 0.3, 0.6) and len(curves["B5"]) == 4


def test_a_static_run_logged_before_its_timeline_holds_its_tier_by_construction():
    cells = {("B2", "S1"): [matrix_run("B2", "S1", r) for r in range(3)],
             ("B1", "S1"): [matrix_run("B1", "S1", 0, survived=False)]}
    table = summary.held_table(cells, TIER_BITS, summary.quality_curves(CONTINUATION, TIER_BITS))
    b2 = table["S1"]["B2"]
    assert (b2["runs"], b2["measured"], b2["by_construction"]) == (3, 3, True)
    assert b2["effective_bits"] == 4.625 and b2["fp16_fraction"] == 0.0
    assert b2["kl"] == pytest.approx(0.0055) and b2["top1"] == pytest.approx(0.966)
    b1 = table["S1"]["B1"]                            # over the time it ran, whether or not
    assert b1["effective_bits"] == 16.0 and b1["fp16_fraction"] == 1.0 and b1["kl"] == 0.0


def test_an_adaptive_run_with_no_timeline_has_no_figure():
    cells = {("B6", "S2"): [matrix_run("B6", "S2", 0)]}
    cell = summary.held_table(cells, TIER_BITS,
                              summary.quality_curves(CONTINUATION, TIER_BITS))["S2"]["B6"]
    assert (cell["runs"], cell["measured"]) == (1, 0)
    assert cell["effective_bits"] is None and cell["kl"] is None


def test_an_adaptive_cell_pools_its_runs_figures_and_a_condition_its_cells_runs():
    cells = {("B6", "S1"): [held_run("B6", "S1", 0, [[30.0, 16.0], [10.0, 4.0]]),
                            held_run("B6", "S1", 1, [[40.0, 16.0]])],
             ("B6", "S2"): [held_run("B6", "S2", 0, [[20.0, 16.0], [20.0, 2.0]])],
             ("B6", "S3"): [matrix_run("B6", "S3", 0)],          # before its timeline
             ("B6", "S5"): [held_run("B6", "S5", 0, [[40.0, 4.0]])]}
    table = summary.held_table(cells, TIER_BITS, summary.quality_curves(CONTINUATION, TIER_BITS))
    s1 = table["S1"]["B6"]
    assert s1["measured"] == 2 and s1["by_construction"] is False
    assert s1["effective_bits"] == pytest.approx((13.0 + 16.0) / 2)
    assert s1["effective_bits_range"] == [13.0, 16.0]
    assert s1["fp16_fraction"] == pytest.approx((0.75 + 1.0) / 2)
    assert s1["kl"] == pytest.approx((0.03 / 4 + 0) / 2) and not s1["kl_lower_bound"]
    assert table["S2"]["B6"]["kl_lower_bound"] is True   # 2 bits is below 80%'s 4
    # by condition, the runs of S1 to S4, not S5's
    pooled = table["S1-S4"]["B6"]
    assert (pooled["runs"], pooled["measured"]) == (4, 3)
    assert pooled["effective_bits"] == pytest.approx((13.0 + 16.0 + 9.0) / 3)
    assert pooled["kl_lower_bound"] is True


def test_perplexity_is_given_for_the_static_tiers_only():
    def ppl(tier, value, halves="both", model=MODEL, window=2048):
        return {"kind": "perplexity", "model": model, "precision_tiers": {tier: 1.0},
                "config": {"kv_halves": halves, "window": window}, "results": {"perplexity": value}}

    entries = [ppl("FP16", 9.64), ppl("INT8", 9.641), ppl("INT4", 9.698), ppl("INT2", 11.54),
               ppl("INT4", 9.5, halves="keys"), ppl("INT4", 14.4, model="qwen2.5-0.5b-instruct"),
               ppl("INT4", 9.2, window=512)]                 # not #18's windows
    assert summary.static_perplexity(entries, MODEL) == {"B1": 9.64, "B8": 9.641, "B2": 9.698,
                                                         "B3": 11.54}


def test_the_tool_logs_the_summary_naming_every_run_it_pooled(tmp_path, monkeypatch):
    log = tmp_path / "benchmark.jsonl"
    for e in [experiment_run("static", False, "s"), matrix_run("B6", "S1", 0, plans=[plan()])]:
        benchlog.append(e["kind"], model=MODEL, context_length=None, precision_tiers=None,
                        config=e["config"], results=e["results"], log=log)
    tool = load_tool("summarise_survival")
    monkeypatch.setattr(tool.benchlog, "environment",
                        lambda log: {"git_dirty": False, "git_commit": "0" * 40})
    assert tool.main(["--issue", "191", "--log", str(log)]) == 0
    entry = benchlog.read(log)[-1]
    assert entry["kind"] == summary.SUMMARY_KIND
    pooled = [e["sha256"] for e in benchlog.read(log)[:2]]
    assert sorted(entry["results"]["entries"]) == sorted(pooled)
    assert entry["results"]["survival"]["S5"]["B1"]["survived"] == 0
    # the held table, the static engine's by construction at the model's tier bits (#207)
    held = entry["results"]["held"]
    assert held["S5"]["B1"]["effective_bits"] == 16.0 and held["S5"]["B1"]["by_construction"]
    assert held["S1"]["B6"]["measured"] == 0
    assert entry["results"]["quality_source"] is None   # no continuation summary logged


def test_the_tool_refuses_a_dirty_tree(tmp_path, monkeypatch):
    tool = load_tool("summarise_survival")
    monkeypatch.setattr(tool.benchlog, "environment", lambda log: {"git_dirty": True})
    with pytest.raises(SystemExit):
        tool.main(["--issue", "191", "--log", str(tmp_path / "b.jsonl")])
