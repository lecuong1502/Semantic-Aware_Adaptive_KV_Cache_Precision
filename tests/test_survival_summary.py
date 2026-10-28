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


def test_the_tool_refuses_a_dirty_tree(tmp_path, monkeypatch):
    tool = load_tool("summarise_survival")
    monkeypatch.setattr(tool.benchlog, "environment", lambda log: {"git_dirty": True})
    with pytest.raises(SystemExit):
        tool.main(["--issue", "191", "--log", str(tmp_path / "b.jsonl")])
