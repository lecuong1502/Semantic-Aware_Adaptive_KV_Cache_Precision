"""The survival matrix (#190): S1 to S4 in the survival experiment, under B1 to B6.

Each scenario's schedule is a pure function of a run's seed and RQ1's
measures, so the seams are the schedules themselves, the base each is
played on, and which runs of the matrix are logged; the run itself is the
survival experiment's (tests/test_survival.py), tested small here for each
scenario.
"""

import numpy as np
import pytest

from conftest import load_tool, require_model
from microinfer import ModelConfig, benchlog, monitor, survival, survival_matrix as matrix
from microinfer.engine import paged_cache_bytes
from microinfer.footprint import MIB
from microinfer.patterns import Measured, Trapezoid
from microinfer.schedule import Schedule

NAME = "qwen2.5-0.5b-instruct"
BASE = 1000 * MIB


def rq1(**more):
    """RQ1 as S2 and S3 read it: a spike that recovered, of 100 MiB, and a
    lasting drop of 300 MiB, a sheer edge, in 600 s recorded."""
    spikes = [Measured(Trapezoid(100 * MIB, 0.5, 1.0, 0.5), False),
              Measured(Trapezoid(300 * MIB, 0.0, 0.0, 0.0), True)]
    return matrix.Rq1(**{"spikes": spikes, "recorded_s": 600.0,
                         "p90_amplitude_bytes": 506.35 * MIB, "median_rise_s": 1.25,
                         "traces": matrix.patterns.TRACES, **more})


@pytest.mark.parametrize("scenario", ["S1", "S2", "S3"])
def test_a_scenario_s_schedule_is_the_same_for_its_seed_and_differs_by_seed(scenario):
    a = matrix.schedule(scenario, 0, rq1())
    assert matrix.schedule(scenario, 0, rq1()) == a
    assert matrix.schedule(scenario, 1, rq1()) != a
    assert a.points[0] == (0.0, 0) and a.points[-1][1] == 0  # the lead, and all given back


def test_s1_is_64_s_grid_on_the_base():
    grid = matrix.grid(seed=0)
    assert len(grid.cells) == 36 and len(grid.pulses) == 108
    assert grid.schedule.peak_bytes == 1408 * MIB
    first = grid.pulses[0].start_s
    assert first == matrix.LEAD_S and grid.schedule.at(first + 0.05) == 768 * MIB


def test_s2_pulses_at_rq1_s_p90_for_the_cooldown_twice_the_cooldown_apart():
    """A pulse of 506 MiB, in granules, rising over 1.25 s, kept 5 s and
    falling over 1.25 s; the next 10 s after it has fallen, and a seeded
    fraction of a poll more; for as long as asked."""
    train = matrix.pulse_train(0, rq1(), cooldown_s=5.0, length_s=100.0)
    first, second = train.pulses[0].start_s, train.pulses[1].start_s
    assert 0 <= second - first - (1.25 + 5.0 + 1.25 + 10.0) < monitor.POLL_S
    top = train.schedule.at(first + 1.25 + 2.5)
    assert abs(top - 506.35 * MIB) <= matrix.patterns.granule_bytes() / 2
    assert train.schedule.at(first + 1.25 + 5.0 + 1.25 + 5.0) == 0  # in the gap
    assert len(train.pulses) == 6 and train.schedule.duration_s >= 100.0


def test_s3_draws_rq1_s_spikes_whole_and_keeps_a_lasting_drop_to_the_end():
    """At RQ1's rate, 2 in 600 s, over an hour: about 12 arrivals, each the
    recovered spike or the lasting drop; the drops add up, kept to the end,
    and the recovered spikes come and go above them."""
    s = matrix.poisson_spikes(3, rq1(), length_s=3600.0)
    above = np.array([s.at(t) for t in np.arange(0, 3600, 0.25)])
    drops = above[-1] // (300 * MIB)
    assert above[-1] == drops * 300 * MIB and drops >= 1  # only drops, at the end
    spikes = above % (300 * MIB)
    assert spikes.max() == 100 * MIB  # a recovered spike, whole, above the drops
    assert np.all(np.diff(above - spikes) >= 0)  # what the drops hold never falls
    assert s.points[-1] == (pytest.approx(matrix.LEAD_S + 3600.0, abs=0.1), 0)


def test_s4_s_seed_chooses_the_recording_it_replays():
    names = [matrix.recording(seed, rq1()).name for seed in range(6)]
    assert names == [f"scenario-with-r{k}.csv.gz" for k in (1, 2, 3, 4, 5, 1)]


def test_s2_and_s3_are_made_from_rq1_s_log_as_it_reports_it():
    """#186's statistics: 48 spikes that recovered or lasted, over about
    132.5 minutes; the P90 amplitude 506 MiB and the median rise 1.09 s."""
    r = matrix.Rq1.read()
    assert len(r.spikes) == 48 and sum(s.lasting for s in r.spikes) == 45
    assert r.p90_amplitude_bytes / MIB == pytest.approx(506.35)
    assert r.median_rise_s * matrix.MEASURED_SHARE == pytest.approx(1.0907, abs=1e-3)
    assert r.rate_per_s * 60 == pytest.approx(48 / 132.5, rel=1e-2)


def test_a_schedule_on_its_base_is_the_base_higher_and_gives_it_back_at_the_end():
    s = matrix.raised(Schedule([(0.0, 0), (1.0, 4 * MIB), (2.0, 0)]), BASE)
    assert s.points == [(0.0, BASE), (1.0, BASE + 4 * MIB), (2.0, 0)]
    with pytest.raises(ValueError):
        matrix.raised(Schedule([(0.0, 4 * MIB)]), BASE)


def test_the_base_leaves_what_the_scenario_starts_from_after_the_simulator_s_context():
    assert matrix.base_bytes(3000 * MIB, leave_bytes=1536 * MIB,
                             simulator_context_bytes=82 * MIB) == 1382 * MIB
    with pytest.raises(ValueError, match="below"):
        matrix.base_bytes(1500 * MIB, leave_bytes=1536 * MIB, simulator_context_bytes=0)


def test_a_static_int4_engine_s_headroom_is_counted_as_an_fp16_engine_s():
    """At 16K positions Qwen2.5-0.5B's FP16 cache holds 192 MiB; an INT4
    cache, less: its engine has that much more free, which the base takes
    too, so that both face the same bytes taken."""
    cfg = ModelConfig.from_card(NAME)
    held = paged_cache_bytes(cfg, 16384, "INT4")
    assert survival.as_fp16(cfg, 2000 * MIB, positions=16384, kv_tier="INT4") == \
        2000 * MIB - (192 * MIB - held)
    assert survival.as_fp16(cfg, 2000 * MIB, positions=16384, kv_tier="FP16") == 2000 * MIB


def test_each_condition_is_named_from_its_engine():
    assert matrix.condition_of("FP16", False, "semantic") == "B1"
    assert matrix.condition_of("INT2", False, None) == "B3"
    assert matrix.condition_of("FP16", True, "random") == "B5"
    with pytest.raises(ValueError):
        matrix.condition_of("INT4", True, "semantic")


def test_the_matrix_runs_every_cell_s_first_run_before_any_s_second():
    cells = matrix.cells()
    assert len(cells) == 6 * 4 * 3 and len(set(cells)) == len(cells)
    assert [c.run for c in cells] == sorted(c.run for c in cells)
    assert cells[0].name == "S1-B1-r0"


def test_the_runs_logged_at_the_same_code_are_taken_up(tmp_path):
    log = tmp_path / "benchmark.jsonl"

    def logged(model, **config):
        benchlog.append("survival", model=model, context_length=None, precision_tiers=None,
                        config=config, results={"survived": True}, log=log)

    logged("m", condition="B6", scenario="S2", run=1)
    logged("other", condition="B6", scenario="S2", run=2)
    logged("m", engine="adaptive")  # the survival experiment's own, before the matrix
    commit = benchlog.read(log)[0]["git_commit"]
    assert matrix.logged(commit, "m", log) == {matrix.Cell("B6", "S2", 1)}
    assert matrix.logged("another commit", "m", log) == set()


@pytest.mark.slowest
@pytest.mark.parametrize("scenario", matrix.SCHEDULED)
def test_a_short_run_of_each_scenario_is_logged(scenario, tmp_path, monkeypatch):
    """On Qwen2.5-0.5B, adaptive and semantic: 1024 positions prefilled and
    600 decoded, the scenario's base taken at 512 and its schedule played
    past its lead until the generation ends, cutting it there; one
    "survival" entry, naming its condition, scenario and run, with its
    plans, its pressure events' waits and the monitor's detection against
    the trace."""
    if scenario == "S4" and not matrix.recording(0, matrix.Rq1.read()).exists():
        pytest.skip("RQ1's recordings are not here")
    require_model(NAME)
    tool = load_tool("survival_experiment")
    log = tmp_path / "benchmark.jsonl"
    log.write_text("".join(line + "\n" for line in
                           benchlog.DEFAULT_LOG.read_text().splitlines()
                           if '"contention-trace"' in line))
    real = benchlog.environment
    monkeypatch.setattr(tool.benchlog, "environment",
                        lambda log=benchlog.DEFAULT_LOG: {**real(log), "git_dirty": False})
    out = tmp_path / f"{scenario}.csv.gz"
    assert tool.main(["adaptive", "--scenario", scenario, "--run", "0", "--out", str(out),
                      "--issue", "190", "--model", NAME, "--prompt-positions", "1024",
                      "--new-tokens", "600", "--contention-at", "512", "--log", str(log)]) == 0
    (entry,) = [e for e in benchlog.read(log) if e["kind"] == "survival"]
    assert matrix.cell_of(entry) == matrix.Cell("B6", scenario, 0)
    r = entry["results"]
    print(f"\n{scenario}: {r['schedule']}, {r['totals']}, RED {r['detection']['red_events']}")
    assert r["survived"] is True and r["ending"]["tokens"] == 600
    assert {"plans", "pressure", "detection", "schedule"} <= set(r)
    assert r["schedule"]["base_mib"] > 0 and r["schedule"]["cut"]
    assert r["schedule"]["played_s"] > matrix.LEAD_S
    assert entry["config"]["scenario_rule"]["scenario"] == scenario


def test_the_matrix_tool_names_each_run_s_engine_and_takes_up_the_logged(tmp_path, capsys):
    """B2 is a static INT4 engine, B5 the adaptive one from random scores;
    a run logged at this code is not run again, and a recording a run
    stopped part way left is kept, the run again numbered beside it."""
    tool = load_tool("survival_matrix")
    log = tmp_path / "benchmark.jsonl"
    benchlog.append("survival", model="m", context_length=None, precision_tiers=None,
                    config={"condition": "B2", "scenario": "S1", "run": 0},
                    results={"survived": True}, log=log)
    out_dir = tmp_path / "matrix"
    out_dir.mkdir()
    (out_dir / "S1-B5-r0.csv.gz").write_bytes(b"")
    assert tool.main(["--issue", "191", "--model", "m", "--runs", "1", "--scenarios", "S1",
                      "--conditions", "B2", "B5", "--out-dir", str(out_dir), "--log", str(log),
                      "--dry-run"]) == 0
    printed = capsys.readouterr().out.splitlines()
    assert printed[0].startswith("1 of 2 runs logged")
    (b5,) = printed[1:]
    assert "adaptive --score-source random --scenario S1 --run 0" in b5
    assert str(out_dir / "S1-B5-r0.2.csv.gz") in b5
    args = tool.argparse.Namespace(issue=191, model="m", log=log)
    b2 = tool.command(matrix.Cell("B2", "S3", 2), out_dir / "x.csv.gz", args)
    assert b2[2:7] == ["static", "--kv-tier", "INT4", "--scenario", "S3"]
