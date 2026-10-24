"""One end-to-end run under simulated contention (#157).

The question-blind needle task with the plan made by the monitor and the
controller, not controlled: part way through turn 1's summary the
contention simulator takes what leaves the engine's headroom the 80%
level's bytes below YELLOW, so that to get back to GREEN it has to reclaim
what the controlled plan at 80% does. Semantic and uniform face the same
rule at the same token, and each run logs the bytes it reclaimed beside its
answer.

Seams: the rule, pure; the summary of a log's entries, pure; and the run,
through the engine and the simulator, which tools/end_to_end.py logs.
"""

import numpy as np
import pytest

from conftest import load_tool, require_model
from microinfer import Engine, _microinfer, benchlog, end_to_end, monitor, needles, replay, rq3
from microinfer.footprint import MIB

SMALL = "qwen2.5-0.5b-instruct"
SET = rq3.NeedleSet("m", 140, 7936)
GRANULE = _microinfer.granule_bytes()


# -- the rule ------------------------------------------------------------------------------

def test_the_take_leaves_the_headroom_the_target_below_yellow():
    """Headroom of 2000 MiB, a simulator context of 80 MiB and a target of
    180 MiB: the take leaves 1024 - 180 MiB, rounded up to a granule, so
    the target is never less than asked."""
    taken = end_to_end.take_bytes(2000 * MIB, 180 * MIB, monitor.DEFAULT,
                                  simulator_context_bytes=80 * MIB)
    assert taken % GRANULE == 0
    left = 2000 * MIB - 80 * MIB - taken
    assert (1024 - 180) * MIB - GRANULE < left <= (1024 - 180) * MIB
    assert monitor.DEFAULT.classify(left) == monitor.YELLOW


def test_an_engine_below_the_level_already_is_refused():
    with pytest.raises(ValueError, match="below"):
        end_to_end.take_bytes(900 * MIB, 180 * MIB, monitor.DEFAULT,
                              simulator_context_bytes=80 * MIB)


# -- the summary ---------------------------------------------------------------------------

def screening(log, kept):
    benchlog.append("needle-screening", model="m", context_length=None,
                    precision_tiers={"FP16": 1.0},
                    config={"needle_seed": 140, "draws": 5, "haystack_tokens": 7936},
                    results={"kept": kept, "rejected": [], "complete": True}, log=log)


def entry(log, sample_id, source, correct, *, reclaimed_mib=180.0, done=True, needle_seed=140):
    return benchlog.append(
        end_to_end.KIND, model="m", context_length=8000, precision_tiers=None,
        config={"sample_id": sample_id, "source": source, "needle_seed": needle_seed,
                "haystack_tokens": 7936},
        results={"correct": correct, "answer": "a",
                 "contention": {"target_bytes": int(180 * MIB), "taken_bytes": int(1000 * MIB)},
                 "at_question": {"reclaimed_bytes": int(reclaimed_mib * MIB),
                                 "target_met": reclaimed_mib >= 180, "plans_done": done,
                                 "effective_bits": 3.4},
                 "at_answer": {"reclaimed_bytes": int((reclaimed_mib + 10) * MIB),
                               "plans_done": True},
                 "totals": {"returned_mib": reclaimed_mib, "seen_mib": reclaimed_mib}},
        log=log)


def test_the_summary_gives_each_source_its_answers_and_its_bytes(tmp_path):
    """Four samples: uniform misses two, semantic one of them; semantic
    reclaims 180 MiB each, uniform 170 on one, short of the target. A later
    run stands for the one before."""
    log = tmp_path / "benchmark.jsonl"
    kept = ["d00-0", "d10-0", "d20-0", "d30-0"]
    screening(log, kept)
    for i, s in enumerate(kept):
        entry(log, s, "semantic", i != 0)
        entry(log, s, "uniform", i > 1, reclaimed_mib=170.0 if i == 3 else 180.0)
    entry(log, "d20-0", "semantic", False)
    entry(log, "d20-0", "semantic", True, done=False)  # the latest stands
    entry(log, "d00-0", "uniform", True, needle_seed=141)  # another set's
    summary = end_to_end.summarise(benchlog.read(log), SET)
    s, u = summary["sources"]["semantic"], summary["sources"]["uniform"]
    assert summary["samples"] == 4
    assert s["correct"] == 3 and u["correct"] == 2
    assert s["target_met"] == 4 and u["target_met"] == 3
    assert s["plans_done_before_question"] == 3 and s["plans_done_at_answer"] == 4
    assert s["reclaimed_mib_mean_at_answer"] == pytest.approx(190.0)
    assert u["reclaimed_mib_mean"] == pytest.approx(177.5)
    assert u["taken_mib_mean"] == pytest.approx(1000.0)
    pair = summary["paired"]["semantic-uniform"]
    assert (pair["semantic_only"], pair["other_only"]) == (1, 0)
    assert len(summary["entries"]) == 8


def test_the_summary_names_what_is_missing(tmp_path):
    log = tmp_path / "benchmark.jsonl"
    screening(log, ["d00-0", "d10-0"])
    entry(log, "d00-0", "semantic", True)
    with pytest.raises(ValueError, match="no run of 3"):
        end_to_end.summarise(benchlog.read(log), SET)


# -- through the engine and the simulator --------------------------------------------------

@pytest.fixture(scope="module")
def engine():
    e = Engine(require_model(SMALL), prefill_chunk=None)
    e.load_weights()
    return e


@pytest.fixture(scope="module")
def sample(engine):
    text = load_tool("perplexity").held_out_text()
    return needles.needle_set(text, engine.tokenizer, seed=157, samples_per_depth=1,
                              haystack_tokens=2048)[5]


@pytest.mark.slowest
def test_the_monitor_s_plan_reclaims_under_the_simulator_before_the_question(
        engine, sample, tmp_path):
    """The simulator takes after 16 of turn 1's tokens; the monitor sees
    YELLOW, and the engine's plan, from the source's scores, reclaims the
    target before turn 2's question. The engine is left as it was."""
    context = replay.simulator_context_bytes(tmp_path)
    results = end_to_end.run(engine, sample, "uniform", out=tmp_path / "run.csv.gz",
                             simulator_context_bytes=context)
    took, at_question = results["contention"], results["at_question"]
    assert took["position"] == len(sample.turn_1) + end_to_end.TAKE_AFTER - 1
    assert results["pressure_levels"][1] in ("YELLOW", "RED")
    assert results["plans"] and results["plans"][0]["level"] in ("YELLOW", "RED")
    assert at_question["reclaimed_bytes"] > 0
    assert at_question["effective_bits"] < 16
    assert results["at_answer"]["reclaimed_bytes"] >= at_question["reclaimed_bytes"]
    assert results["correct"] == needles.correct(results["answer"], sample.code)
    assert not engine.kv_adaptive and engine._monitor is None
    print(f"\ntarget {took['target_bytes'] / MIB:.1f} MiB, took {took['taken_bytes'] / MIB:.0f}, "
          f"reclaimed {at_question['reclaimed_bytes'] / MIB:.1f} at the question, "
          f"{at_question['effective_bits']:.2f} bits, plans done "
          f"{at_question['plans_done']}: {results['answer']!r}")
    assert np.isfinite(at_question["effective_bits"])


@pytest.mark.slowest
def test_the_tool_runs_takes_a_stopped_run_up_and_summarises(tmp_path, monkeypatch):
    tool = load_tool("end_to_end")
    log = tmp_path / "benchmark.jsonl"
    real = benchlog.environment  # this test runs on a tree being changed
    monkeypatch.setattr(tool.benchlog, "environment",
                        lambda log=benchlog.DEFAULT_LOG: {**real(log), "git_dirty": False})
    benchlog.append("needle-screening", model=SMALL, context_length=None,
                    precision_tiers={"FP16": 1.0},
                    config={"needle_seed": 157, "draws": 1, "haystack_tokens": 2048},
                    results={"kept": ["d50-0", "d90-0"], "rejected": [], "complete": True},
                    log=log)
    shape = ["--model", SMALL, "--needle-seed", "157", "--haystack-tokens", "2048",
             "--out", str(tmp_path / "out"), "--log", str(log)]
    assert tool.main(["--issue", "157", "--samples", "d50-0", *shape]) == 0
    runs = [e for e in benchlog.read(log) if e["kind"] == end_to_end.KIND]
    assert {e["config"]["source"] for e in runs} == set(end_to_end.SOURCES)
    assert all(e["config"]["sample_id"] == "d50-0" and e["config"]["issue"] == 157
               and len(e["config"]["files"]) == 6 for e in runs)
    (tmp_path / "out" / "d90-0-semantic.csv.gz").write_text("left by a run stopped part way")
    assert tool.main(["--issue", "157", *shape]) == 0
    count = len(benchlog.read(log))
    assert count == 1 + 4  # d50-0's runs were not run again
    assert tool.main(["--issue", "157", *shape]) == 0
    assert len(benchlog.read(log)) == count
    assert tool.main(["--issue", "157", "--summarise", *shape]) == 0
    (summary,) = [e for e in benchlog.read(log) if e["kind"] == end_to_end.SUMMARY_KIND]
    assert summary["results"]["samples"] == 2
    assert set(summary["results"]["sources"]) == set(end_to_end.SOURCES)
