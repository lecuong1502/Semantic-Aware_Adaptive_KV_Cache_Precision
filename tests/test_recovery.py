"""Recovery after pressure passes, on the needle set (#155).

After turn 2, the compressed turn, a restore brings every page back to FP16
from its shadow, and turn 3 asks the question again; a run logs turn 3
beside turn 2, and the summary gives each condition's accuracy at both.

Seams: the log, pure, its runs taken up and summarised; and the harness
through the engine, whose runs tools/recovery.py logs (Seam A).
"""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from conftest import require_model
from microinfer import Engine, benchlog, needles, recovery, rq3

REPO = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("recovery_tool", REPO / "tools" / "recovery.py")
tool = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tool)

SMALL = "qwen2.5-0.5b-instruct"
SET = rq3.NeedleSet("m", 140, 7936)


def screening(log, kept, *, draws=5):
    benchlog.append("needle-screening", model="m", context_length=None,
                    precision_tiers={"FP16": 1.0},
                    config={"needle_seed": 140, "draws": draws, "haystack_tokens": 7936},
                    results={"kept": kept, "rejected": [], "complete": True}, log=log)


def entry(log, sample_id, name, turn_2, turn_3, *, plan_seed=0, needle_seed=140, model="m"):
    cond = rq3.condition(name)
    restore = None if cond.source is None else {
        "moves": 10, "applied": 10, "batches": 1, "ended": "applied", "seconds": 0.5,
        "restored": True, "restored_bytes": 1000}
    return benchlog.append(
        recovery.KIND, model=model, context_length=8000, precision_tiers={"FP16": 1.0},
        config={"sample_id": sample_id, "condition": {"name": name}, "plan_seed": plan_seed,
                "needle_seed": needle_seed, "haystack_tokens": 7936},
        results={"correct": turn_2, "turn_3_correct": turn_3, "answer": "a",
                 "turn_3_answer": "a" if turn_2 == turn_3 else "b",
                 "turn_3_repeats_turn_2": turn_2 == turn_3, "restore": restore,
                 "effective_bits": 16.0 if cond.source is None else 3.4,
                 "turn_3_effective_bits": 16.0},
        log=log)


def test_the_runs_logged_at_the_same_code_are_taken_up(tmp_path):
    log = tmp_path / "benchmark.jsonl"
    entry(log, "d00-0", "semantic-80", True, True)
    entry(log, "d00-1", "uniform-80", False, True)
    entry(log, "d00-2", "uniform-80", True, True, plan_seed=1)
    entry(log, "d00-3", "uniform-80", True, True, needle_seed=141)
    entry(log, "d00-4", "uniform-80", True, True, model="other")
    commit = benchlog.read(log)[0]["git_commit"]
    assert recovery.logged(SET, commit, log) == {("d00-0", "semantic-80"): True,
                                                 ("d00-1", "uniform-80"): True}
    assert recovery.logged(SET, "0" * 40, log) == {}


def test_the_summary_gives_turn_2_and_turn_3_and_what_turn_3_recovered(tmp_path):
    """Four samples: uniform-80 wrong on two at turn 2 and right on one of
    them at turn 3, semantic-80 right throughout; FP16 right throughout. A
    later run of a sample stands for the one before."""
    log = tmp_path / "benchmark.jsonl"
    kept = ["d00-0", "d10-0", "d20-0", "d30-0"]
    screening(log, kept)
    for i, s in enumerate(kept):
        for name in recovery.CONDITIONS:
            uniform_80 = name == "uniform-80"
            entry(log, s, name, not (uniform_80 and i < 2), not (uniform_80 and i == 0))
    entry(log, "d30-0", "random-80", False, False)  # the latest stands
    summary = recovery.summarise(benchlog.read(log), SET)
    a = summary["accuracy"]
    assert summary["samples"] == 4 and list(a) == list(recovery.CONDITIONS)
    assert a["uniform-80"]["turn 2"]["correct"] == 2 and a["uniform-80"]["turn 3"]["correct"] == 3
    assert a["uniform-80"]["recovered"] == 1 and a["uniform-80"]["lost"] == 0
    assert a["random-80"]["turn 2"]["correct"] == 3
    assert a["fp16"]["restore"] is None and a["semantic-80"]["restore"]["every_page_restored"]
    assert a["fp16"]["turn_3_repeats_turn_2"] == 4
    pairs = summary["paired"]
    assert pairs["turn 2"]["80%"]["semantic-uniform"]["semantic_only"] == 2
    assert pairs["turn 3"]["80%"]["semantic-uniform"]["semantic_only"] == 1
    assert pairs["turn 3"]["80%"]["semantic-random"]["semantic_only"] == 1
    assert len(summary["entries"]) == 4 * len(recovery.CONDITIONS)


def test_the_summary_names_what_is_missing(tmp_path):
    log = tmp_path / "benchmark.jsonl"
    screening(log, ["d00-0"])
    entry(log, "d00-0", "fp16", True, True)
    with pytest.raises(ValueError, match="no run of 6"):
        recovery.summarise(benchlog.read(log), SET)


# -- through the engine --------------------------------------------------------------------

@pytest.fixture(scope="module")
def engine():
    e = Engine(require_model(SMALL), prefill_chunk=None)
    e.load_weights()
    return e


@pytest.fixture(scope="module")
def samples(engine):
    return needles.needle_set(tool.held_out_text(), engine.tokenizer, seed=155,
                              samples_per_depth=1, haystack_tokens=1024)


@pytest.mark.slow
def test_turn_3_asks_again_over_a_cache_restored_to_fp16(engine, samples):
    """A source condition's run is rq3's question-blind run to turn 2's
    answer, token for token, then a restore that leaves every page at
    FP16, and turn 3, the question again. FP16 restores nothing."""
    s = samples[5]
    cond = rq3.condition("uniform-80")
    plain = rq3.run(engine, s, cond, recovery.EXPERIMENT)
    results = recovery.run(engine, s, cond)
    assert results["answer_tokens"] == plain["answer_tokens"]
    assert results["candidate_effective_bits"] <= 16 * 0.2 + 1e-9
    restore = results["restore"]
    assert restore["restored"] and restore["ended"] == "applied" and restore["moves"] > 0
    assert results["turn_3_effective_bits"] == 16.0
    assert results["turn_3_correct"] == needles.correct(results["turn_3_answer"], s.code)
    assert results["turn_3_repeats_turn_2"] == (results["turn_3_answer"] == results["answer"])
    fp16 = recovery.run(engine, s, rq3.condition("fp16"))
    assert fp16["restore"] is None and fp16["turn_3_effective_bits"] == 16.0
    print(f"\nturn 2 {results['answer']!r}\nturn 3 {results['turn_3_answer']!r}\n"
          f"restore {restore['moves']} upgrades, {restore['seconds'] * 1e3:.0f} ms")


@pytest.mark.slowest
def test_conditions_that_share_turn_1_give_what_each_gives_alone(engine, samples):
    s = samples[0]
    conds = [rq3.condition(n) for n in ("semantic-80", "random-70")]
    together = dict(recovery.runs(engine, s, conds))
    for cond in conds:
        alone = recovery.run(engine, s, cond)
        for key in ("answer_tokens", "turn_3_answer_tokens", "turn_3_effective_bits"):
            assert together[cond][key] == alone[key], (cond.name, key)


@pytest.mark.slowest
def test_the_tool_runs_takes_a_stopped_run_up_and_summarises(tmp_path, monkeypatch):
    log = tmp_path / "benchmark.jsonl"
    real = benchlog.environment  # this test runs on a tree being changed
    monkeypatch.setattr(tool.benchlog, "environment",
                        lambda log=benchlog.DEFAULT_LOG: {**real(log), "git_dirty": False})
    benchlog.append("needle-screening", model=SMALL, context_length=None,
                    precision_tiers={"FP16": 1.0},
                    config={"needle_seed": 155, "draws": 1, "haystack_tokens": 512},
                    results={"kept": ["d00-0", "d90-0"], "rejected": [], "complete": True},
                    log=log)
    shape = ["--model", SMALL, "--needle-seed", "155", "--haystack-tokens", "512",
             "--log", str(log)]
    conds = ["--conditions", "fp16", "semantic-80", "uniform-80"]
    assert tool.main(["--issue", "155", *conds, *shape]) == 0
    runs = [e for e in benchlog.read(log) if e["kind"] == recovery.KIND]
    assert {(e["config"]["sample_id"], e["config"]["condition"]["name"]) for e in runs} == {
        (s, c) for s in ("d00-0", "d90-0") for c in ("fp16", "semantic-80", "uniform-80")}
    assert all(e["config"]["issue"] == 155 for e in runs)
    count = len(benchlog.read(log))
    assert tool.main(["--issue", "155", *conds, *shape]) == 0
    assert len(benchlog.read(log)) == count  # nothing run again
    assert tool.main(["--issue", "155", "--summarise", *conds, *shape]) == 0
    (summary,) = [e for e in benchlog.read(log) if e["kind"] == recovery.SUMMARY_KIND]
    assert summary["results"]["samples"] == 2
    assert set(summary["results"]["accuracy"]) == {"fp16", "semantic-80", "uniform-80"}
    assert np.isclose(summary["results"]["accuracy"]["fp16"]["turn_3_effective_bits"], 16.0)
