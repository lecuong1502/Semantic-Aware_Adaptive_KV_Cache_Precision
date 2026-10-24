"""ADR-0018's quality ablations on the needle set (#158).

Nine variants of the full policy, the semantic source at 80% question-blind,
each changing one element: the recency floor W = 0 and 512, scoring every
step, the EWMA weight α = 0.05 and 0.5, importance alone, no INT2, layer
weights and a protected first page. Each is run on every screened sample,
and the summary gives its change in accuracy from the full policy, its
interval, McNemar's test and its average effective bits.

Seams: the ablations as conditions, pure; the engine's recency floor; the
runs, through the engine; the summary of a log's entries, pure; and the
tools, which log both.
"""

from dataclasses import asdict, fields, replace

import numpy as np
import pytest

from conftest import load_tool, require_model
from microinfer import Engine, _microinfer, benchlog, controller, needles, rq3, rq3_summary

SMALL = "qwen2.5-0.5b-instruct"
LAYERS = 24  # the 0.5B model's
WEIGHTS = tuple(1.0 + 0.1 * i for i in range(LAYERS))


# -- the ablations as conditions -----------------------------------------------------------

def test_adr_0018_s_nine_quality_ablations_each_change_one_element():
    """And no INT2 at 70% too, against the semantic source there: INT4
    gives back 71% of a page's bytes at most, short of 80%."""
    conds = rq3.ablations(WEIGHTS)
    assert [c.name for c in conds] == list(rq3.ABLATION_NAMES) == [
        "semantic-80-w0", "semantic-80-w512", "semantic-80-every-step",
        "semantic-80-alpha-0.05", "semantic-80-alpha-0.5", "semantic-80-importance-only",
        "semantic-80-no-int2", "semantic-70-no-int2", "semantic-80-layer-weights",
        "semantic-80-first-page"]
    changed = {c.name: [f.name for f in fields(c) if f.name not in ("name", "ablates")
                        and getattr(c, f.name) != getattr(rq3.measured_against(c), f.name)]
               for c in conds}
    assert all(len(names) == 1 for names in changed.values()), changed
    by_name = {c.name: c for c in conds}
    assert by_name["semantic-80-w512"].recency_floor == 512
    assert by_name["semantic-80-every-step"].score_every == 1
    assert by_name["semantic-80-no-int2"].options.lowest_tier == "INT4"
    assert by_name["semantic-80-layer-weights"].options.layer_weights == WEIGHTS
    assert rq3.FULL_POLICY == rq3.condition("semantic-80")
    assert rq3.ABLATED_FROM == ("semantic-80", "semantic-70")
    assert by_name["semantic-70-no-int2"].options == by_name["semantic-80-no-int2"].options
    assert rq3.FULL_POLICY.ablates is None and {c.ablates for c in conds} == {
        "recency floor", "scoring cadence", "EWMA weight", "marginal cost", "the INT2 tier",
        "layer weights", "attention sinks"}


def test_only_the_planner_s_ablations_share_turn_one_with_the_full_policy():
    """An ablation of the scorer's cadence or weight scores turn 1 as no
    other does, and runs it alone; the others change only the plan."""
    shared = [c.name for c in rq3.ablations(WEIGHTS) if rq3.shares_turn_1(c)]
    assert shared == ["semantic-80-w0", "semantic-80-w512", "semantic-80-importance-only",
                      "semantic-80-no-int2", "semantic-70-no-int2",
                      "semantic-80-layer-weights", "semantic-80-first-page"]


# -- the engine's recency floor ------------------------------------------------------------

def test_the_engine_s_recency_floor_is_adr_0016_s_by_default_and_checked():
    engine = Engine(require_model(SMALL))
    assert engine.kv_recency_floor == controller.DEFAULT_RECENCY_FLOOR == 128
    assert Engine(require_model(SMALL), kv_recency_floor=0).kv_recency_floor == 0
    for floor in (-1, 1.5, True):
        with pytest.raises(ValueError, match="kv_recency_floor"):
            engine.kv_recency_floor = floor


def ablated(engine):
    """What an ablation may change of the engine."""
    return (engine.kv_recency_floor, engine.kv_score_every, engine.kv_score_alpha,
            engine.kv_plan_options)


def test_a_condition_sets_what_it_ablates_and_puts_it_back():
    engine = Engine(require_model(SMALL))
    before = ablated(engine)
    for cond in rq3.ablations(WEIGHTS):
        with rq3.set_for(engine, cond):
            assert [h for h, b in zip(ablated(engine), before) if h != b] == [
                v for v in (cond.recency_floor, cond.score_every, cond.score_alpha,
                            cond.options) if v is not None]
            assert engine.kv_adaptive and engine.kv_scoring
        assert ablated(engine) == before and not engine.kv_adaptive


@pytest.fixture(scope="module")
def engine():
    e = Engine(require_model(SMALL), prefill_chunk=256)
    e.load_weights()
    return e


@pytest.mark.slow
def test_a_controlled_plan_keeps_the_engine_s_floor_and_chases_the_full_policy_s_bytes(engine):
    """At W = 0 and W = 512 the plan records its floor and names no page
    within it, W = 0 moves a page the default floor holds, and the target
    is the bytes it is at W = 128."""
    ids = np.random.default_rng(158).integers(1000, 100_000, 2048).astype(np.int32)
    page_tokens = _microinfer.device.page_tokens
    targets, last = {}, {}
    for floor in (0, 128, 512):
        cond = replace(rq3.FULL_POLICY, recency_floor=floor)
        with rq3.set_for(engine, cond), engine.conversation() as chat:
            chat.turn(ids, 8, stop_at_eos=False)
            positions = chat.positions
            record = chat.plan(0.8)
        assert record.plan.recency_floor == floor
        ends = (record.plan.pages + 1) * page_tokens
        assert (ends <= positions - floor).all()
        targets[floor], last[floor] = record.controlled["target_bytes"], int(ends.max())
    assert targets[0] == targets[128] == targets[512]
    assert last[0] > positions - controller.DEFAULT_RECENCY_FLOOR
    assert engine.kv_recency_floor == controller.DEFAULT_RECENCY_FLOOR


# -- the runs ------------------------------------------------------------------------------

@pytest.fixture(scope="module")
def sample(engine):
    text = load_tool("perplexity").held_out_text()
    return needles.needle_set(text, engine.tokenizer, seed=158, samples_per_depth=1,
                              haystack_tokens=1024)[5]


@pytest.mark.slowest
def test_an_ablation_sharing_turn_one_gives_what_it_gives_alone(engine, sample):
    """The full policy and W = 512 run turn 1 once; α = 0.5 runs it alone,
    after them. Each gives what it gives alone, and logs what its plan and
    its scorer were made with."""
    by_name = {c.name: c for c in rq3.ablations(WEIGHTS)}
    conds = [by_name["semantic-80-alpha-0.5"], rq3.FULL_POLICY, by_name["semantic-80-w512"]]
    got = list(rq3.runs(engine, sample, conds, "question-blind"))
    assert [c.name for c, _ in got] == ["semantic-80", "semantic-80-w512",
                                        "semantic-80-alpha-0.5"]
    for cond, results in got:
        assert results == rq3.run(engine, sample, cond, "question-blind"), cond.name
    results = {c.name: r for c, r in got}
    assert results["semantic-80-w512"]["plan"]["recency_floor"] == 512
    assert results["semantic-80"]["plan"]["recency_floor"] == 128
    assert results["semantic-80"]["plan"]["options"] == {
        "lowest_tier": "INT2", "importance_only": False, "protect_first_page": False,
        "layer_weights": None}
    assert results["semantic-80-alpha-0.5"]["score_alpha"] == pytest.approx(0.5)
    assert results["semantic-80"]["score_alpha"] == pytest.approx(Engine.DEFAULT_SCORE_ALPHA)
    assert results["semantic-80"]["score_every"] == Engine.DEFAULT_SCORE_EVERY


# -- the summary ---------------------------------------------------------------------------

SET = ("m", 140, 7936)
KEPT = ["d00-0", "d10-0", "d20-0", "d30-0"]


def screening(log, kept=KEPT):
    benchlog.append("needle-screening", model="m", context_length=None,
                    precision_tiers={"FP16": 1.0},
                    config={"needle_seed": 140, "draws": 5, "haystack_tokens": 7936},
                    results={"kept": kept, "rejected": [], "complete": True}, log=log)


CONDITIONS = {c.name: c for c in (*map(rq3.condition, rq3.ABLATED_FROM),
                                    *rq3.ablations(WEIGHTS))}


def run(log, sample_id, name, correct, *, bits=3.4, short=False, weights=WEIGHTS):
    cond = CONDITIONS[name]
    if cond.ablates == "layer weights":
        cond = replace(cond, options=controller.PlanOptions(layer_weights=weights))
    benchlog.append(
        "rq3-needle", model="m", context_length=8000, precision_tiers=None,
        config={"sample_id": sample_id, "condition": asdict(cond),
                "experiment": "question-blind", "needle_seed": 140, "plan_seed": 0,
                "haystack_tokens": 7936},
        results={"correct": correct, "effective_bits": bits, "candidate_effective_bits": bits,
                 "plan": {"short": short, "target_met": not short}},
        log=log)


def test_each_ablation_is_given_against_the_full_policy(tmp_path):
    """The full policy answers three of four; no INT2 answers one, short of
    its bytes on every plan, and two only the full policy answered; W = 0
    answers all four. No INT2 at 70% is measured against semantic-70,
    which answers only the one the others miss. A later run stands for the
    one before."""
    log = tmp_path / "benchmark.jsonl"
    screening(log)
    for i, s in enumerate(KEPT):
        run(log, s, "semantic-80", i != 0)
        run(log, s, "semantic-70", i == 0)
        for name in rq3.ABLATION_NAMES:
            if name == "semantic-80-no-int2":
                run(log, s, name, i == 3, bits=4.6, short=True)
            else:
                run(log, s, name, name == "semantic-80-w0" or i != 0)
    run(log, "d00-0", "semantic-80-w0", False)
    run(log, "d00-0", "semantic-80-w0", True)  # the latest stands
    summary = rq3_summary.summarise_ablations(benchlog.read(log), *SET)
    full = summary["against"]["semantic-80"]
    assert (full["correct"], full["n"]) == (3, 4) and list(summary["against"]) == [
        "semantic-80", "semantic-70"]
    assert list(summary["ablations"]) == list(rq3.ABLATION_NAMES)
    no_int2 = summary["ablations"]["semantic-80-no-int2"]
    assert no_int2["element"] == "the INT2 tier" and no_int2["correct"] == 1
    assert no_int2["change"] == pytest.approx(-0.5)
    assert (no_int2["ablation_only"], no_int2["against_only"]) == (0, 2)
    assert no_int2["against"] == "semantic-80"
    at_70 = summary["ablations"]["semantic-70-no-int2"]
    assert at_70["against"] == "semantic-70"
    assert (at_70["ablation_only"], at_70["against_only"]) == (3, 1)
    assert no_int2["mcnemar_p"] == pytest.approx(rq3_summary.mcnemar(0, 2))
    assert no_int2["effective_bits"] == pytest.approx(4.6)
    assert (no_int2["short"], no_int2["target_met"]) == (4, 0)
    low, high = no_int2["change_bootstrap_95"]
    assert low <= -0.5 <= high <= 0
    w0 = summary["ablations"]["semantic-80-w0"]
    assert (w0["change"], w0["ablation_only"], w0["against_only"]) == (0.25, 1, 0)
    assert summary["ablations"]["semantic-80-first-page"]["change"] == 0
    assert len(summary["entries"]) == 4 * (2 + len(rq3.ABLATION_NAMES))
    assert summary["layer_weights"] == pytest.approx(WEIGHTS)


def test_the_ablation_summary_refuses_layer_weights_runs_of_other_weights(tmp_path):
    log = tmp_path / "benchmark.jsonl"
    screening(log)
    for i, s in enumerate(KEPT):
        for name in CONDITIONS:
            run(log, s, name, True, weights=(2.0,) * LAYERS if i == 0 else WEIGHTS)
    with pytest.raises(ValueError, match="2 sets of weights"):
        rq3_summary.summarise_ablations(benchlog.read(log), *SET)


def test_the_ablation_summary_names_what_is_missing(tmp_path):
    log = tmp_path / "benchmark.jsonl"
    screening(log)
    for s in KEPT:
        run(log, s, "semantic-80", True)
    with pytest.raises(ValueError, match=r"no run of 44 .*--ablations"):
        rq3_summary.summarise_ablations(benchlog.read(log), *SET)


# -- the tools -----------------------------------------------------------------------------

@pytest.mark.slowest
def test_the_tools_run_every_ablation_and_summarise_them(tmp_path, monkeypatch):
    """One screened sample: the semantic source at both levels and the
    ablations, each
    logged, the layer weights' with those logged; a second command runs
    none again; the summary is logged. Sensitivity logged again, with other
    weights, refuses more runs, whose summary would pool both."""
    tool, summarise = load_tool("rq3"), load_tool("summarise_rq3")
    log = tmp_path / "benchmark.jsonl"
    real = benchlog.environment  # this test runs on a tree being changed
    for t in (tool, summarise):
        monkeypatch.setattr(t.benchlog, "environment",
                            lambda log=benchlog.DEFAULT_LOG: {**real(log), "git_dirty": False})
    shape = ["--issue", "158", "--model", SMALL, "--needle-seed", "158",
             "--haystack-tokens", "1024", "--log", str(log)]
    benchlog.append("needle-screening", model=SMALL, context_length=None,
                    precision_tiers={"FP16": 1.0},
                    config={"needle_seed": 158, "draws": 1, "haystack_tokens": 1024},
                    results={"kept": ["d50-0"], "rejected": [], "complete": True}, log=log)
    with pytest.raises(SystemExit):  # no layer sensitivity logged yet
        tool.main([*shape, "--ablations"])
    benchlog.append("layer-sensitivity", model=SMALL, context_length=None,
                    precision_tiers=None, config={},
                    results={"kl": [0.01 * (i + 1) for i in range(LAYERS)]}, log=log)
    with pytest.raises(SystemExit):  # the ablations choose their runs
        tool.main([*shape, "--ablations", "--samples", "d50-0"])
    assert tool.main([*shape, "--ablations"]) == 0
    runs = [e for e in benchlog.read(log) if e["kind"] == "rq3-needle"]
    assert [e["config"]["condition"]["name"] for e in runs] == [
        "semantic-80", "semantic-70", "semantic-80-w0", "semantic-80-w512",
        "semantic-80-importance-only", "semantic-80-no-int2", "semantic-70-no-int2",
        "semantic-80-layer-weights", "semantic-80-first-page",
        "semantic-80-every-step", "semantic-80-alpha-0.05", "semantic-80-alpha-0.5"]
    weighted = runs[7]["results"]["plan"]["options"]["layer_weights"]
    assert weighted == pytest.approx(controller.logged_layer_weights(SMALL, log))
    assert tool.main([*shape, "--ablations"]) == 0
    assert len([e for e in benchlog.read(log) if e["kind"] == "rq3-needle"]) == len(runs)
    assert summarise.main([*shape, "--ablations"]) == 0
    (summary,) = [e for e in benchlog.read(log)
                  if e["kind"] == rq3_summary.ABLATION_SUMMARY_KIND]
    assert summary["results"]["samples"] == 1
    assert list(summary["results"]["ablations"]) == list(rq3.ABLATION_NAMES)
    benchlog.append("layer-sensitivity", model=SMALL, context_length=None,
                    precision_tiers=None, config={},
                    results={"kl": [0.01] * (LAYERS - 1) + [0.5]}, log=log)
    with pytest.raises(SystemExit):  # runs of other weights would be pooled
        tool.main([*shape, "--ablations"])
    for t in (tool, summarise):
        with pytest.raises(SystemExit):
            t.main([*shape, "--ablations", "--long"])
