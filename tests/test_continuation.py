"""Continuation fidelity on WikiText-2 (#153).

A passage's prefix is prefilled and its warm tokens forced a position a
step, so that the scorer has scores; a controlled plan of each source is
applied; the measured tokens are forced, and their logits compared with
FP16's: top-1 agreement and KL divergence. The summary pools the passages,
with the paired difference in mean KL of semantic against uniform and
random and its interval (ADR-0017).

Seams: the passages, the measures and the summary, pure; a run through the
engine; and tools/continuation.py, which logs each run and the summary.
"""

import numpy as np
import pytest

from conftest import load_tool, require_model
from microinfer import Engine, benchlog, continuation, rq3

NAME = "qwen2.5-0.5b-instruct"


def test_passages_are_consecutive_and_do_not_overlap():
    tokens = np.arange(1000, dtype=np.int32)
    ps = continuation.passages(tokens, 3, continuation.Shape(200, 30, 70))
    assert [p.passage_id for p in ps] == ["p00", "p01", "p02"]
    assert [p.start for p in ps] == [0, 300, 600]
    assert ps[1].prefix[0] == 300 and ps[1].warm[0] == 500 and ps[1].measured[-1] == 599
    assert (len(ps[2].prefix), len(ps[2].warm), len(ps[2].measured)) == (200, 30, 70)
    with pytest.raises(ValueError, match="split holds 1000"):
        continuation.passages(tokens, 4, continuation.Shape(200, 30, 70))


def test_top_1_agreement_and_kl_from_fp16_at_each_position():
    reference = np.array([[2.0, 0.0, 0.0], [0.0, 1.0, 0.0]], np.float32)
    logits = np.array([[2.0, 0.0, 0.0], [1.0, 0.0, 0.0]], np.float32)
    m = continuation.measures(reference, logits)
    assert m["top1"] == [True, False]
    p = np.exp([0.0, 1.0, 0.0]) / np.exp([0.0, 1.0, 0.0]).sum()
    q = np.exp([1.0, 0.0, 0.0]) / np.exp([1.0, 0.0, 0.0]).sum()
    assert m["kl"][0] == 0.0 and m["kl"][1] == pytest.approx(float((p * np.log(p / q)).sum()))
    with pytest.raises(ValueError, match="reference"):
        continuation.measures(reference, logits[:1])


SHAPE = continuation.Shape()


def entry(log, passage, name, top1, kl, *, shape=SHAPE):
    prefix, warm, measured, plan_seed = shape
    benchlog.append(continuation.KIND, model="m", context_length=sum(shape[:3]),
                    precision_tiers={"FP16": 1.0},
                    config={"passage_id": passage, "condition": {"name": name},
                            "prefix_tokens": prefix, "warm_tokens": warm,
                            "measured_tokens": measured, "plan_seed": plan_seed},
                    results={"top1": top1, "kl_mean": kl, "effective_bits": 3.4,
                             "plan": {"target_met": passage != "p01"}}, log=log)


def test_a_family_with_a_begin_of_text_opens_every_passage_with_it():
    """Llama-3.2 was trained on contexts that open with <|begin_of_text|>
    (#209): each passage opens with it, then the split's next tokens, so
    that passage 3 is as well formed as passage 0. Its tokens, the split's,
    still follow each other with no overlap, and a passage keeps its length."""
    tokens = np.arange(1000, dtype=np.int32)
    shape = continuation.Shape(200, 30, 70)
    ps = continuation.passages(tokens, 3, shape, bos=7)
    for i, p in enumerate(ps):
        whole = np.concatenate([p.prefix, p.warm, p.measured])
        assert len(whole) == 300 and whole[0] == 7
        np.testing.assert_array_equal(whole[1:], tokens[i * 299:(i + 1) * 299])
        assert p.start == i * 299
    with pytest.raises(ValueError, match="split holds"):
        continuation.passages(tokens, 4, shape, bos=7)


def test_a_way_of_holding_the_passage_it_cannot_measure_is_refused():
    """Before the engine is touched: an ablation of the scorer holds the
    passage as neither a plan at FP16 nor a static tier does."""
    ablation = rq3.Condition("score-every-4", source="semantic", share=0.8, score_every=4)
    with pytest.raises(ValueError, match=ablation.name):
        list(continuation.runs(None, continuation.passages(
            np.arange(1000, dtype=np.int32), 1, continuation.Shape(200, 30, 70))[0],
            [ablation]))


def test_the_summary_pairs_semantic_against_uniform_and_random_by_passage(tmp_path):
    log = tmp_path / "benchmark.jsonl"
    for i in range(10):
        p = f"p{i:02d}"
        entry(log, p, "semantic-80", 0.9, 0.10 + i * 0.01)
        entry(log, p, "uniform-80", 0.8, 0.20 + i * 0.01)
        entry(log, p, "random-80", 0.8, 0.10 + i * 0.01 + (0.01 if i < 3 else -0.01))
    entry(log, "p00", "semantic-80", 0.9, 0.10)  # a repeat stands for the run before it
    entry(log, "p00", "uniform-80", 0.0, 9.0, shape=continuation.Shape(1024))  # another shape
    conds = ["semantic-80", "uniform-80", "random-80"]
    s = continuation.summarise(benchlog.read(log), "m", SHAPE, 10, conds)
    assert s["passages"] == 10 and len(s["entries"]) == 30
    assert s["by_condition"]["semantic-80"]["plans_short_of_target"] == 1
    assert len(s["commits"]) == 1
    assert s["by_condition"]["uniform-80"]["kl_mean"] == pytest.approx(0.245)
    low, high = s["by_condition"]["uniform-80"]["kl_bootstrap_95"]
    assert low < 0.245 < high
    uniform = s["paired"]["80%"]["semantic-uniform"]
    assert uniform["kl_difference"] == pytest.approx(-0.1)
    assert uniform["kl_difference_bootstrap_95"] == pytest.approx([-0.1, -0.1])
    assert uniform["semantic_lower_kl"] == 10 and uniform["top1_difference"] == pytest.approx(0.1)
    random = s["paired"]["80%"]["semantic-random"]
    assert random["semantic_lower_kl"] == 3
    lo, hi = random["kl_difference_bootstrap_95"]
    assert lo < random["kl_difference"] < hi
    assert "semantic-zipcache" not in s["paired"]["80%"]
    for i in range(10):
        entry(log, f"p{i:02d}", "zipcache-80", 0.85, 0.15 + i * 0.01)
    z = continuation.summarise(benchlog.read(log), "m", SHAPE, 10, [*conds, "zipcache-80"])
    assert z["paired"]["80%"]["semantic-zipcache"]["kl_difference"] == pytest.approx(-0.05)
    without_one = [e for e in benchlog.read(log)
               if not (e["config"]["passage_id"] == "p05"
                       and e["config"]["condition"]["name"] == "random-80")]
    with pytest.raises(ValueError, match="p05"):
        continuation.summarise(without_one, "m", SHAPE, 10, conds)
    # a run stopped part way, its passages after the tenth not run, is not pooled as whole
    with pytest.raises(ValueError, match="of the 50 passages"):
        continuation.summarise(benchlog.read(log), "m", SHAPE, 50, conds)


def test_the_runs_logged_at_the_same_code_are_taken_up(tmp_path):
    log = tmp_path / "benchmark.jsonl"
    assert continuation.logged("m", SHAPE, "any", log) == set()
    entry(log, "p00", "semantic-80", 0.9, 0.1)
    entry(log, "p01", "semantic-80", 0.9, 0.1, shape=continuation.Shape(1024))
    commit = benchlog.read(log)[0]["git_commit"]
    assert continuation.logged("m", SHAPE, commit, log) == {("p00", "semantic-80")}
    assert continuation.logged("m", SHAPE, "another commit", log) == set()


# -- through the engine --------------------------------------------------------------------

@pytest.fixture(scope="module")
def engine():
    e = Engine(require_model(NAME), prefill_chunk=128)
    e.load_weights()
    return e


@pytest.mark.slow
def test_a_passage_under_no_plan_is_its_reference_and_under_a_plan_departs_from_it(engine):
    """No plan applied, the logits are FP16's: agreement 1 and KL 0, to the
    bit. Under a plan at 80%, its bytes reclaimed, the logits depart."""
    tokens = np.random.default_rng(153).integers(1000, 100_000, 400).astype(np.int32)
    (passage,) = continuation.passages(tokens, 1, continuation.Shape(320, 16, 24))
    conds = [rq3.NO_PLAN, rq3.condition("uniform-80"), rq3.condition("random-80")]
    got = dict(continuation.runs(engine, passage, conds))
    none = got[rq3.NO_PLAN]
    assert none["top1"] == 1.0 and none["kl"] == [0.0] * 24 and none["plan"] is None
    uniform = got[rq3.condition("uniform-80")]
    assert uniform["plan"]["target_met"] and uniform["kl_mean"] > 0
    assert len(uniform["kl"]) == len(uniform["top1_by_position"]) == 24
    assert got[rq3.condition("random-80")]["random_seed"] == rq3.sample_seed("p00", 0)


@pytest.mark.slow
def test_a_static_tier_holds_the_whole_passage_and_is_measured_against_fp16(engine):
    """A static tier holds the passage at its tier from its first position
    (#206), so it runs alone, and its logits are compared with the same FP16
    reference as a plan's: INT8 departs from it, and INT4 further, at the
    bits each tier spends, with no plan."""
    tokens = np.random.default_rng(206).integers(1000, 100_000, 400).astype(np.int32)
    (passage,) = continuation.passages(tokens, 1, continuation.Shape(320, 16, 24))
    conds = [rq3.condition("uniform-80"), rq3.condition("static-int8"),
             rq3.condition("static-int4")]
    got = {c.name: r for c, r in continuation.runs(engine, passage, conds)}
    int8, int4 = got["static-int8"], got["static-int4"]
    for r in (int8, int4):
        assert r["plan"] is None and r["random_seed"] is None
        assert len(r["kl"]) == len(r["top1_by_position"]) == 24
    assert 0 < int8["kl_mean"] < int4["kl_mean"]
    assert int8["tier_shares"]["INT8"] > 0.9 and int4["tier_shares"]["INT4"] > 0.9
    assert int4["effective_bits"] < int8["effective_bits"] < 9
    assert got["uniform-80"]["plan"]["target_met"]


@pytest.mark.slow
def test_the_tool_logs_each_run_takes_a_stopped_run_up_and_summarises(tmp_path, monkeypatch):
    tool = load_tool("continuation")
    log = tmp_path / "benchmark.jsonl"
    real = benchlog.environment  # this test runs on a tree being changed
    monkeypatch.setattr(tool.benchlog, "environment",
                        lambda log=benchlog.DEFAULT_LOG: {**real(log), "git_dirty": False})
    small = ["--model", NAME, "--passages", "2", "--prefix-tokens", "256", "--warm-tokens",
             "8", "--measured-tokens", "8", "--conditions", "semantic-80", "uniform-80",
             "--log", str(log)]
    assert tool.main(["--issue", "153", *small]) == 0
    runs = benchlog.read(log)
    assert {(e["config"]["passage_id"], e["config"]["condition"]["name"]) for e in runs} == {
        (p, c) for p in ("p00", "p01") for c in ("semantic-80", "uniform-80")}
    assert all(e["kind"] == continuation.KIND and e["config"]["issue"] == 153 for e in runs)
    assert tool.main(["--issue", "153", *small]) == 0
    assert len(benchlog.read(log)) == 4  # nothing run again
    assert tool.main(["--issue", "153", "--summarise", *small]) == 0
    (summary,) = [e for e in benchlog.read(log) if e["kind"] == "continuation-summary"]
    assert summary["results"]["passages"] == 2
    assert set(summary["results"]["entries"]) == {e["sha256"] for e in runs}
    assert "semantic-uniform" in summary["results"]["paired"]["80%"]
