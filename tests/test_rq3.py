"""The RQ3 harness: one needle sample under one condition, logged (#147).

A run takes a sample of the needle set (#146) through its turns under a
condition of ADR-0017, FP16, a score source at a level, or a static tier,
in the question-blind or the question-aware experiment, and logs one entry:
the plan, the average effective bits, the answer and whether it is right.

Seams: the conditions, pure; a conversation's turn that only goes on with
its answer; and the harness through the engine, whose runs are logged by
tools/rq3.py (Seam A).
"""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from conftest import require_model
from microinfer import Engine, benchlog, needles, rq3

REPO = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("rq3_tool", REPO / "tools" / "rq3.py")
tool = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tool)

NAME = "qwen2.5-0.5b-instruct"


def test_adr_0017_s_conditions():
    """FP16; semantic, uniform and random at 70% and 80%; static INT4 and
    INT2 for the whole session."""
    names = [c.name for c in rq3.CONDITIONS]
    assert names == ["fp16", "semantic-70", "semantic-80", "uniform-70", "uniform-80",
                     "random-70", "random-80", "static-int4", "static-int2"]
    assert rq3.condition("uniform-80") == rq3.Condition("uniform-80", source="uniform",
                                                        share=0.8)
    assert rq3.condition("static-int2").tier == "INT2"
    with pytest.raises(ValueError, match="condition"):
        rq3.condition("semantic-50")


def test_the_question_asks_for_a_sentence_that_names_the_code_last():
    """Question-aware compresses after the answer's first tokens: they open
    the sentence, "The access code is", so the code itself comes after."""
    assert needles.question_text("Juniper").endswith(
        'Answer in one sentence, starting with "The access code is".')


# -- through the engine --------------------------------------------------------------------

@pytest.fixture(scope="module")
def engine():
    e = Engine(require_model(NAME), prefill_chunk=None)
    e.load_weights()
    return e


@pytest.fixture(scope="module")
def samples(engine):
    return needles.needle_set(tool.held_out_text(), engine.tokenizer, seed=147,
                              haystack_tokens=1024)


@pytest.mark.slow
def test_a_turn_with_no_prompt_goes_on_with_the_answer(engine, samples):
    """A turn of 4 tokens, then one with no prompt of 12, decode the tokens
    one turn of 16 would: the last token chosen is fed first, as ever."""
    s = samples[0]
    with engine.conversation() as chat:
        chat.turn(s.turn_1, 4, stop_at_eos=False)
        whole = chat.turn(s.turn_2, 16, stop_at_eos=False)
    with engine.conversation() as chat:
        chat.turn(s.turn_1, 4, stop_at_eos=False)
        first = chat.turn(s.turn_2, 4, stop_at_eos=False)
        rest = chat.turn(np.empty(0, np.int32), 12, stop_at_eos=False)
    assert np.concatenate([first, rest]).tolist() == whole.tolist()


@pytest.mark.slow
@pytest.mark.parametrize("experiment", rq3.EXPERIMENTS)
def test_fp16_and_no_plan_applied_give_the_same_answer(engine, samples, experiment):
    """The adaptive, scoring engine a controlled plan needs changes nothing
    until a plan is applied: its answer is FP16's, token for token."""
    s = samples[3]
    fp16 = rq3.run(engine, s, rq3.condition("fp16"), experiment)
    unplanned = rq3.run(engine, s, rq3.NO_PLAN, experiment)
    assert unplanned["answer_tokens"] == fp16["answer_tokens"]
    assert unplanned["effective_bits"] == fp16["effective_bits"] == 16.0
    assert fp16["plan"] is None


@pytest.mark.slow
def test_a_controlled_run_records_its_plan_and_what_the_needle_s_pages_hold(engine, samples):
    s = samples[5]
    r = rq3.run(engine, s, rq3.condition("uniform-80"), "question-blind")
    assert r["plan"]["source"] == "uniform" and r["plan"]["target_met"]
    assert r["candidate_effective_bits"] <= 16 * 0.2 + 1e-9  # a fifth of FP16's bytes
    assert r["candidate_effective_bits"] < r["effective_bits"] < 16  # the floor stays FP16
    assert set(r["needle_page_tiers"]) <= {"FP16", "INT8", "INT4", "INT2"}
    assert sum(r["needle_page_tiers"].values()) == engine.config.num_hidden_layers * (
        s.needle_pages[1] - s.needle_pages[0] + 1)
    assert r["correct"] == needles.correct(r["answer"], s.code)


@pytest.mark.slow
def test_question_aware_plans_after_the_answer_s_first_tokens(engine, samples):
    s = samples[7]
    r = rq3.run(engine, s, rq3.condition("semantic-80"), "question-aware")
    planned_at = len(s.turn_1) + r["summary_length"] + len(s.turn_2) + rq3.AWARE_TOKENS - 1
    assert r["plan_positions"] in (planned_at, planned_at - 1)  # turn 2 after EOS is 1 shorter
    assert len(r["answer_tokens"]) <= rq3.ANSWER_TOKENS


@pytest.mark.slow
def test_every_condition_runs_from_one_command(tmp_path, monkeypatch):
    """One sample, both experiments, every condition: an entry for each,
    naming the sample, the condition, the experiment and the code."""
    log = tmp_path / "benchmark.jsonl"
    real = benchlog.environment  # this test runs on a tree being changed
    monkeypatch.setattr(tool.benchlog, "environment",
                        lambda log=benchlog.DEFAULT_LOG: {**real(log), "git_dirty": False})
    assert tool.main(["--issue", "147", "--model", NAME, "--samples", "d50-0",
                      "--experiments", *rq3.EXPERIMENTS, "--haystack-tokens", "1024",
                      "--log", str(log)]) == 0
    entries = benchlog.read(log)
    assert len(entries) == 2 * len(rq3.CONDITIONS)
    seen = {(e["config"]["condition"]["name"], e["config"]["experiment"]) for e in entries}
    assert seen == {(c.name, x) for c in rq3.CONDITIONS for x in rq3.EXPERIMENTS}
    for e in entries:
        assert e["kind"] == "rq3-needle" and e["model"] == NAME
        assert e["config"]["sample_id"] == "d50-0" and e["git_commit"]
        assert isinstance(e["results"]["correct"], bool)
