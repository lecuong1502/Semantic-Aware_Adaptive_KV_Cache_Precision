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
from microinfer.engine import Conversation

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


def test_the_answer_s_opening_is_the_tokens_decoded_before_a_question_aware_plan():
    tokenizer = Engine(require_model(NAME)).tokenizer
    assert len(tokenizer.encode(rq3.AWARE_OPENING).ids) == rq3.AWARE_TOKENS


@pytest.mark.parametrize("opening, after", [
    ("The access code is", True),
    ("The access code for the", True),
    ("The access code is W", False),
    ("The access code is WDM", False),
    ("WDM-2615.", False),
])
def test_the_code_comes_after_an_opening_that_holds_none_of_it(opening, after):
    assert rq3.code_after(opening, "WDM-2615") is after


def test_each_sample_draws_its_own_random_scores():
    """One seed for every sample would rank the needle's page alike in the
    five of a depth, and pair their outcomes; each draws its own, logged."""
    seeds = {rq3.sample_seed(f"d{d:02d}-{k}", 0) for d in range(0, 100, 10) for k in range(5)}
    assert len(seeds) == 50
    assert rq3.sample_seed("d50-0", 0) == rq3.sample_seed("d50-0", 0) != rq3.sample_seed(
        "d50-0", 1)


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
def test_fp16_and_no_plan_applied_give_the_same_answer(engine, samples):
    """The adaptive, scoring engine a controlled plan needs changes nothing
    until a plan is applied: its answer is FP16's, token for token, here
    question-aware, whose answer is decoded in two parts."""
    experiment = "question-aware"
    s = samples[3]
    fp16 = rq3.run(engine, s, rq3.condition("fp16"), experiment)
    unplanned = rq3.run(engine, s, rq3.NO_PLAN, experiment)
    assert unplanned["answer_tokens"] == fp16["answer_tokens"]
    assert unplanned["effective_bits"] == fp16["effective_bits"] == 16.0
    assert fp16["plan"] is None and fp16["tier_shares"] == {"FP16": 1.0}


@pytest.mark.slow
def test_a_controlled_run_records_its_plan_and_what_the_needle_s_pages_hold(engine, samples):
    s = samples[5]
    r = rq3.run(engine, s, rq3.condition("uniform-80"), "question-blind")
    assert r["plan"]["source"] == "uniform" and r["plan"]["target_met"]
    assert r["random_seed"] is None  # a seed only for the random source
    assert sum(r["tier_shares"].values()) == pytest.approx(1.0) and "INT2" in r["tier_shares"]
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
    turn_2 = s.turn_2_after_eos if r["summary_stopped"] else s.turn_2
    assert r["plan_positions"] == (len(s.turn_1) + r["summary_length"] + len(turn_2)
                                   + rq3.AWARE_TOKENS - 1)
    assert len(r["answer_tokens"]) <= rq3.ANSWER_TOKENS
    assert r["aware_opening"] == engine.decode(r["answer_tokens"][:rq3.AWARE_TOKENS])
    assert r["opened_as_asked"] == r["aware_opening"].startswith(rq3.AWARE_OPENING)
    assert r["code_after_plan"] == rq3.code_after(r["aware_opening"], s.code)


@pytest.fixture(scope="module")
def chunked(engine):
    """The engine prefilling in chunks, as the RQ3 tool's does."""
    held = engine.prefill_chunk
    engine.prefill_chunk = 256
    yield engine
    engine.prefill_chunk = held


def test_only_the_conditions_at_fp16_and_adaptive_share_turn_one():
    shared = [c.name for c in (*rq3.CONDITIONS, rq3.NO_PLAN) if rq3.shares_turn_1(c)]
    assert shared == [*rq3.SOURCE_CONDITIONS, "no-plan"]


@pytest.mark.slowest
@pytest.mark.parametrize("experiment", rq3.EXPERIMENTS)
def test_conditions_that_share_turn_one_give_what_each_gives_alone(chunked, samples, experiment):
    """#172: the source conditions run turn 1 once, and each starts from its
    snapshot; a static tier, holding turn 1 as no other does, runs alone
    after them. Every result is the one the condition gives run alone, and
    the engine is left as it was found. Which conditions share is
    test_only_the_conditions_at_fp16_and_adaptive_share_turn_one's."""
    s = samples[11]
    before = (chunked.kv_tier, chunked.kv_adaptive, chunked.kv_scoring,
              chunked.kv_score_source)
    names = ["semantic-80", "static-int2", "random-70"]
    shared = list(rq3.runs(chunked, s, [rq3.condition(n) for n in names], experiment))
    assert [c.name for c, _ in shared] == ["semantic-80", "random-70", "static-int2"]
    for cond, results in shared:
        assert results == rq3.run(chunked, s, cond, experiment), cond.name
    random = dict(shared)[rq3.condition("random-70")]
    assert random["random_seed"] == random["plan"]["seed"] == rq3.sample_seed(s.sample_id, 0)
    assert (chunked.kv_tier, chunked.kv_adaptive, chunked.kv_scoring,
            chunked.kv_score_source) == before


@pytest.mark.slow
def test_a_snapshot_refused_runs_each_condition_alone(chunked, samples, monkeypatch):
    """A page gone below FP16 in turn 1, an emergency, refuses the
    snapshot: each condition then runs alone, as `run` runs it."""
    s = samples[12]
    alone = []

    def refused(chat):
        raise ValueError("a page of this cache is below it now")

    def run(engine, sample, cond, experiment, plan_seed=0, then=None):
        alone.append(cond.name)
        return {"run": cond.name}

    monkeypatch.setattr(Conversation, "snapshot", refused)
    monkeypatch.setattr(rq3, "run", run)
    conds = [rq3.condition("semantic-70"), rq3.condition("random-80")]
    got = list(rq3.runs(chunked, s, conds, "question-blind"))
    assert got == [(c, {"run": c.name}) for c in conds] and alone == ["semantic-70", "random-80"]


@pytest.mark.slowest
def test_the_conditions_chosen_run_from_one_command(tmp_path, monkeypatch):
    """One sample, both experiments, a source condition and a static tier:
    an entry for each, naming the sample, the condition, the experiment and
    the code. That every condition is one the tool can name is
    test_adr_0017_s_conditions'."""
    names = ["semantic-80", "static-int2"]
    log = tmp_path / "benchmark.jsonl"
    real = benchlog.environment  # this test runs on a tree being changed
    monkeypatch.setattr(tool.benchlog, "environment",
                        lambda log=benchlog.DEFAULT_LOG: {**real(log), "git_dirty": False})
    assert tool.main(["--issue", "147", "--model", NAME, "--samples", "d50-0",
                      "--experiments", *rq3.EXPERIMENTS, "--conditions", *names,
                      "--haystack-tokens", "1024", "--log", str(log)]) == 0
    entries = benchlog.read(log)
    assert len(entries) == 2 * len(names)
    seen = {(e["config"]["condition"]["name"], e["config"]["experiment"]) for e in entries}
    assert seen == {(n, x) for n in names for x in rq3.EXPERIMENTS}
    for e in entries:
        assert e["kind"] == "rq3-needle" and e["model"] == NAME
        assert e["config"]["sample_id"] == "d50-0" and e["git_commit"]
        assert isinstance(e["results"]["correct"], bool)
        assert e["precision_tiers"] == e["results"]["tier_shares"]


# -- a run of hours, taken up again --------------------------------------------------------

def logged_run(log, sample_id, name, correct, *, experiment="question-blind", plan_seed=0,
               screening=False, seed=140, haystack=1024, model="m"):
    config = {"sample_id": sample_id, "condition": {"name": name}, "experiment": experiment,
              "needle_seed": seed, "plan_seed": plan_seed, "haystack_tokens": haystack}
    if screening:
        config["screening"] = True
    benchlog.append("rq3-needle", model=model, context_length=1100,
                    precision_tiers={"FP16": 1.0}, config=config,
                    results={"correct": correct}, log=log)


def test_the_runs_logged_at_a_commit_are_named_by_sample_experiment_and_condition(tmp_path):
    """A run of hours stopped part way is taken up again at the same commit:
    what it logged, for this model, set and plan seed, is not run again."""
    log = tmp_path / "benchmark.jsonl"
    assert rq3.logged("m", 140, 1024, "any", log) == {}  # no log yet
    logged_run(log, "d00-0", "semantic-70", True)
    logged_run(log, "d00-0", "uniform-80", False)
    logged_run(log, "d00-1", "semantic-70", True, experiment="question-aware")
    logged_run(log, "d00-2", "fp16", True, screening=True)
    logged_run(log, "d00-3", "random-70", True, plan_seed=1)
    logged_run(log, "d00-4", "semantic-70", True, seed=141)
    logged_run(log, "d00-5", "semantic-70", True, haystack=7936)
    logged_run(log, "d00-6", "semantic-70", True, model="other")
    commit = benchlog.read(log)[0]["git_commit"]
    assert rq3.logged("m", 140, 1024, commit, log) == {
        ("d00-0", "question-blind", "semantic-70"): True,
        ("d00-0", "question-blind", "uniform-80"): False,
        ("d00-1", "question-aware", "semantic-70"): True}
    assert rq3.logged("m", 140, 1024, commit, log, screening=True) == {
        ("d00-2", "question-blind", "fp16"): True}
    assert rq3.logged("m", 140, 1024, commit, log, plan_seed=1) == {
        ("d00-3", "question-blind", "random-70"): True}
    assert rq3.logged("m", 140, 1024, "another commit", log) == {}


def test_a_run_logged_at_other_code_is_run_again_and_at_the_same_code_is_not(tmp_path,
                                                                            monkeypatch):
    """A commit that changed only the log holds the code of the runs before
    it (benchlog.same_code): they are taken up; after a change to the code,
    they run again. One git call a commit, not one an entry."""
    log = tmp_path / "benchmark.jsonl"
    logged_run(log, "d00-0", "semantic-70", True)
    logged_run(log, "d00-1", "semantic-70", True)
    commit = benchlog.read(log)[0]["git_commit"]
    asked = []
    monkeypatch.setattr(rq3.benchlog, "same_code",
                        lambda a, b, log: asked.append(a) or (a, b) == (commit, "log only"))
    assert set(rq3.logged("m", 140, 1024, "log only", log)) == {
        ("d00-0", "question-blind", "semantic-70"), ("d00-1", "question-blind", "semantic-70")}
    assert asked == [commit]
    assert rq3.logged("m", 140, 1024, "code changed", log) == {}


def test_question_blind_s_closing_run_is_every_source_at_both_levels():
    """#149: semantic, uniform and random at 70% and 80%; FP16 is
    screening's, and the static tiers are #151's."""
    assert tool.conditions_for(["sources"], screened=True) == [
        "semantic-70", "semantic-80", "uniform-70", "uniform-80", "random-70", "random-80"]
    assert tool.conditions_for(["sources", "semantic-70", "fp16", "fp16"], screened=True) == [
        *rq3.SOURCE_CONDITIONS, "fp16"]  # each once
