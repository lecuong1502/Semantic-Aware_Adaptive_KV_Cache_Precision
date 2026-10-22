"""The anchor-fact dialogue (#154).

A fact stated in an early turn of ten, each a WikiText passage and a request
to summarise it, the replies decoded; a controlled plan after the last; and
a question asking for the fact back, scored as the needle is. The set is
built from a seed, screened at FP16, and its recall logged per source and
level by the turns between the fact and the question.

Seams: the set, pure on the tokenizer; the runs and the summary read from
the log, pure; a dialogue through the engine; and tools/anchors.py, which
screens, runs and summarises.
"""

import pytest

from conftest import load_tool, require_model
from microinfer import Engine, anchors, benchlog, controller, needles, rq3

NAME = "qwen2.5-1.5b-instruct"
SMALL = "qwen2.5-0.5b-instruct"
tool = load_tool("anchors")


@pytest.fixture(scope="module")
def tokenizer():
    return Engine(require_model(NAME)).tokenizer  # the tokenizer alone: no weights


@pytest.fixture(scope="module")
def text():
    return tool.held_out_text()


@pytest.fixture(scope="module")
def built(tokenizer, text):
    return anchors.dialogue_set(text, tokenizer)


def test_ten_gaps_of_five_dialogues(built):
    assert len(built) == 50
    gaps = [d.gap for d in built]
    assert sorted(set(gaps)) == list(range(10)) and all(gaps.count(g) == 5 for g in set(gaps))
    assert all(d.fact_turn == 9 - d.gap and len(d.turns) == 10 for d in built)
    assert len({d.sample_id for d in built}) == len({d.code for d in built}) == 50


def test_the_same_seed_gives_the_same_set_byte_for_byte(tokenizer, text, built):
    assert anchors.dumps(anchors.dialogue_set(text, tokenizer)) == anchors.dumps(built)
    other = anchors.dialogue_set(text, tokenizer, seed=155)
    assert anchors.dumps(other) != anchors.dumps(built)


def test_more_dialogues_a_gap_leave_the_first_ones_as_they_are(tokenizer, text, built):
    more = anchors.dialogue_set(text, tokenizer, samples_per_gap=7)
    first = [d for d in more if int(d.sample_id.split("-")[1]) < 5]
    assert anchors.dumps(first) == anchors.dumps(built)


def ids(tokenizer, s):
    return tokenizer.encode(s).ids


def test_the_fact_opens_its_turn_s_message_once_and_nowhere_else(tokenizer, built):
    for d in built:
        start, end = d.fact_offsets
        planted = ids(tokenizer, needles.piece(anchors.note(d.fact)))
        assert d.turns[d.fact_turn][start:end].tolist() == planted
        assert d.fact == needles.needle_text(d.name, d.code)
        held = [i for i, t in enumerate(d.turns) if d.code in tokenizer.decode(t.tolist())]
        assert held == [d.fact_turn]
        assert d.code not in tokenizer.decode(d.question_turn.tolist())


def test_each_turn_is_a_passage_of_its_length_in_the_chat_template(tokenizer, built):
    first, opening = ids(tokenizer, needles.TURN_1_OPENING), ids(tokenizer, anchors.TURN_OPENING)
    closing = ids(tokenizer, anchors.TURN_CLOSING)
    for d in built:
        planted = d.fact_offsets[1] - d.fact_offsets[0]
        for i, t in enumerate(d.turns):
            head = first if i == 0 else opening
            assert t[:len(head)].tolist() == head and t[-len(closing):].tolist() == closing
            passage = len(t) - len(head) - len(closing) - (planted if i == d.fact_turn else 0)
            assert passage == anchors.PASSAGE_TOKENS
        assert d.question_turn.tolist() == ids(
            tokenizer, needles.TURN_2_AFTER_EOS.format(question=d.question))


def test_a_dialogue_s_passages_are_consecutive_and_its_own(tokenizer, built):
    """Each turn goes on from the line after the last one's; the dialogues
    of a gap start at different articles."""
    d = built[0]
    for a, b in zip(d.turns[1:-1], d.turns[2:]):
        assert tokenizer.decode(a.tolist()) != tokenizer.decode(b.tolist())
    for gap in range(10):
        assert len({x.first_article for x in built if x.gap == gap}) == 5


def test_the_dialogue_is_about_8k_positions_and_the_fact_outside_the_recency_floor(built):
    for d in built:
        prompts = sum(len(t) for t in d.turns) + len(d.question_turn)
        assert 7300 < prompts < 7700  # with ten replies of up to 32 tokens, under 8K
        after = sum(len(t) for t in d.turns[d.fact_turn + 1:]) + len(
            d.turns[d.fact_turn]) - d.fact_offsets[1]
        assert after > controller.DEFAULT_RECENCY_FLOOR


# -- the log -------------------------------------------------------------------------------

def entry(log, sample_id, name, correct, *, gap=0, screening=False, seed=154, plan_seed=0,
          turns=10, tiers=None, echoed=()):
    config = {"sample_id": sample_id, "gap": gap, "condition": {"name": name},
              "anchor_seed": seed, "plan_seed": plan_seed, "turns": turns,
              "passage_tokens": anchors.PASSAGE_TOKENS}
    if screening:
        config["screening"] = True
    benchlog.append(anchors.KIND, model="m", context_length=7900,
                    precision_tiers={"FP16": 1.0}, config=config,
                    results={"correct": correct, "effective_bits": 3.4,
                             "fact_page_tiers": tiers or {"INT8": 28},
                             "code_echoed": list(echoed)}, log=log)


def screening(log, kept, *, complete=True, seed=154):
    benchlog.append(anchors.SCREENING_KIND, model="m", context_length=None,
                    precision_tiers={"FP16": 1.0},
                    config={"anchor_seed": seed, "turns": 10,
                            "passage_tokens": anchors.PASSAGE_TOKENS, "draws": 15},
                    results={"kept": kept, "rejected": [], "complete": complete}, log=log)


def test_the_runs_logged_at_the_same_code_are_taken_up(tmp_path):
    log = tmp_path / "benchmark.jsonl"
    assert anchors.logged("m", 154, "any", log) == {}
    entry(log, "g0-0", "fp16", True, screening=True)
    entry(log, "g0-0", "semantic-80", False)
    entry(log, "g0-1", "semantic-80", True, seed=155)
    entry(log, "g0-2", "semantic-80", True, turns=3)
    entry(log, "g0-3", "semantic-80", True, plan_seed=1)
    commit = benchlog.read(log)[0]["git_commit"]
    assert anchors.logged("m", 154, commit, log) == {("g0-0", "semantic-80"): False}
    assert anchors.logged("m", 154, commit, log, screening=True) == {("g0-0", "fp16"): True}
    assert anchors.logged("m", 154, "another commit", log) == {}


def test_the_screened_set_is_the_last_complete_screening_s(tmp_path):
    log = tmp_path / "benchmark.jsonl"
    with pytest.raises(ValueError, match="anchor-screening"):
        anchors.last_screening([], "m", 154)
    screening(log, ["g0-0"])
    screening(log, ["g0-1"])
    screening(log, ["g0-2"], complete=False)
    screening(log, ["g0-3"], seed=155)
    assert anchors.last_screening(benchlog.read(log), "m", 154)["results"]["kept"] == ["g0-1"]


def test_the_screening_screens_by_gap():
    class Fake:
        def __init__(self, gap, k):
            self.gap, self.sample_id = gap, f"g{gap}-{k}"

    drawn = [Fake(g, k) for g in range(10) for k in range(7)]
    kept, rejected = rq3.screen(drawn, lambda d: d.sample_id != "g3-1", groups=anchors.gaps(),
                                group=lambda d: d.gap)
    assert [d.sample_id for d in kept if d.gap == 3] == ["g3-0", "g3-2", "g3-3", "g3-4", "g3-5"]
    assert [d.sample_id for d in rejected] == ["g3-1"]
    with pytest.raises(rq3.ShortScreening, match=r"gaps short .*\(4 turns: 2\)"):
        rq3.screen(drawn, lambda d: d.gap != 4 or d.sample_id < "g4-2", groups=anchors.gaps(),
                   group=lambda d: d.gap, label=anchors.gap_name, named="gaps")


def test_the_summary_gives_recall_by_gap_and_pairs_semantic_by_dialogue(tmp_path):
    log = tmp_path / "benchmark.jsonl"
    kept = [f"g{g}-{k}" for g in range(10) for k in range(5)]
    screening(log, kept)
    for s in kept:
        gap = int(s[1])
        entry(log, s, "fp16", True, gap=gap, screening=True, tiers={"FP16": 28})
        entry(log, s, "semantic-80", gap < 8, gap=gap, echoed=[1] if s == "g0-0" else ())
        entry(log, s, "uniform-80", gap < 2, gap=gap, tiers={"INT2": 28})
    entry(log, "g9-0", "uniform-80", True, gap=9, screening=True)  # screening's is not pooled
    conds = ["fp16", "semantic-80", "uniform-80"]
    s = anchors.summarise(benchlog.read(log), "m", 154, conditions=conds)
    assert s["dialogues"] == 50 and len(s["entries"]) == 150
    semantic = s["recall"]["semantic-80"]
    assert (semantic["correct"], semantic["n"]) == (40, 50) and semantic["code_echoed"] == 1
    assert semantic["by_gap"]["7 turns"]["correct"] == 5
    assert semantic["by_gap"]["8 turns"]["correct"] == 0
    assert s["recall"]["uniform-80"]["fact_page_tiers"] == {"INT2": 1.0}
    pair = s["paired"]["80%"]["semantic-uniform"]
    assert (pair["semantic_only"], pair["other_only"]) == (30, 0) and pair["mcnemar_p"] < 1e-6
    assert "semantic-random" not in s["paired"]["80%"] and "70%" not in s["paired"]
    with pytest.raises(ValueError, match="random-80"):
        anchors.summarise(benchlog.read(log), "m", 154, conditions=[*conds, "random-80"])


# -- through the engine --------------------------------------------------------------------

@pytest.fixture(scope="module")
def engine():
    e = Engine(require_model(SMALL), prefill_chunk=256)
    e.load_weights()
    return e


@pytest.fixture(scope="module")
def small(engine, text):
    return anchors.dialogue_set(text, engine.tokenizer, turns=4, passage_tokens=160)


@pytest.mark.slow
def test_a_run_finds_the_fact_where_it_says_and_records_what_its_pages_hold(engine, small):
    """The turns' prompts and replies, laid end to end, hold the fact's line
    at the positions the run gives; the plan comes after the last reply."""
    d = small[5]
    r = anchors.run(engine, d, rq3.condition("uniform-80"))
    end_of_turn = engine.tokenizer.token_to_id(rq3.END_OF_TURN)
    held, cut = [], False
    for body, reply, stopped in zip(d.turns, r["reply_tokens"], r["replies_stopped"]):
        held += ([end_of_turn] if cut else []) + body.tolist() + reply
        cut = not stopped
    start, end = r["fact_positions"]
    assert held[start:end] == engine.tokenizer.encode(needles.piece(anchors.note(d.fact))).ids
    assert r["plan_positions"] == len(held) - 1  # the last reply's last token not yet fed
    assert r["plan"]["source"] == "uniform" and r["plan"]["target_met"]
    pages = r["fact_pages"]
    assert pages == [start // d.page_tokens, (end - 1) // d.page_tokens]
    assert sum(r["fact_page_tiers"].values()) == engine.config.num_hidden_layers * (
        pages[1] - pages[0] + 1)
    assert r["correct"] == needles.correct(r["answer"], d.code)
    assert len(r["replies"]) == 4 and all(len(t) <= anchors.REPLY_TOKENS
                                          for t in r["reply_tokens"])


@pytest.mark.slowest
def test_conditions_that_share_the_turns_give_what_each_gives_alone(engine, small):
    d = small[2]
    conds = [rq3.condition("semantic-80"), rq3.condition("fp16"), rq3.condition("random-70")]
    got = list(anchors.runs(engine, d, conds))
    assert [c.name for c, _ in got] == ["semantic-80", "random-70", "fp16"]
    for cond, results in got:
        assert results == anchors.run(engine, d, cond), cond.name


@pytest.mark.slowest
def test_the_tool_screens_runs_takes_a_stopped_run_up_and_summarises(tmp_path, monkeypatch):
    log = tmp_path / "benchmark.jsonl"
    real = benchlog.environment  # this test runs on a tree being changed
    monkeypatch.setattr(tool.benchlog, "environment",
                        lambda log=benchlog.DEFAULT_LOG: {**real(log), "git_dirty": False})
    shape = ["--model", SMALL, "--turns", "2", "--passage-tokens", "96", "--log", str(log)]
    assert tool.main(["--issue", "154", "--screen", "--per-gap", "1", "--draws", "3",
                      *shape]) == 0
    (screened,) = [e for e in benchlog.read(log) if e["kind"] == anchors.SCREENING_KIND]
    kept = screened["results"]["kept"]
    assert len(kept) == 2 and screened["results"]["complete"]
    conds = ["--conditions", "semantic-80", "uniform-80"]
    assert tool.main(["--issue", "154", *conds, *shape]) == 0
    runs = [e for e in benchlog.read(log)
            if e["kind"] == anchors.KIND and not e["config"].get("screening")]
    assert {(e["config"]["sample_id"], e["config"]["condition"]["name"]) for e in runs} == {
        (s, c) for s in kept for c in ("semantic-80", "uniform-80")}
    assert all(e["config"]["issue"] == 154 for e in runs)
    count = len(benchlog.read(log))
    assert tool.main(["--issue", "154", *conds, *shape]) == 0
    assert len(benchlog.read(log)) == count  # nothing run again
    assert tool.main(["--issue", "154", "--summarise", *conds, *shape]) == 0
    (summary,) = [e for e in benchlog.read(log) if e["kind"] == "anchor-summary"]
    assert summary["results"]["dialogues"] == 2
    assert set(summary["results"]["recall"]) == {"fp16", "semantic-80", "uniform-80"}
