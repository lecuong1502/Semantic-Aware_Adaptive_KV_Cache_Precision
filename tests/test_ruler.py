"""RULER's tasks as RQ3 runs them (#208, ADR-0017's amendment).

Five of RULER's tasks, niah_multikey_1 to 3, niah_multivalue and vt, by
RULER's templates and arguments, generated from a fixed seed into RQ3's two
turns: turn 1 the template's opening, the context and the general request;
turn 2 the template's question, the answer prefix opening the reply. Each
sample is scored by RULER's string_match_all.

Seams: the pure pieces, the sentence rule, the essays' extraction from a
page and the scoring; and the set, built from a small essay and word list
with the checkpoint's tokenizer, no download.
"""

import pytest

from conftest import require_model
from microinfer import Engine, chat, ruler

NAME = "qwen2.5-1.5b-instruct"
ESSAY = " ".join(f"Sentence {i} of the essay says something about startups, and why."
                 for i in range(400))
WORDS = ruler.Words(adjectives=("amber", "brisk", "calm", "dusky", "eager"),
                    nouns=("anchor", "beacon", "candle", "dagger", "ember", "falcon"))
TOKENS = 1024


@pytest.fixture(scope="module")
def tokenizer():
    return Engine(require_model(NAME)).tokenizer  # the tokenizer alone: no weights


@pytest.fixture(scope="module")
def built(tokenizer):
    return {task: ruler.task_set(task, ESSAY, WORDS, tokenizer, seed=208, samples=4,
                                 template=chat.QWEN,
                                 haystack_tokens=TOKENS, page_tokens=32)
            for task in ruler.TASK_NAMES}


# -- the pure pieces ----------------------------------------------------------------------

def test_sentences_end_at_a_stop_before_a_capital():
    text = "It works. Does it? Yes! 3 more, e.g. this stays. \"Quoted\" too."
    assert ruler.sentences(text) == ["It works.", "Does it?", "Yes!",
                                     "3 more, e.g. this stays.", "\"Quoted\" too."]


def test_an_essay_is_the_text_of_its_page_s_first_font_tag():
    page = ("<html><body><font size=1>menu</font></body></html>")
    assert ruler.font_text(page) == "menu"
    page = ("<table><tr><td><font size=2 face=verdana>Why <i>startups</i> fail."
            "<br><br>Because &amp; so <font color=red>on</font>.</font>"
            "<font>after</font></td></tr></table>")
    assert ruler.font_text(page) == "Why startups fail.\n\nBecause & so on."
    # a script or a style inside it is no essay's text, as html2text drops it
    page = "<font>Text.<script>var x = {1};</script><style>p {}</style> More.</font>"
    assert ruler.font_text(page) == "Text. More."


def test_the_essays_are_joined_as_ruler_s_script_joins_them():
    """The GitHub copies first, then paulgraham.com's pages, each sorted by
    file name, joined with nothing between; a page is its first font's text."""
    fetched = {
        "http://www.paulgraham.com/zeta.html": "<font>Zeta.</font>",
        "http://www.paulgraham.com/alpha.html": "<font>Alpha.</font>",
        "https://github.com/x/y/raw/main/PaulGrahamEssays/worked.txt": "Worked.",
        "https://github.com/x/y/raw/main/PaulGrahamEssays/apple.txt": "Apple.",
    }
    assert ruler.essay_text(fetched) == "Apple.Worked.Alpha.Zeta."


def test_a_score_is_the_share_of_references_found_regardless_of_case():
    assert ruler.score("the numbers are 1234567 and 7654321", ("1234567", "7654321")) == 1.0
    assert ruler.score("ABCDE, fghij", ("abcde", "FGHIJ", "KLMNO")) == pytest.approx(2 / 3)
    assert ruler.score("none here", ("1234567",)) == 0.0


def test_the_singular_template_is_ruler_s_for_one_value_asked():
    """RULER's replacements when one value is asked: A, is, and the type's
    singular; niah_multivalue asks four and keeps the plural."""
    one = ruler.texts(ruler.TASKS["niah_multikey_1"], "brisk-anchor")
    assert one.opening.startswith("A special magic number is hidden within the following "
                                  "text. Make sure to memorize it. I will quiz you about the "
                                  "number afterwards.")
    assert one.question == ("What is the special magic number for brisk-anchor mentioned in "
                            "the provided text?")
    assert one.prefix == ("The special magic number for brisk-anchor mentioned in the "
                          "provided text is")
    four = ruler.texts(ruler.TASKS["niah_multivalue"], "calm-ember")
    assert four.question == ("What are all the special magic numbers for calm-ember "
                             "mentioned in the provided text?")
    uuid = ruler.texts(ruler.TASKS["niah_multikey_3"], "k")
    assert "special magic uuid for k" in uuid.question
    vt = ruler.texts(ruler.TASKS["vt"], "12345")
    assert vt.question == ("Question: Find all variables that are assigned the value 12345 "
                           "in the text above.")
    assert vt.prefix == ("Answer: According to the chain(s) of variable assignment in the "
                         "text above, 5 variables are assigned the value 12345, they are:")


# -- the set ------------------------------------------------------------------------------

def test_each_task_s_samples_hold_their_references_where_they_say(tokenizer, built):
    """Every target's tokens are at its recorded positions of turn 1, and
    hold a reference; its pages are those positions' pages."""
    for task, samples in built.items():
        assert len(samples) == 4 and len({s.sample_id for s in samples}) == 4
        for s in samples:
            text = tokenizer.decode(s.turn_1.tolist(), skip_special_tokens=False)
            assert all(r in text for r in s.expected), task
            for (start, end), (first, last) in zip(s.target_positions, s.target_pages):
                held = tokenizer.decode(s.turn_1[start:end].tolist())
                assert any(r in held for r in s.expected), (task, held)
                assert (first, last) == (start // 32, (end - 1) // 32)


def test_the_tasks_ask_as_ruler_s_arguments_say(tokenizer, built):
    one = built["niah_multikey_1"][0]
    assert len(one.needle_positions) == 4 and len(one.expected) == 1
    assert len(built["niah_multivalue"][0].expected) == 4
    assert len(built["niah_multivalue"][0].target_positions) == 4
    vt = built["vt"][0]
    assert len(vt.expected) == 5 and all(len(r) == 5 and r.isupper() for r in vt.expected)
    assert len(built["niah_multikey_3"][0].expected[0]) == 36       # a UUID
    for task in ("niah_multikey_2", "niah_multikey_3"):
        s = built[task][0]
        # the haystack is needles of other keys, none of them the one asked
        text = tokenizer.decode(s.turn_1.tolist())
        assert text.count(f"for {s.query} is:") == 1
        assert text.count("One of the special magic") > 5


def test_turn_2_asks_the_question_and_feeds_the_answer_prefix(tokenizer, built):
    for samples in built.values():
        s = samples[0]
        after = tokenizer.decode(s.turn_2_after_eos.tolist(), skip_special_tokens=False)
        assert after.endswith("<|im_start|>assistant\n" + s.prefix)
        assert s.query in after
        assert tokenizer.decode(s.turn_2.tolist(), skip_special_tokens=False).startswith(
            "<|im_end|>")
    assert built["vt"][0].answer_tokens == 30 and built["niah_multivalue"][0].answer_tokens == 128


def test_a_haystack_is_cut_to_its_tokens(tokenizer, built):
    for task, samples in built.items():
        for s in samples:
            assert s.haystack_tokens == TOKENS
            held = len(s.turn_1) - s.document_start - s.closing_tokens - sum(
                end - start for start, end in s.needle_positions)
            assert held == TOKENS - s.example_tokens, task
            assert (s.example_tokens > 0) == (task == "vt")


def test_the_same_seed_gives_the_same_set_and_more_draws_keep_the_first(tokenizer, built):
    for task in ruler.TASK_NAMES:
        again = ruler.task_set(task, ESSAY, WORDS, tokenizer, seed=208, samples=6,
                               template=chat.QWEN,
                               haystack_tokens=TOKENS, page_tokens=32)
        assert ruler.dumps(again[:4]) == ruler.dumps(built[task])
        other = ruler.task_set(task, ESSAY, WORDS, tokenizer, seed=209, samples=4,
                               template=chat.QWEN,
                               haystack_tokens=TOKENS, page_tokens=32)
        assert ruler.dumps(other) != ruler.dumps(built[task])


def test_a_sample_scores_its_answer_and_is_answered_only_with_every_reference(built):
    s = built["niah_multivalue"][0]
    assert s.scored(" ".join(s.expected)) == {"correct": True, "score": 1.0}
    half = s.scored(" ".join(s.expected[:2]))
    assert half == {"correct": False, "score": 0.5}


def test_essay_needles_sit_at_ruler_s_depths_in_order(tokenizer, built):
    """An essay's needles go at sentence boundaries, in order of their
    depths, drawn without replacement from RULER's 40."""
    for s in built["niah_multikey_1"] + built["niah_multivalue"]:
        starts = [start for start, _ in s.needle_positions]
        assert starts == sorted(starts)
        assert len(set(s.depths)) == len(s.depths)
        assert set(s.depths) <= set(int(d) for d in ruler.DEPTHS)


def test_variable_tracking_keeps_its_chain_in_order(tokenizer, built):
    for s in built["vt"]:
        text = tokenizer.decode(s.turn_1.tolist()).split(ruler.VT_OPENING)[-1]  # past the example
        lines = [line for line in text.split("\n") if line.startswith("VAR ")]
        first, *hops = s.expected
        assert lines[0] == f"VAR {first} = {s.query}"
        for (before, name), line in zip(zip(s.expected, hops), lines[1:]):
            assert line == f"VAR {name} = VAR {before} "


def test_variable_tracking_opens_with_ruler_s_one_shot_example(tokenizer, built):
    """RULER puts a worked example before VT's template: a chain of three-
    letter names among noise, its question and answer prefix, and its
    answer; its value and names drawn again for each sample, its layout
    once, and within RULER's 500 tokens."""
    examples = []
    for s in built["vt"]:
        text = tokenizer.decode(s.turn_1[:s.document_start].tolist())
        assert text.count("Memorize and track the chain(s)") == 2
        example = text.split("user\n", 1)[1].rsplit(ruler.VT_OPENING, 1)[0]
        assert example.endswith("\n\n")
        *_, answer = example.strip().split("they are: ")
        names = answer.split()
        assert len(names) == 5 and all(len(n) == 3 and n.isupper() for n in names)
        assert s.example_tokens <= ruler.VT_EXAMPLE_TOKENS + 1
        examples.append(example)
        lines = [line for line in example.split("\n") if line.startswith(("VAR", "The grass"))]
        examples[-1] = [line.startswith("VAR") for line in lines]
    assert all(layout == examples[0] for layout in examples)   # one layout for the task
    first, second = (tokenizer.decode(s.turn_1[:s.document_start].tolist())
                     for s in built["vt"][:2])
    assert first != second                                     # its names and value drawn again


def test_a_task_not_ruler_s_is_refused(tokenizer):
    with pytest.raises(ValueError):
        ruler.task_set("qa_1", ESSAY, WORDS, tokenizer, seed=0, template=chat.QWEN, samples=1)


# -- through RQ3's harness ----------------------------------------------------------------

@pytest.fixture(scope="module")
def small_engine():
    from microinfer import Engine as E
    e = E(require_model("qwen2.5-0.5b-instruct"), kv_cache="paged")
    e.load_weights()
    return e


@pytest.mark.slow
def test_a_ruler_sample_runs_through_rq3_s_turns_scored_as_ruler_scores_it(small_engine):
    """A sample of each kind of haystack, at FP16 and under the semantic
    source at 80%: the answer is decoded for the task's tokens after the
    fed prefix, scored by string_match_all, and the target pages' tiers are
    what the plan left them."""
    from microinfer import rq3
    for task in ("niah_multivalue", "niah_multikey_2", "vt"):
        (s,) = ruler.task_set(task, ESSAY, WORDS, small_engine.tokenizer, seed=208, samples=1,
                              template=chat.QWEN,
                              haystack_tokens=1024)
        for name in ("fp16", "semantic-80"):
            r = rq3.run(small_engine, s, rq3.condition(name), "question-blind")
            assert len(r["answer_tokens"]) <= s.answer_tokens
            assert r["score"] == ruler.score(r["answer"], s.expected)
            assert r["correct"] == (r["score"] == 1.0)
            assert len(r["target_page_tiers"]) == len(s.target_pages)
            assert "needle_page_tiers" not in r
            if name == "fp16":
                assert all(set(t) == {"FP16"} for t in r["target_page_tiers"])
            else:
                assert r["plan"] is not None and r["effective_bits"] < 16


def test_a_ruler_sample_is_refused_question_aware(tokenizer, built):
    from microinfer import rq3
    with pytest.raises(ValueError, match="question-blind"):
        rq3._check(None, "question-aware", built["vt"][0])
