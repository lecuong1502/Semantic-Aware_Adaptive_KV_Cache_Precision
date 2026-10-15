"""ADR-0017's needle set (#146).

Haystacks from WikiText-2's test split joined to about 8K tokens, a
synthetic needle naming a code, at 10 depths x 5 samples, from a fixed seed,
each with its turn-1 request and its question. Turn 1 is built from pieces
tokenized apart, so the needle's positions, and the pages that hold them,
are known exactly.
"""

import importlib.util
from pathlib import Path

import pytest

from conftest import require_model
from microinfer import Engine, needles

REPO = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("perplexity", REPO / "tools" / "perplexity.py")
perplexity = importlib.util.module_from_spec(spec)
spec.loader.exec_module(perplexity)

NAME = "qwen2.5-1.5b-instruct"
IM_END = 151645


@pytest.fixture(scope="module")
def engine():
    return Engine(require_model(NAME))  # the tokenizer alone: no weights


@pytest.fixture(scope="module")
def text():
    return perplexity.held_out_text()


@pytest.fixture(scope="module")
def built(engine, text):
    return needles.needle_set(text, engine.tokenizer, seed=146)


def test_ten_depths_of_five_samples(built):
    assert len(built) == 50
    depths = [s.depth for s in built]
    assert sorted(set(depths)) == pytest.approx([d / 10 for d in range(10)])
    assert all(depths.count(d) == 5 for d in set(depths))
    assert len({s.sample_id for s in built}) == len({s.code for s in built}) == 50


def test_the_same_seed_gives_the_same_set_byte_for_byte(engine, text, built):
    again = needles.needle_set(text, engine.tokenizer, seed=146)
    assert needles.dumps(again) == needles.dumps(built)
    other = needles.needle_set(text, engine.tokenizer, seed=147)
    assert needles.dumps(other) != needles.dumps(built)


def test_the_needle_sits_where_the_sample_says(engine, built):
    """The needle's tokens are at its recorded positions of turn 1, and its
    pages are those positions' pages: what a layer-0 position holds."""
    for s in built:
        start, end = s.needle_positions
        assert s.turn_1[start:end].tolist() == engine.encode(needles.piece(s.needle)).tolist()
        assert s.needle_pages == (start // s.page_tokens, (end - 1) // s.page_tokens)


def test_the_haystack_is_about_8k_tokens_and_the_depth_where_it_is(built):
    """Each document is the haystack's tokens and the needle's, and the
    needle starts within a paragraph's length of its depth."""
    for s in built:
        start = s.needle_positions[0]
        assert s.haystack_tokens == needles.DEFAULT_HAYSTACK_TOKENS
        assert 7900 < len(s.turn_1) < 8100
        offset = start - s.document_start
        assert abs(offset - s.depth * s.haystack_tokens) <= 400, s.sample_id
    assert min(s.needle_positions[0] - s.document_start for s in built if s.depth == 0) == 0


def text_of(engine, ids):
    return engine.tokenizer.decode(ids.tolist(), skip_special_tokens=False)


def test_the_code_is_in_turn_one_once_and_not_in_the_question(engine, built):
    for s in built:
        assert text_of(engine, s.turn_1).count(s.code) == 1
        assert s.code not in text_of(engine, s.turn_2) and s.name in s.question
    assert "Summarize the document above in two sentences." in text_of(engine, built[0].turn_1)


def test_each_turn_is_in_the_chat_template(engine, built):
    one, two = text_of(engine, built[0].turn_1), text_of(engine, built[0].turn_2)
    assert one.startswith("<|im_start|>system\n") and one.endswith("<|im_start|>assistant\n")
    assert two.startswith("<|im_end|>\n<|im_start|>user\n") and two.endswith(
        "<|im_start|>assistant\n")


@pytest.mark.parametrize("answer, right", [
    ("XQT-4827", True),
    ("The code is XQT-4827.", True),
    ("XQT-482", False),
    ("xqt-4827", False),
    ("XQT 4827", False),
    ("", False),
])
def test_an_answer_is_correct_iff_it_contains_the_code(answer, right):
    assert needles.correct(answer, "XQT-4827") is right


def test_a_small_text_still_gives_a_whole_set(engine):
    """Fewer tokens than a haystack: the articles wrap around."""
    text = " = A = \n The first article , short .\n = B = \n The second , shorter .\n"
    small = needles.needle_set(text, engine.tokenizer, seed=1, haystack_tokens=300)
    assert len(small) == 50 and all(s.haystack_tokens == 300 for s in small)


def test_every_needle_lies_outside_the_recency_floor(built):
    """A needle within the last 128 positions of turn 1 would never be
    compressed, and could test nothing: at 90% it is some 800 before."""
    from microinfer import controller

    for s in built:
        assert s.needle_positions[1] <= len(s.turn_1) - controller.DEFAULT_RECENCY_FLOOR


def test_wikitext_2_s_test_split_holds_60_articles(text):
    """A title is a line ' = Title = ' after a blank one: lines of a formula
    such as ' = 1 henry and capacitance C = ' are not titles."""
    found = needles.articles(text)
    assert len(found) == 60
    titles = [title for title, _ in found]
    assert "Constant k filter" in titles and "2003 Pacific typhoon season" in titles
    assert not any("henry" in title or "rad / s" in title for title in titles)


def test_turn_two_after_a_turn_one_that_stopped_at_its_end_of_turn(engine, built):
    """A turn 1 cut short at its 64 tokens needs its message closed; one that
    stopped at <|im_end|> has fed it already, so turn 2 goes on after it."""
    for s in built:
        assert s.turn_2.tolist() == [IM_END] + s.turn_2_after_eos.tolist()
    assert text_of(engine, built[0].turn_2_after_eos).startswith("\n<|im_start|>user\n")


def test_the_samples_of_a_depth_have_their_own_haystacks(built):
    for depth in needles.DEPTHS:
        firsts = [s.first_article for s in built if s.depth == depth]
        assert len(set(firsts)) == len(firsts)


def test_more_samples_a_depth_leave_the_first_ones_as_they_are(engine, text, built):
    """Screening replaces a sample FP16 answers wrongly (#148) with the next
    of its depth: drawing more leaves the first five, byte for byte."""
    more = needles.needle_set(text, engine.tokenizer, seed=146, samples_per_depth=7)
    assert len(more) == 70
    first_five = [s for s in more if int(s.sample_id.split("-")[1]) < 5]
    assert needles.dumps(first_five) == needles.dumps(built)
