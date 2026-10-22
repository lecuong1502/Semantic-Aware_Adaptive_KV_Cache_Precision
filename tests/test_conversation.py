"""A conversation: one cache, turn by turn (#141, Seam A).

RQ3 compresses a cache between two turns (ADR-0017), so the engine must keep
one cache across turns. A turn appends a prompt to what the cache holds and
decodes after it; the last token a turn chose is fed back at the start of the
next, as the token after it would have been in one generation. The scorer's
scores live with the cache, so turn 2 starts from turn 1's.

The reference here is the same tokens run by hand through the model on one
cache: the prompt of turn 1 prefilled, its tokens decoded one by one, then
turn 1's last token with the prompt of turn 2 prefilled, and its tokens
decoded. A conversation does exactly that, so its logits, and its scores,
are the reference's to the bit.
"""

import gc

import numpy as np
import pytest
from test_emergency_downgrade import failing_at

from conftest import require_model
from microinfer import Engine, _microinfer, model, nvml
from microinfer.engine import EXHAUSTED

NAME = "qwen2.5-0.5b-instruct"
Tier = _microinfer.Tier


def prompt(n, seed):
    return np.random.default_rng(seed).integers(1000, 100_000, n).astype(np.int32)


def loaded(**options):
    e = Engine(require_model(NAME), prefill_chunk=None, **options)
    e.load_weights()
    return e


@pytest.fixture(scope="module")
def fp16():
    return loaded()


def by_hand(engine, cache, turns, after_each=lambda: None):
    """Each turn's (prompt, new tokens) run through the model on `cache`, as
    one generation would run them, `after_each` called after each turn: the
    logits of every token chosen, and the tokens of each turn."""
    m = engine._model
    logits, tokens, unfed = [], [], []
    for ids, new in turns:
        ids = np.concatenate([np.asarray(unfed, np.int32), ids])
        ws = model.Workspace(engine.config, rows=len(ids))
        m.run(ws, cache, ids)
        logits.append(m.last_logits(ws, len(ids)))
        out = [m.greedy_last(ws, len(ids))]
        step = model.Workspace(engine.config, rows=1)
        while len(out) < new:
            m.run(step, cache, np.array(out[-1:], np.int32))
            logits.append(m.last_logits(step, 1))
            out.append(m.greedy_last(step, 1))
        tokens.append(out)
        unfed = out[-1:]
        after_each()
    return np.stack(logits), tokens


@pytest.mark.slow
@pytest.mark.parametrize("tier", ["FP16", "INT8"])
def test_two_turns_give_the_logits_of_one_generation_of_the_same_tokens(fp16, tier):
    """Two turns over one cache, at FP16 and at a quantised tier, against the
    same tokens run by hand on one cache: every logit the same, to the bit,
    and every token."""
    engine = fp16 if tier == "FP16" else loaded(kv_tier=tier)
    first, second = prompt(300, 141), prompt(40, 1410)
    logits = []
    with engine.conversation() as chat:
        out1 = chat.turn(first, 12, stop_at_eos=False, logits=logits)
        out2 = chat.turn(second, 8, stop_at_eos=False, logits=logits)
    cache = model.PagedCache(engine.config, getattr(Tier, tier))
    expected, tokens = by_hand(engine, cache, [(first, 12), (second, 8)])
    assert [list(out1), list(out2)] == tokens
    assert np.array_equal(np.stack(logits), expected)


@pytest.mark.slow
def test_turn_two_starts_from_turn_one_s_scores():
    """The scores the conversation holds after each turn are those of one
    scoring cache the same tokens were run through by hand, to the bit. Had
    turn 2 started afresh, the pages of turn 1 would hold turn 2's mass
    alone, and the second comparison would fail."""
    engine = loaded(kv_scoring=True, kv_score_every=1)
    turns = [(prompt(300, 142), 16), (prompt(40, 1420), 12)]
    held = []
    with engine.conversation() as chat:
        for ids, new in turns:
            chat.turn(ids, new, stop_at_eos=False)
            held.append(chat.scores())
    cache = model.PagedCache(engine.config, Tier.FP16, scoring=True, score_every=1)
    expected = []
    by_hand(engine, cache, turns, after_each=lambda: expected.append(cache.scores.download()))
    assert np.isfinite(held[0][:, :9]).all()  # turn 1's pages were observed
    for got, want in zip(held, expected):
        assert np.array_equal(got, want, equal_nan=True)


@pytest.mark.slow
def test_a_conversation_releases_its_cache_when_it_ends(fp16):
    """Leaving the conversation releases its cache: what this process holds,
    by the driver's account, is back within a granule of where it was. A
    generation of the same size runs first, so that what the first run of
    a kind takes lazily, cuBLAS's workspace among it, is in the baseline."""
    fp16.generate(prompt(2000, 143), 4, stop_at_eos=False)  # warm
    gc.collect()
    before = nvml.settled_own_used_bytes()
    with fp16.conversation() as chat:
        chat.turn(prompt(2000, 143), 4, stop_at_eos=False)
        chat.turn(prompt(2000, 1430), 4, stop_at_eos=False)
        assert nvml.settled_own_used_bytes() > before
    gc.collect()
    assert nvml.settled_own_used_bytes() <= before + _microinfer.granule_bytes()


@pytest.mark.slow
def test_a_turn_after_the_conversation_ends_is_refused(fp16):
    with fp16.conversation() as chat:
        chat.turn(prompt(10, 144), 2, stop_at_eos=False)
    with pytest.raises(RuntimeError, match="ended"):
        chat.turn(prompt(10, 1440), 2)


@pytest.mark.slow
def test_a_turn_that_ends_exhausted_ends_the_conversation(monkeypatch):
    """An adaptive engine whose step is refused memory, with nothing outside
    the recency floor to downgrade, ends the turn exhausted (#107): the
    conversation ends with it, releasing its cache, and the next turn is
    refused."""
    engine = loaded(kv_adaptive=True)
    with engine.conversation() as chat:
        chat.turn(prompt(40, 145), 4, stop_at_eos=False)
        failing_at(monkeypatch, 40 + 4 + 6, times=None)
        chat.turn(prompt(6, 1450), 4, stop_at_eos=False)
        assert engine.ending.state == EXHAUSTED
        assert chat.positions == 0
        with pytest.raises(RuntimeError, match="exhausted"):
            chat.turn(prompt(6, 1451), 2)


@pytest.mark.slow
def test_a_turn_that_raises_ends_the_conversation(fp16, monkeypatch):
    """Any other failure leaves a turn half written, so the conversation
    ends rather than go on over that cache."""
    with fp16.conversation() as chat:
        chat.turn(prompt(40, 146), 4, stop_at_eos=False)

        def interrupted(*_):
            raise KeyboardInterrupt

        monkeypatch.setattr(model.Model, "greedy_last", interrupted)
        with pytest.raises(KeyboardInterrupt):
            chat.turn(prompt(6, 1460), 4)
        monkeypatch.undo()
        with pytest.raises(RuntimeError, match="ended"):
            chat.turn(prompt(6, 1461), 2)


@pytest.mark.slow
def test_between_turns_the_footprint_counts_the_conversation_s_cache(fp16):
    """Between turns the cache is alive though no turn runs: the engine's
    footprint counts it, as it does during a turn, and not once it is
    released."""
    with fp16.conversation() as chat:
        chat.turn(prompt(2000, 147), 4, stop_at_eos=False)
        between = fp16.footprint()
        assert between.kv_cache == chat.nbytes > 0
    assert fp16.footprint().kv_cache == 0


# -- teacher forcing (#153) ----------------------------------------------------------------

def forced_by_hand(engine, cache, prefix, steps):
    """`prefix` prefilled into `cache` at once, then `steps` fed one position
    at a time: the logits after each of them."""
    m = engine._model
    m.run(model.Workspace(engine.config, rows=len(prefix)), cache, prefix)
    one, logits = model.Workspace(engine.config, rows=1), []
    for token in steps:
        m.run(one, cache, np.array([token], np.int32))
        logits.append(m.last_logits(one, 1))
    return np.stack(logits)


@pytest.mark.slow
def test_forcing_feeds_the_tokens_given_and_gives_the_logits_after_each():
    """Continuation fidelity teacher-forces a passage (#153): its tokens, not
    the ones the model would choose, go into the cache, one position a step,
    so the scorer folds in each as it does a decoded one. The logits after
    each and the scores are those of the same tokens run by hand, to the bit;
    a prefill forced at once gives no logits, and is the prompt of a turn."""
    engine = loaded(kv_scoring=True, kv_score_every=1)
    prefix, steps = prompt(300, 153), prompt(24, 1530)
    with engine.conversation() as chat:
        assert chat.force(prefix) is None
        logits = chat.force(steps, one_by_one=True, logits=True)
        assert chat.positions == len(prefix) + len(steps)
        scores = chat.scores()
    cache = model.PagedCache(engine.config, Tier.FP16, scoring=True, score_every=1)
    assert np.array_equal(logits, forced_by_hand(engine, cache, prefix, steps))
    assert np.array_equal(scores, cache.scores.download(), equal_nan=True)


@pytest.mark.slow
def test_forcing_goes_on_from_forcing_and_not_from_a_turn(fp16):
    """A turn leaves the token it chose last to be fed first; forcing feeds
    only the tokens it is given, so it is refused after a turn. A turn after
    forcing appends its prompt to the tokens forced, as a turn after a turn
    does, and decodes what the same tokens run by hand decode."""
    with fp16.conversation() as chat:
        chat.turn(prompt(40, 1531), 2)
        with pytest.raises(RuntimeError, match="turn"):
            chat.force(prompt(4, 1532))
    forced, second = prompt(40, 1533), prompt(8, 1534)
    with fp16.conversation() as chat:
        chat.force(forced)
        out = chat.turn(second, 4, stop_at_eos=False)
    cache = model.PagedCache(fp16.config, Tier.FP16)
    fp16._model.run(model.Workspace(fp16.config, rows=len(forced)), cache, forced)
    _, tokens = by_hand(fp16, cache, [(second, 4)])
    assert out.tolist() == tokens[0]


@pytest.mark.slow
def test_forcing_that_runs_out_ends_the_conversation_and_says_so(fp16, monkeypatch):
    """A step that fails again after an emergency plan, or has nothing to
    downgrade, ends forcing as it would end a turn exhausted; forcing has no
    tokens to end with, so it raises the engine's ForcingExhausted."""
    from microinfer.engine import ForcingExhausted, _Exhausted

    def exhausted(*args, **kwargs):
        raise _Exhausted("out of memory; nothing to downgrade")

    with fp16.conversation() as chat:
        chat.force(prompt(40, 1535))
        monkeypatch.setattr(fp16, "_run", exhausted)
        with pytest.raises(ForcingExhausted, match="nothing to downgrade"):
            chat.force(prompt(4, 1536), one_by_one=True)
        with pytest.raises(RuntimeError, match="ended exhausted"):
            chat.force(prompt(4, 1536))
