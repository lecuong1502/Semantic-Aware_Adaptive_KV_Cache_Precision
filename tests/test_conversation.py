"""A conversation: one cache, turn by turn (#141, Seam A).

RQ3 compresses a cache between two turns (ADR-0017), so the engine must keep
one cache across turns. A turn appends a prompt to what the cache holds and
decodes after it; the last token a turn chose is fed back at the start of the
next, as the token after it would have been in one generation. The scorer's
scores live with the cache, so turn 2 starts from turn 1's.

The reference here is the same tokens run by hand through the model on one
cache: the prompt of turn 1 prefilled, its tokens decoded one by one, then
turn 1's last token with the prompt of turn 2 prefilled, and its tokens
decoded. A conversation does exactly that, so its logits are the reference's
to the bit.
"""

import gc

import numpy as np
import pytest

from conftest import require_model
from microinfer import Engine, _microinfer, model, nvml

NAME = "qwen2.5-0.5b-instruct"
Tier = _microinfer.Tier


def prompt(n, seed):
    return np.random.default_rng(seed).integers(1000, 100_000, n).astype(np.int32)


def by_hand(engine, tier, turns):
    """Each turn's (prompt, new tokens) run through the model on one fresh
    cache at `tier`, as one generation would run them: the logits of every
    token chosen, and the tokens."""
    m = engine._model
    cache = model.PagedCache(engine.config, tier)
    logits, tokens, pending = [], [], []
    for ids, new in turns:
        ids = np.concatenate([np.asarray(pending, np.int32), ids])
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
        pending = out[-1:]
    return np.stack(logits), tokens


@pytest.mark.parametrize("tier", ["FP16", "INT8"])
def test_two_turns_give_the_logits_of_one_generation_of_the_same_tokens(tier):
    """Two turns over one cache, at FP16 and at a quantised tier, against the
    same tokens run by hand on one cache: every logit the same, to the bit,
    and every token."""
    engine = Engine(require_model(NAME), kv_tier=tier, prefill_chunk=None)
    engine.load_weights()
    first, second = prompt(300, 141), prompt(40, 1410)
    logits = []
    with engine.conversation() as chat:
        out1 = chat.turn(first, 12, stop_at_eos=False, logits=logits)
        out2 = chat.turn(second, 8, stop_at_eos=False, logits=logits)
    expected, tokens = by_hand(engine, getattr(Tier, tier), [(first, 12), (second, 8)])
    assert [list(out1), list(out2)] == tokens
    assert np.array_equal(np.stack(logits), expected)


def test_turn_two_starts_from_turn_one_s_scores():
    """The scores after turn 1 are those turn 2 starts from: read between
    the turns, they are what the cache holds, and turn 2 folds its mass into
    them rather than starting afresh."""
    engine = Engine(require_model(NAME), kv_scoring=True, kv_score_every=1,
                    prefill_chunk=None)
    engine.load_weights()
    with engine.conversation() as chat:
        chat.turn(prompt(300, 142), 16, stop_at_eos=False)
        after_first = chat.scores()
        assert np.isfinite(after_first[:, :9]).all()  # the turn's pages were observed
        chat.turn(prompt(40, 1420), 1, stop_at_eos=False)
        # One token of turn 2 decodes no step: nothing new is folded in.
        assert np.array_equal(chat.scores(), after_first, equal_nan=True)
        chat.turn(prompt(8, 1421), 6, stop_at_eos=False)
        later = chat.scores()
    with engine.conversation() as fresh:
        fresh.turn(prompt(8, 1421), 6, stop_at_eos=False)
        alone = fresh.scores()
    assert not np.array_equal(later[:, :1], alone[:, :1], equal_nan=True)


def test_a_conversation_releases_its_cache_when_it_ends():
    """Leaving the conversation releases its cache: what this process holds,
    by the driver's account, is back within a granule of where it was. A
    generation of the same size runs first, so that what the first run of
    a kind takes lazily, cuBLAS's workspace among it, is in the baseline."""
    engine = Engine(require_model(NAME), prefill_chunk=None)
    engine.load_weights()
    engine.generate(prompt(2000, 143), 4, stop_at_eos=False)  # warm
    gc.collect()
    before = nvml.settled_own_used_bytes()
    with engine.conversation() as chat:
        chat.turn(prompt(2000, 143), 4, stop_at_eos=False)
        chat.turn(prompt(2000, 1430), 4, stop_at_eos=False)
        assert nvml.settled_own_used_bytes() > before
    gc.collect()
    assert nvml.settled_own_used_bytes() <= before + _microinfer.granule_bytes()


def test_a_turn_after_the_conversation_ends_is_refused():
    engine = Engine(require_model(NAME), prefill_chunk=None)
    engine.load_weights()
    with engine.conversation() as chat:
        chat.turn(prompt(10, 144), 2, stop_at_eos=False)
    with pytest.raises(RuntimeError, match="ended"):
        chat.turn(prompt(10, 1440), 2)
