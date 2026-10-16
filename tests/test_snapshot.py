"""A conversation started from a snapshot of another (#172, Seam A).

A needle run spends most of its time on turn 1, which every source
condition holds alike: the engine at FP16, adaptive and scoring, the score
source read only by a plan, after turn 1. A conversation made for a
snapshot keeps on the host what its cache stores and can take one after
its turns, and a conversation
started from it holds what that one's cache held: its keys and values, its
scores, its length and the token not yet fed. What it decodes, the scores it
folds and the plans made on it are then the first conversation's, to the
bit, whatever came between.
"""

import numpy as np
import pytest

from conftest import require_model
from microinfer import Engine

NAME = "qwen2.5-0.5b-instruct"


def prompt(n, seed):
    return np.random.default_rng(seed).integers(1000, 100_000, n).astype(np.int32)


@pytest.fixture(scope="module")
def engine():
    # Turn 1 is prefilled in chunks, as the RQ3 tool's engine prefills it:
    # a snapshot holds every chunk's keys and values.
    e = Engine(require_model(NAME), kv_adaptive=True, kv_scoring=True, prefill_chunk=128)
    e.load_weights()
    return e


TURN_1, TURN_2 = prompt(300, 1), prompt(20, 2)


def scores(chat):
    return chat._cache.scores.download()


@pytest.mark.slow
@pytest.mark.parametrize("source", ["semantic", "uniform"])
def test_a_conversation_from_a_snapshot_goes_on_as_the_one_it_was_taken_from(engine, source):
    """Turn 1, a controlled plan and turn 2 in one conversation; and turn 1
    in another, recorded, its snapshot, and the plan and turn 2 in a third
    started from it: the same tokens, plan and scores."""
    with engine.conversation() as chat:
        first = chat.turn(TURN_1, 12, stop_at_eos=False)
        planned = chat.plan(0.8, source=source)
        answer = chat.turn(TURN_2, 10, stop_at_eos=False)
        after, tiers, positions = scores(chat), chat.tiers(), chat.positions
    with engine.conversation(for_snapshot=True) as chat:
        assert chat.turn(TURN_1, 12, stop_at_eos=False).tolist() == first.tolist()
        snapshot = chat.snapshot()
    # the snapshot outlives the conversation and its cache
    with engine.conversation(start=snapshot) as chat:
        assert chat.positions == len(TURN_1) + 11
        again = chat.plan(0.8, source=source)
        assert chat.turn(TURN_2, 10, stop_at_eos=False).tolist() == answer.tolist()
        np.testing.assert_array_equal(scores(chat), after)
        np.testing.assert_array_equal(chat.tiers(), tiers)
        assert chat.positions == positions
    assert again.controlled == planned.controlled
    for moves in ("layers", "pages", "current", "lower", "scores"):  # the same downgrades
        np.testing.assert_array_equal(getattr(again.plan, moves), getattr(planned.plan, moves))


@pytest.mark.slow
def test_one_snapshot_starts_many_conversations(engine):
    """Each condition of a sample starts from the same turn 1: a
    conversation started from a snapshot changes nothing of it."""
    with engine.conversation(for_snapshot=True) as chat:
        chat.turn(TURN_1, 8, stop_at_eos=False)
        snapshot = chat.snapshot()
    answers = []
    for share in (0.8, 0.7, 0.8):
        with engine.conversation(start=snapshot) as chat:
            chat.plan(share, source="semantic")
            answers.append(chat.turn(TURN_2, 8, stop_at_eos=False).tolist())
    assert answers[0] == answers[2]


@pytest.mark.slow
def test_what_a_snapshot_refuses(engine):
    with engine.conversation() as chat:
        chat.turn(TURN_1, 4)
        with pytest.raises(ValueError, match="made for one"):
            chat.snapshot()
    with engine.conversation(for_snapshot=True) as chat:
        chat.turn(TURN_1, 4)
        chat.snapshot()
        with pytest.raises(ValueError, match="once"):
            chat.snapshot()  # the host held what was stored once, and gave it away
    with engine.conversation(for_snapshot=True) as chat:
        chat.turn(TURN_1, 4)
        chat.plan(0.8, source="uniform")
        with pytest.raises(ValueError, match="FP16"):
            chat.snapshot()  # what was stored is no longer what the cache holds
    with engine.conversation(for_snapshot=True) as chat:
        chat.turn(TURN_1, 4)
        snapshot = chat.snapshot()
    with pytest.raises(ValueError, match="not made for one"):
        engine.conversation(for_snapshot=True, start=snapshot)
    engine.kv_tier = "INT8"
    try:
        with pytest.raises(ValueError, match="engine"):
            engine.conversation(start=snapshot)  # its pages would be born at INT8
    finally:
        engine.kv_tier = "FP16"
