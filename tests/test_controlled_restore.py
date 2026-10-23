"""A controlled restore between turns (#155).

RQ3's recovery task asks the question again after pressure has passed
(ADR-0017): the upgrades restore every page that holds a shadow, from it, to
FP16. As a controlled plan stands for a pressure event, a controlled restore
stands for the GREEN after it: the upgrade policy's plan (#103), for headroom
enough for every upgrade, with no monitor and no cooldown, applied at once.

Through a conversation (Seam A).
"""

import numpy as np
import pytest

from conftest import require_model
from microinfer import Engine, controller, monitor

NAME = "qwen2.5-0.5b-instruct"
FP16 = controller.TIERS.index("FP16")


def prompt(n, seed):
    return np.random.default_rng(seed).integers(1000, 100_000, n).astype(np.int32)


@pytest.fixture(scope="module")
def engine():
    e = Engine(require_model(NAME), kv_adaptive=True, kv_scoring=True, prefill_chunk=None)
    e.load_weights()
    return e


@pytest.mark.slow
@pytest.mark.parametrize("source", ["semantic", "uniform"])
def test_a_restore_brings_every_downgraded_page_back_to_fp16(engine, source):
    """After a plan at 80%, a restore upgrades every page it moved, in the
    policy's steps, to FP16: the cache holds 16 bits an element again, and
    the record says so."""
    with engine.conversation() as chat:
        chat.turn(prompt(2048, 155), 16, stop_at_eos=False)
        chat.plan(0.8, source=source, seed=7)
        below = int((chat.tiers() > FP16).sum())
        record = chat.restore(source=source, seed=7)
        tiers = chat.tiers()
        bits = chat.effective_bits()
    assert below > 0
    assert record.upgrades and record.ended == "applied" and record.pressure is None
    assert record.applied == len(record.plan) >= below
    assert (tiers[tiers >= 0] == FP16).all()
    assert bits["effective_bits"] == 16.0
    c = record.controlled
    assert c["restore"] and c["source"] == source and c["restored"]
    assert c["effective_bits"] == 16.0 and c["restored_bytes"] > 0


@pytest.mark.slow
def test_a_restored_cache_answers_as_one_never_planned(engine):
    """The shadows hold each page's FP16 bytes before its first downgrade,
    so after a plan and a restore the next turn decodes what it would have
    with no plan, logit for logit."""
    def turn_2(planned):
        logits = []
        with engine.conversation() as chat:
            chat.turn(prompt(1024, 1550), 8, stop_at_eos=False)
            if planned:
                chat.plan(0.8, source="uniform")
                chat.restore(source="uniform")
            tokens = chat.turn(prompt(16, 1551), 8, stop_at_eos=False, logits=logits)
        return tokens, np.stack(logits)

    restored, plain = turn_2(True), turn_2(False)
    assert (restored[0] == plain[0]).all()
    assert np.array_equal(restored[1], plain[1])


@pytest.mark.slow
def test_a_restore_with_nothing_downgraded_does_nothing(engine):
    with engine.conversation() as chat:
        chat.turn(prompt(512, 1552), 4, stop_at_eos=False)
        record = chat.restore(source="uniform")
    assert len(record.plan) == 0 and record.ended == "applied"
    assert record.controlled["restored"] and record.controlled["restored_bytes"] == 0


@pytest.mark.slow
def test_a_restore_is_refused_while_a_monitor_runs(engine):
    """A monitor running would make upgrades of its own, at its GREEN."""
    with engine.conversation() as chat:
        chat.turn(prompt(64, 1553), 2, stop_at_eos=False)
        engine.start_monitor(monitor.Monitor(reader=lambda: 8000 * 2**20, poll_s=0.01))
        try:
            with pytest.raises(ValueError, match="monitor"):
                chat.restore(source="uniform")
        finally:
            engine.stop_monitor()
