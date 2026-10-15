"""The scorer's EWMA weight, on the engine (#143, Seam A).

ADR-0018's α ablation is configuration: the engine takes the weight the
importance scorer folds each step's attention mass in with, 0.2 by default,
passes it to its cache's scorer, and records it in every plan made from
semantic scores, so that a run says which α its plans came from.
"""

import numpy as np
import pytest
from test_adaptive_engine import simulated

from conftest import require_model
from microinfer import Engine, model, monitor

NAME = "qwen2.5-0.5b-instruct"


def prompt(n, seed):
    return np.random.default_rng(seed).integers(1000, 100_000, n).astype(np.int32)


def loaded(**options):
    e = Engine(require_model(NAME), prefill_chunk=None, **options)
    e.load_weights()
    return e


def test_the_default_is_the_scorer_s():
    assert Engine.DEFAULT_SCORE_ALPHA == model.device.ImportanceScores.default_alpha == \
        pytest.approx(0.2)


@pytest.mark.parametrize("alpha", [0.0, -0.1, 1.5])
def test_a_weight_outside_zero_to_one_is_refused(alpha):
    with pytest.raises(ValueError, match="alpha"):
        Engine(require_model(NAME), kv_score_alpha=alpha)


@pytest.mark.slow
def test_the_weight_reaches_the_cache_s_scorer():
    """The same turn scored at α = 0.5 and at the default gives different
    scores, and each conversation's scorer folds at its engine's α."""
    held = {}
    for alpha in (None, 0.5):
        options = {} if alpha is None else {"kv_score_alpha": alpha}
        engine = loaded(kv_scoring=True, kv_score_every=1, **options)
        with engine.conversation() as chat:
            chat.turn(prompt(300, 143), 16, stop_at_eos=False)
            held[alpha] = chat.scores()
            assert chat.score_alpha == pytest.approx(0.2 if alpha is None else 0.5)
    assert not np.array_equal(held[None], held[0.5], equal_nan=True)


@pytest.mark.slow
def test_a_plan_from_semantic_scores_records_the_weight():
    """A controlled plan and a pressure plan from semantic scores record
    the α they came from; a plan from uniform scores records none."""
    engine = loaded(kv_adaptive=True, kv_scoring=True, kv_score_alpha=0.5)
    with engine.conversation() as chat:
        chat.turn(prompt(1024, 1431), 8, stop_at_eos=False)
        assert chat.plan(0.5, source="semantic").score_alpha == pytest.approx(0.5)
        assert chat.plan(0.8, source="uniform").score_alpha is None
    engine.start_monitor(simulated(100))
    try:
        engine.generate(prompt(1024, 1432), 8, stop_at_eos=False)
    finally:
        engine.stop_monitor()
    pressured = [p for p in engine.plans if p.pressure is not None and len(p.plan)]
    assert pressured and all(p.score_alpha == pytest.approx(0.5) for p in pressured)
    assert all(p.plan.level == monitor.RED for p in pressured)
