"""The importance score: an EWMA of each page's attention mass, on the device
(#99, Seam B, and the paged cache that keeps it).

Every decode step, each (layer, page)'s attention mass (#98), averaged over
the layer's query heads, is folded into an exponentially weighted moving
average: score = alpha * mass + (1 - alpha) * score, alpha 0.2 by default. A
page's first observation seeds its score; a page never observed has none
(NaN), and the controller, to come, will order it by position (#88). The
scores stay on the device, and are copied to the host only when asked for.

The device's arithmetic is fp32, each operation rounded once, so a host
computation in fp32 in the same order is its reference to the bit: the
EWMA's arithmetic is its specification, and the device must carry it out
exactly. An independent float64 reference, as CONTRIBUTING's gate asks of a
kernel, bounds what that arithmetic loses, in ulps of fp32 at 1.0, the
scores' scale.
"""

import numpy as np
import pytest
from test_paged_engine import decode
from test_static_tiers import MODEL, P, REPO

from conftest import each, require_model
from microinfer import Engine, _microinfer, model
from microinfer.golden import GoldenError, GoldenSet

device = _microinfer.device
f32 = np.float32


def host_ewma(steps, layers, max_pages, alpha):
    """The scores a sequence of folds should leave, in fp32, in the
    device's order: each head's mass summed in order and divided by the
    head count, then alpha * mass + (1 - alpha) * score, a NaN seeded."""
    scores = np.full((layers, max_pages), np.nan, f32)
    alpha, keep = f32(alpha), f32(1) - f32(alpha)
    for layer, mass in steps:
        heads, pages = mass.shape
        total = np.zeros(pages, f32)
        for h in range(heads):
            total = (total + mass[h]).astype(f32)
        mean = (total / f32(heads)).astype(f32)
        old = scores[layer, :pages]
        scores[layer, :pages] = np.where(np.isnan(old), mean,
                                         (alpha * mean).astype(f32) + (keep * old).astype(f32))
    return scores


#: Independent of the order: a head mean's rounding, and the EWMA's,
#: which contracts its past error by 1 - alpha a step.
EPS = float(np.finfo(np.float32).eps)
EWMA_ULPS = 4


def float64_ewma(steps, layers, max_pages, alpha):
    """The same EWMA in float64, with the head mean exact."""
    scores = np.full((layers, max_pages), np.nan)
    for layer, mass in steps:
        mean = mass.astype(np.float64).mean(axis=0)
        old = scores[layer, :mass.shape[1]]
        scores[layer, :mass.shape[1]] = np.where(np.isnan(old), mean,
                                                 alpha * mean + (1 - alpha) * old)
    return scores


def random_masses(rng, layers, heads, steps, start_pages):
    """A decode's worth of masses: every layer each step, one more page
    every few steps, each head's row a distribution over the pages."""
    out = []
    for t in range(steps):
        pages = start_pages + t // 3
        for layer in range(layers):
            m = rng.random((heads, pages)).astype(f32)
            out.append((layer, (m / m.sum(axis=1, keepdims=True)).astype(f32)))
    return out


def fold_all(scores, steps):
    for layer, mass in steps:
        heads, pages = mass.shape
        scores.fold(layer, device.upload_f32(mass.ravel()), heads, pages)


def test_the_device_ewma_is_a_host_computation():
    """Over a decode's worth of masses for several layers, at the default
    alpha and another, the device's scores are the host's, to the bit."""
    def matches(alpha):
        rng = np.random.default_rng(99)
        layers, heads, max_pages = 3, 12, 40
        steps = random_masses(rng, layers, heads, 30, 5)
        scores = (device.ImportanceScores(layers, max_pages) if alpha is None else
                  device.ImportanceScores(layers, max_pages, alpha))
        assert scores.alpha == pytest.approx(0.2 if alpha is None else alpha)
        fold_all(scores, steps)
        got = scores.download()
        np.testing.assert_array_equal(got, host_ewma(steps, layers, max_pages, scores.alpha))
        exact = float64_ewma(steps, layers, max_pages, scores.alpha)
        assert (np.isnan(got) == np.isnan(exact)).all()
        seen = ~np.isnan(exact)
        assert np.abs(got[seen] - exact[seen]).max() < EWMA_ULPS * EPS

    each([None, 0.5], matches, name=lambda a: f"alpha={a}")


def test_a_page_s_first_observation_seeds_its_score():
    """A page's score after its first observation is that observation, not
    alpha times it; a page never observed has no score."""
    scores = device.ImportanceScores(2, 8)
    mass = np.array([[0.5, 0.25, 0.25], [0.7, 0.2, 0.1]], f32)
    scores.fold(1, device.upload_f32(mass.ravel()), 2, 3)
    got = scores.download()
    np.testing.assert_array_equal(got[1, :3], ((mass[0] + mass[1]) / f32(2)).astype(f32))
    assert np.isnan(got[1, 3:]).all() and np.isnan(got[0]).all()


def test_no_copy_to_the_host_without_a_request():
    """Folding copies nothing to the host, however many steps; each
    request is one copy."""
    rng = np.random.default_rng(3)
    scores = device.ImportanceScores(2, 16)
    fold_all(scores, random_masses(rng, 2, 4, 20, 2))
    assert scores.downloads == 0
    scores.download()
    assert scores.downloads == 1
    fold_all(scores, random_masses(rng, 2, 4, 5, 2))
    assert scores.downloads == 1


def test_what_the_scores_refuse():
    with pytest.raises(ValueError, match="alpha"):
        device.ImportanceScores(2, 8, 0.0)
    with pytest.raises(ValueError, match="alpha"):
        device.ImportanceScores(2, 8, 1.5)
    scores = device.ImportanceScores(2, 8)
    with pytest.raises(ValueError, match="pages"):
        scores.fold(0, device.empty_f32(4 * 9), 4, 9)
    with pytest.raises(ValueError, match="mass"):
        scores.fold(0, device.empty_f32(4 * 3 - 1), 4, 3)
    with pytest.raises(IndexError):
        scores.fold(2, device.empty_f32(4 * 3), 4, 3)


# -- the paged cache that keeps them ---------------------------------------------------


@pytest.fixture(scope="module")
def engine() -> Engine:
    e = Engine(require_model(MODEL))
    e.load_weights()
    return e


@pytest.fixture(scope="module")
def golden() -> GoldenSet:
    try:
        return GoldenSet(REPO / "tests" / "golden" / MODEL)
    except GoldenError as exc:
        pytest.skip(str(exc))


def test_a_scoring_cache_folds_every_decode_step_and_copies_nothing(engine, golden):
    """A paged cache that scores folds each layer's mass into its scores at
    every decode step, and at no prefill: after a real prefill and decode,
    every page a decode step attended has a score in [0, 1] in every layer,
    and nothing has been copied to the host until asked for. At FP16 and at
    a quantised tier. A cache that does not score has no scores."""
    cfg = engine.config
    ids = golden["medium-01"].token_ids

    def scores(tier):
        cache = model.PagedCache(cfg, getattr(_microinfer.Tier, tier), scoring=True)
        new = P + 3
        decode(engine, ids, new, cache)
        assert cache.scores.downloads == 0
        got = cache.scores.download()
        attended = -(-(len(ids) + new - 1) // P)
        assert got.shape[0] == cfg.num_hidden_layers
        assert np.isfinite(got[:, :attended]).all()
        assert ((got[:, :attended] >= 0) & (got[:, :attended] <= 1)).all()
        assert np.isnan(got[:, attended:]).all()
        assert cache.scores.downloads == 1

    each(["FP16", "INT4"], scores)
    assert model.PagedCache(cfg).scores is None


def test_scoring_changes_no_token(engine, golden):
    """The masses change no bit of attention's output (#98), so a session
    that scores generates what one that does not generates."""
    ids = golden["long-03"].token_ids
    want = engine.generate(ids, 24, stop_at_eos=False)
    engine.kv_scoring = True
    try:
        got = engine.generate(ids, 24, stop_at_eos=False)
    finally:
        engine.kv_scoring = False
    np.testing.assert_array_equal(got, want)


def test_scoring_is_chosen_by_configuration():
    path = require_model(MODEL)
    assert Engine(path).kv_scoring is False
    assert Engine(path, kv_scoring=True).kv_scoring is True
    with pytest.raises(ValueError, match="contiguous"):
        Engine(path, kv_cache="contiguous", kv_scoring=True)


def test_a_cache_can_score_every_r_th_decode_step(engine, golden):
    """With score_every R, the fallback #88 allows if scoring every step
    costs decode more than noise (#100), a decode step folds its masses
    only when the positions it attends number a multiple of R: every layer
    of such a step, and no layer of another. R 1 is every step."""
    cfg = engine.config
    ids = golden["medium-01"].token_ids
    new = 2 * P + 3

    class Counting:
        def __init__(self, scores):
            self.scores, self.folds = scores, []

        def fold(self, layer, mass, heads, pages):
            self.folds.append(layer)
            self.scores.fold(layer, mass, heads, pages)

    def folds(every):
        cache = model.PagedCache(cfg, scoring=True, score_every=every)
        cache.scores = Counting(cache.scores)
        decode(engine, ids, new, cache)
        return len(cache.scores.folds)

    steps = range(len(ids) + 1, len(ids) + new)  # seq_k of each decode step
    for every in (1, 3, 8):
        want = sum(seq_k % every == 0 for seq_k in steps) * cfg.num_hidden_layers
        assert folds(every) == want, every
    with pytest.raises(ValueError, match="score_every"):
        model.PagedCache(cfg, scoring=True, score_every=0)
    path = require_model(MODEL)
    assert Engine(path).kv_score_every == Engine.DEFAULT_SCORE_EVERY
    assert Engine(path, kv_score_every=2).kv_score_every == 2
    with pytest.raises(ValueError, match="kv_score_every"):
        Engine(path, kv_score_every=0)
