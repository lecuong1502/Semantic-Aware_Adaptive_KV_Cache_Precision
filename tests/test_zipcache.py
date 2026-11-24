"""ZipCache's normalised attention as a score source (#210).

The reference is ZipCache's equation 8, a token's attention summed over the
probe queries and divided by how many of them could attend to it, computed
here token by token from a causal attention matrix; the source holds only
each page's mass, and gives each page the mean over its (query, token)
pairs. Then the pass that takes the masses, on the engine: the probes fed a
position at a time, everything between them in chunks.
"""

import numpy as np
import pytest

from conftest import require_model
from microinfer import Engine, _microinfer, controller, model, rq3, zipcache

NAME = "qwen2.5-0.5b-instruct"
P = _microinfer.device.page_tokens


def prompt(n, seed):
    return np.random.default_rng(seed).integers(1000, 100_000, n).astype(np.int32)


# -- the probes ------------------------------------------------------------------------


def test_the_probes_are_a_twentieth_recent_and_a_twentieth_random():
    """ZipCache's hybrid probes (its section 4.3): the last 5% of the
    positions, and 5% drawn from those before them, in order, none twice."""
    probes = zipcache.probe_positions(1000, seed=3)
    assert len(probes) == 100 and len(set(probes.tolist())) == 100
    assert (np.diff(probes) > 0).all()
    assert probes[-50:].tolist() == list(range(950, 1000))
    assert (probes[:50] < 950).all()
    assert probes[0] < 100 and probes[49] > 850  # drawn over the whole of them


def test_the_random_probes_are_the_seed_s():
    again = zipcache.probe_positions(1000, seed=3)
    assert (again == zipcache.probe_positions(1000, seed=3)).all()
    assert (again != zipcache.probe_positions(1000, seed=4)).any()


def test_the_last_position_is_always_a_probe():
    """So that every page has been attended: a short cache's too."""
    for n in (1, 2, 5, 19, 33):
        probes = zipcache.probe_positions(n, seed=0)
        assert probes[-1] == n - 1 and len(set(probes.tolist())) == len(probes)
    with pytest.raises(ValueError, match="position"):
        zipcache.probe_positions(0, seed=0)


# -- the masses ------------------------------------------------------------------------


class Mass:
    """A layer's masses on the device, as fold is given them."""

    def __init__(self, rows):
        self.rows = np.asarray(rows, np.float32)

    def to_numpy(self):
        return np.concatenate([self.rows.ravel(), np.zeros(7, np.float32)])  # a buffer, longer


def causal_attention(heads, n, seed):
    """Rows of a causal softmax, (heads, n, n)."""
    a = np.random.default_rng(seed).exponential(size=(heads, n, n)) * np.tri(n)
    return a / a.sum(axis=2, keepdims=True)


def fold_probes(attention, probes, page_tokens, max_pages):
    held = zipcache.ProbedMass(len(attention), max_pages, page_tokens)
    for q in probes:
        pages = -(-(q + 1) // page_tokens)
        with held.at(q):  # one step, every layer's masses
            for layer, rows in enumerate(attention):
                padded = np.zeros((rows.shape[0], pages * page_tokens))
                padded[:, :q + 1] = rows[:, q, :q + 1]
                mass = padded.reshape(rows.shape[0], pages, page_tokens).sum(axis=2)
                held.fold(layer, Mass(mass), rows.shape[0], pages)
    return held.scores()


def equation_8(rows, probes, n):
    """A token's attention over the probe queries, by how many of them are
    at its position or past it, the heads' mean: ZipCache's equation 8."""
    a = rows.mean(axis=0)[probes]
    return a.sum(axis=0) / np.array([(probes >= i).sum() for i in range(n)])


def test_a_page_every_probe_can_attend_to_is_its_tokens_equation_8():
    """Where every probe is past a page, each of its tokens has the same
    divisor, and the page's score is its tokens' mean normalised attention."""
    n, page = 24, 4
    attention = [causal_attention(3, n, seed) for seed in (1, 2)]
    probes = np.array([17, 19, 20, 22, 23])
    got = fold_probes(attention, probes, page, max_pages=8)
    for layer, rows in enumerate(attention):
        tokens = equation_8(rows, probes, n)
        for p in range(4):  # positions 0 to 15
            assert got[layer, p] == pytest.approx(tokens[p * page:(p + 1) * page].mean(),
                                                  rel=1e-6)
    assert np.isnan(got[:, 6:]).all()


def test_a_page_s_divisor_counts_only_the_queries_and_tokens_that_meet():
    """A probe attends to itself and what is before it: a page it is inside
    of counts its tokens up to it, and a page past it none."""
    attention = [np.tri(8)[None] / np.arange(1, 9)[:, None]]  # each row even over what it sees
    got = fold_probes(attention, np.array([1, 5]), 4, max_pages=3)
    # page 0: 1 from query 1 over 2 tokens, and 4/6 from query 5 over 4
    assert got[0, 0] == pytest.approx((1 + 4 / 6) / 6)
    # page 1: 2/6 from query 5 over 2 tokens
    assert got[0, 1] == pytest.approx((2 / 6) / 2)
    assert np.isnan(got[0, 2])


def test_normalising_takes_off_what_accumulating_gives_early_positions():
    """Attention even over what each query sees: accumulated, the first page
    is far above the last; normalised, every page every probe passed is
    alike, and accumulated again it is not."""
    n, page = 64, 4
    even = [np.tri(n)[None] / np.arange(1, n + 1)[:, None]]
    probes = np.arange(n)
    got = fold_probes(even, probes, page, max_pages=n // page)[0]
    accumulated = np.array([even[0][0, probes, p * page:(p + 1) * page].sum()
                            for p in range(n // page)])
    assert accumulated[0] / accumulated[-1] > 4 * got[0] / got[-1]


def test_masses_outside_a_probe_are_not_taken():
    """A chunk of one position between two probes gives masses too; they
    are no probe's."""
    held = zipcache.ProbedMass(1, 2, 4)
    held.fold(0, Mass([[0.5, 0.5]]), 1, 2)
    assert np.isnan(held.scores()).all()


# -- the pass on the engine --------------------------------------------------------------


@pytest.fixture(scope="module")
def engine():
    e = Engine(require_model(NAME), prefill_chunk=128)
    e.load_weights()
    return e


@pytest.mark.slow
def test_the_pass_gives_what_masses_taken_at_every_position_give(engine):
    """Forced a position at a time, every position's masses are taken; the
    probes' among them, normalised here, are the pass's scores, whose
    prefill between the probes is in chunks, to what chunking rounds."""
    ids, seed = prompt(150, 1), 5
    probes = set(zipcache.probe_positions(len(ids), seed).tolist())
    got = engine.probed_scores(ids, seed=seed)
    cfg = engine.config
    cache = model.PagedCache(cfg, scoring=True)
    taken = []

    class Every:
        nbytes = 0

        def fold(self, layer, mass, heads, pages):
            taken.append((layer, pages,
                          mass.to_numpy()[:heads * pages].reshape(heads, pages).mean(axis=0)))

    cache.scores = Every()
    ws = model.Workspace(cfg, rows=1)
    sums = np.zeros((cfg.num_hidden_layers, -(-len(ids) // P)))
    pairs = np.zeros(sums.shape[1])
    with engine._holding(cache, ws):
        for q in range(len(ids)):
            taken.clear()
            engine._model.run(ws, cache, ids[q:q + 1])
            if q in probes:
                for layer, pages, mass in taken:
                    sums[layer, :pages] += mass
                pairs += np.clip(q + 1 - np.arange(len(pairs)) * P, 0, P)
    want = sums / pairs
    assert got.shape == (cfg.num_hidden_layers, -(-cfg.context_window // P))
    np.testing.assert_allclose(got[:, :len(pairs)], want, rtol=0.01)  # 0.0018 at most, as run
    assert np.isnan(got[:, len(pairs):]).all()


@pytest.mark.slow
def test_a_controlled_plan_takes_the_source_and_leaves_the_cache_as_it_was(engine):
    """The pass runs on a cache of its own: the scorer's scores and the
    cache's length are untouched, and what the conversation goes on to
    decode is what it decodes with no plan's pass made. The tokens a pass
    reads are those the cache holds."""
    with rq3.set_for(engine, rq3.NO_PLAN):
        def turns(probing):
            with engine.conversation() as chat:
                first = chat.turn(prompt(300, 2), 6, stop_at_eos=False)
                assert chat.tokens.tolist() == [*prompt(300, 2), *first[:-1]]
                before = chat.scores()
                if probing:
                    chat.probed_scores(seed=1)
                    np.testing.assert_array_equal(chat.scores(), before)
                assert chat.positions == 305
                return chat.turn(prompt(10, 3), 8, stop_at_eos=False).tolist()

        assert turns(True) == turns(False)


@pytest.mark.slow
def test_the_source_plans_at_the_same_bytes_as_the_others(engine):
    """ADR-0016's planner, at 70% and 80%, from the pass's scores: the
    target met, over it by less than one page's move, as the others meet it;
    and its scores are not theirs, so its plan is not either."""
    page = dict(zip(controller.TIERS, model.tier_page_bytes(engine.config)))
    largest = max(page[a] - page[b] for a, b in zip(controller.TIERS, controller.TIERS[1:]))
    with rq3.set_for(engine, rq3.NO_PLAN):
        for share in (0.7, 0.8):
            reclaimed, tiers = {}, {}
            for source in ("semantic", "uniform", "zipcache"):
                with engine.conversation() as chat:
                    chat.turn(prompt(1024, 4), 12, stop_at_eos=False)
                    c = chat.plan(share, source=source, seed=7).controlled
                    reclaimed[source], tiers[source] = c["reclaimed_bytes"], chat.tiers()
                assert c["source"] == source and c["target_met"]
                assert c["target_bytes"] <= c["reclaimed_bytes"] < c["target_bytes"] + largest
            assert max(reclaimed.values()) - min(reclaimed.values()) < largest
            assert not np.array_equal(tiers["zipcache"], tiers["semantic"])


@pytest.mark.slow
def test_a_conversation_from_a_snapshot_probes_the_tokens_it_was_taken_with(engine):
    """A snapshot's conversation holds the tokens of the one it was taken
    from: the same scores, the same plan (#172)."""
    with rq3.set_for(engine, rq3.NO_PLAN):
        with engine.conversation(for_snapshot=True) as chat:
            chat.turn(prompt(300, 5), 6, stop_at_eos=False)
            tokens, scores = chat.tokens, chat.probed_scores(seed=2)
            snapshot = chat.snapshot()
        with engine.conversation(start=snapshot) as chat:
            np.testing.assert_array_equal(chat.tokens, tokens)
            np.testing.assert_array_equal(chat.probed_scores(seed=2), scores)
            chat.plan(0.8, source="zipcache", seed=2)
            chat.turn(prompt(5, 6), 3, stop_at_eos=False)
            assert len(chat.tokens) == chat.positions


def test_forced_tokens_are_the_conversation_s_too(engine):
    with rq3.set_for(engine, rq3.NO_PLAN), engine.conversation() as chat:
        chat.force(prompt(40, 7))
        chat.force(prompt(3, 8), one_by_one=True)
        assert chat.tokens.tolist() == [*prompt(40, 7), *prompt(3, 8)]


def test_an_engine_under_pressure_cannot_plan_from_it():
    """Its scores take a pass over everything the cache holds: an engine's
    own plans, made under pressure, read scores as they stand."""
    with pytest.raises(ValueError, match="zipcache"):
        Engine(require_model(NAME), kv_score_source="zipcache")
