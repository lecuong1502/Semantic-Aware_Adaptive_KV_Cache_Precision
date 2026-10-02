"""The engine over a cache built from a tier map (#92, Seam A).

A session's cache can be built from a tier map: the tier each page of
positions is born at, by (layer, page), the cache's own tier beyond a row
(#90, ADR-0011 and its amendments). Attention reads each page at its own tier
(#91), proven at Seam B in test_quantised_pages.py. What is left to show here
is that the engine reaches such a cache, prefill and decode alike, and that
ADR-0011's property holds over it: a prompt decoded one token at a time and
the same tokens prefilled at once read the same cache, and choose the same
tokens. And a map with one tier everywhere is the static engine.

One test reaches below the engine's surface: to read the tier of each page a
session's cache holds, it takes the cache from engine._new_cache, as
test_paged_engine.py's decode takes engine._model.
"""

import numpy as np
import pytest
from test_chunked_prefill import LOGIT_BOUND
from test_paged_engine import decode
from test_static_tiers import MODEL, P, REPO, TIERS, at_tier

from conftest import each, require_model
from microinfer import Engine, _microinfer
from microinfer.golden import GoldenError, GoldenSet

PROMPTS = ("short-00", "medium-01", "adversarial-04", "long-03")


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


def mixed_map(layers, positions):
    """The four tiers in turn, in a different order per layer, over the pages
    of `positions`: every tier in every layer once it has four pages."""
    order = ("INT8", "FP16", "INT2", "INT4")
    pages = positions // P + 1
    return [[order[(layer + i) % 4] for i in range(pages)] for layer in range(layers)]


def with_map(engine, tier_map, tier="FP16"):
    """The engine with its caches at `tier` and built from `tier_map` for the
    duration of a with-block."""
    class Switch:
        def __enter__(self):
            self.tier = at_tier(engine, tier)
            self.tier.__enter__()
            self.before, engine.kv_tier_map = engine.kv_tier_map, tier_map
            return engine

        def __exit__(self, *exc):
            engine.kv_tier_map = self.before
            self.tier.__exit__(*exc)

    return Switch()


def test_a_session_runs_over_the_cache_its_map_builds(engine, golden):
    """The cache a session makes is born at the map's tiers: after a real
    prefill and decode, every page of positions is at the tier the map
    names for it, or the cache's beyond a row, so every tier is read."""
    ids = golden["medium-01"].token_ids
    new = 2 * P + 3
    layers = engine.config.num_hidden_layers
    tier_map = [row[:-1] for row in mixed_map(layers, len(ids) + new)]  # the last page: INT4
    with with_map(engine, tier_map, "INT4"):
        cache = engine._new_cache(capacity=len(ids) + new)
    decode(engine, ids, new, cache)
    pages = cache.pages
    assert pages.pages_per_layer == (len(ids) + new - 1) // P
    seen = set()
    for layer in range(layers):
        for i in range(pages.pages_per_layer):
            want = tier_map[layer][i] if i < len(tier_map[layer]) else "INT4"
            assert pages.page_tier(layer, i) == getattr(_microinfer.Tier, want), (layer, i)
            seen.add(want)
    assert seen == set(TIERS)


def test_decoding_continues_as_a_full_prefill_would_over_a_mixed_map(engine, golden):
    """ADR-0011's property, with a tier map: a query reads its own page at
    FP16 and every page before it at that page's own tier, however many
    positions came with it. So a prompt decoded one token at a time and the
    same tokens prefilled at once read the same cache, and choose the same
    tokens, up to where cuBLAS rounds a key differently in a one-row GEMM than
    in a long one: the first difference, if any, is at a near-tie of the
    prefill's top two logits, as test_static_tiers.py has it. Over caches at
    FP16 and at INT4, so that a page beyond a row is at either; and with
    prefill in one chunk and in chunks of 40, which seal pages mid-chunk."""
    layers = engine.config.num_hidden_layers

    def continues(tier, chunk, prompt_id):
        ids = golden[prompt_id].token_ids
        new = 40
        tier_map = mixed_map(layers, (len(ids) + new) // 2)  # half the pages beyond the map
        before, engine.prefill_chunk = engine.prefill_chunk, chunk
        try:
            with with_map(engine, tier_map, tier):
                out = engine.generate(ids, new, stop_at_eos=False)
                full = engine.forward(np.concatenate([ids, out[:-1]]))[len(ids) - 1:]
        finally:
            engine.prefill_chunk = before
        differ = np.flatnonzero(full.argmax(-1) != out)
        if differ.size:
            top_two = np.sort(full[differ[0]])[-2:]
            assert top_two[1] - top_two[0] < LOGIT_BOUND, (
                f"decode left the prefill's choice at step {differ[0]}, away from a near-tie")

    each([(t, c, p) for t in ("FP16", "INT4") for c in (None, 40) for p in PROMPTS], continues)


def test_a_map_of_one_tier_is_the_static_engine(engine, golden):
    """At every tier, a map naming that tier on every page chooses the static
    engine's tokens, and gives its logits, to the bit: over a cache at that
    tier, where the map changes nothing, and over a cache at FP16, where the
    map alone makes the cache seal and places every page."""
    layers = engine.config.num_hidden_layers

    def static(tier, cache_tier, prompt_id):
        ids = golden[prompt_id].token_ids
        new = 24
        tier_map = [[tier] * ((len(ids) + new) // P + 1) for _ in range(layers)]
        with at_tier(engine, tier):
            want = engine.generate(ids, new, stop_at_eos=False)
            want_logits = engine.forward(ids)
        with with_map(engine, tier_map, cache_tier):
            got = engine.generate(ids, new, stop_at_eos=False)
            got_logits = engine.forward(ids)
        np.testing.assert_array_equal(got, want)
        np.testing.assert_array_equal(got_logits, want_logits)

    each([(t, c, p) for t in TIERS for c in {t, "FP16"} for p in ("short-00", "long-03")],
         static, name=lambda c: f"{c[0]} map over a {c[1]} cache, {c[2]}")


def test_the_map_is_chosen_by_configuration():
    """A map is the engine's kv_tier_map, of tier names, a row per layer,
    checked as kv_tier is. The contiguous cache holds FP16 only and takes
    none, a diagnostic halves takes none, and a hold, which rewrites
    positions in place, takes none that seals a page."""
    path = require_model(MODEL)
    e = Engine(path)
    layers = e.config.num_hidden_layers
    assert e.kv_tier_map is None
    row = ["INT8", "FP16"]
    assert Engine(path, kv_tier_map=[row] * layers).kv_tier_map == [row] * layers
    with pytest.raises(ValueError, match="layer"):
        Engine(path, kv_tier_map=[row] * (layers - 1))
    with pytest.raises(ValueError, match="kv_tier_map"):
        Engine(path, kv_tier_map=[["INT3"]] * layers)
    with pytest.raises(ValueError, match="contiguous"):
        Engine(path, kv_cache="contiguous", kv_tier_map=[row] * layers)
    with pytest.raises(ValueError, match="diagnostic"):
        Engine(path, kv_tier="INT4", kv_halves="keys", kv_tier_map=[row] * layers)
    mapped = Engine(path, kv_tier="INT4", kv_tier_map=[row] * layers)
    with pytest.raises(ValueError, match="diagnostic"):
        mapped.kv_halves = "keys"
    held = Engine(path, kv_tier_map=[row] * layers)
    with pytest.raises(ValueError, match="in place"):
        held.hold([1, 2, 3], context=8, stop=None)
