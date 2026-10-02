"""Downgrading a sealed FP16 page at runtime, end to end (#93, #95, Seam A).

KVPages.downgrade moves a page of positions sealed at FP16 to a lower tier
mid-session: allocated at the target tier, quantised from FP16 (the page
itself while it is FP16, its shadow once below), put in its place in the
page table, the old page freed. Seam B
(test_quantised_pages.py) proves the page's bytes and attention over it.
What is left here is the model and the driver: a session that downgrades
its pages as it goes reads, to the bit, what a session over a cache built at
those tiers reads; and what the driver holds afterwards is what
tier_page_bytes predicts.
"""

import numpy as np
import pytest
from test_kv_pages import put
from test_static_tiers import MODEL, P, REPO

from conftest import each, require_model
from microinfer import Engine, ModelConfig, _microinfer, model, nvml, paged_cache_bytes
from microinfer.engine import tier_map_of
from microinfer.golden import GoldenError, GoldenSet

Tier = _microinfer.Tier
LADDER = (Tier.INT8, Tier.INT4, Tier.INT2)


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


def test_after_runtime_downgrades_the_logits_are_a_cache_built_at_those_tiers(engine, golden):
    """Two caches holding one prompt's keys and values. One holds every page
    at FP16, and seals (always_seal); after the prefill, and after every
    decode step that seals a page, it downgrades each page it holds to the
    page's final tier, page 0 staying FP16 and the others going down the
    tiers in turn: in one step, or, chained, through INT8 first and from
    there on down from the page's shadow (#95). The other is born at
    those final tiers, from the same rows: the prefill's keys and values,
    which the first holds, to the bit, before it downgrades anything. From
    the first decode step on, every step's logits are the same, to the bit,
    and so is every token. (Prefilling the second through the model instead
    would not do: its later layers would read the earlier pages quantised,
    and so store other rows.)"""
    cfg = engine.config
    m = engine._model
    layers, kv_heads, hd = cfg.num_hidden_layers, cfg.num_key_value_heads, cfg.head_dim

    def final_tier(layer, i):
        return Tier.FP16 if i == 0 else LADDER[(layer + i) % 3]

    def session(prompt_id, chained):
        ids = golden[prompt_id].token_ids
        new = 2 * P + 5
        pages = (len(ids) + new) // P
        final = [[final_tier(layer, i) for i in range(pages)] for layer in range(layers)]
        moving = model.PagedCache(cfg, Tier.FP16, always_seal=True)
        built = model.PagedCache(cfg, Tier.FP16, tier_map=final)

        prefill = model.Workspace(cfg, rows=len(ids))
        m.run(prefill, moving, ids)
        token = m.greedy_last(prefill, len(ids))
        keys, values = engine.cached_kv(ids)
        built.reserve(len(ids))
        for layer in range(layers):
            built.store(layer, put(keys[layer]), put(values[layer]), 0, len(ids))
        built.length = len(ids)
        for layer in range(layers):
            for i in range(moving.pages.pages_per_layer):
                span = slice(i * P, (i + 1) * P)
                np.testing.assert_array_equal(
                    moving.allocator.read(layer, i).view(np.float16).reshape(2, P, kv_heads, hd),
                    np.stack([keys[layer][span], values[layer][span]]).astype(np.float16),
                    err_msg=f"the prefill stored other rows: layer {layer} page {i}")

        moved = 0

        def downgrade_sealed():
            nonlocal moved
            for i in range(moved, moving.pages.pages_per_layer):
                for layer in range(layers):
                    target = final[layer][i]
                    if chained and int(target) > int(Tier.INT8):
                        moving.pages.downgrade(layer, i, Tier.INT8)
                    if target != Tier.FP16:
                        moving.pages.downgrade(layer, i, target)
            moved = moving.pages.pages_per_layer

        downgrade_sealed()
        step = model.Workspace(cfg, rows=1)
        for t in range(new):
            logits = []
            for cache in (built, moving):
                m.run(step, cache, np.array([token], np.int32))
                logits.append(m.logits(step, 1))
            np.testing.assert_array_equal(logits[1], logits[0], err_msg=f"step {t}")
            downgrade_sealed()
            token = int(logits[0][-1].argmax())
        assert moved == pages
        for layer in range(layers):
            for i in range(pages):
                assert moving.pages.page_tier(layer, i) == final[layer][i], (layer, i)

    each([(p, c) for p in ("medium-01", "long-03") for c in (False, True)], session,
         name=lambda c: f"{c[0]}{' chained' if c[1] else ''}")


def test_the_driver_sees_the_memory_tier_page_bytes_predicts_returned():
    """Qwen2.5-1.5B with 4096 positions held, every page at FP16 but each
    layer's first, at INT8. In three rounds, a third of the FP16 pages each
    go to INT4, to INT2 and to INT8, layer by layer. After each round, what this process
    holds by the driver's own account has fallen by what paged_cache_bytes
    computes from tier_page_bytes for the tiers the pages are now at, to a
    granule. The prediction is checked against the allocator too, to the
    byte."""
    cfg = ModelConfig.from_card("qwen2.5-1.5b-instruct")
    layers, kv_width = cfg.num_hidden_layers, cfg.num_key_value_heads * cfg.head_dim
    positions = 4096
    pages = positions // P
    names = [["INT8"] + ["FP16"] * (pages - 1) for _ in range(layers)]
    cache = model.PagedCache(cfg, Tier.FP16, tier_map=tier_map_of(names))
    cache.rope.cover(positions)
    rows = _microinfer.upload_fp16(
        np.random.default_rng(93).standard_normal(positions * kv_width).astype(np.float32))
    cache.reserve(positions)
    for layer in range(layers):
        cache.store(layer, rows, rows, 0, positions)
    granule = _microinfer.granule_bytes()

    def held():
        return sum(cache.allocator.mapped_bytes(t) for t in Tier.__members__.values())

    rounds = [{i: "INT4" for i in range(1, pages, 3)},
              {i: "INT2" for i in range(2, pages, 3)},
              {i: "INT8" for i in range(3, pages, 3)}]
    for moves in rounds:
        before = nvml.settled_own_used_bytes()
        predicted = paged_cache_bytes(cfg, positions, "FP16", names)
        for layer in range(layers):
            for i, target in moves.items():
                if names[layer][i] == "FP16":
                    cache.pages.downgrade(layer, i, getattr(Tier, target))
                    names[layer][i] = target
        returned = predicted - paged_cache_bytes(cfg, positions, "FP16", names)
        measured = before - nvml.settled_own_used_bytes()
        print(f"\npredicted {returned / 2**20:.1f} MiB returned, measured {measured / 2**20:.1f}")
        assert held() == paged_cache_bytes(cfg, positions, "FP16", names)
        assert returned > 0
        assert abs(measured - returned) <= granule
