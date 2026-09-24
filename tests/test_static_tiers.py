"""Static per-tier operation end to end (#18, Seam A).

What Milestone 0 exists to deliver: the engine runs with its cache held
entirely at FP16, INT8, INT4 or INT2, chosen by configuration, with no
adaptivity anywhere. A page's tier is chosen when it is allocated and never
changes. What each tier costs is measured, not assumed: here as agreement with
the golden reference per tier, and as the memory each tier takes from the
driver; as perplexity on held-out text by tools/perplexity.py, which logs it.

The paged cache at a quantised tier is proven at Seam B to compute attention
over exactly what quantisation keeps (test_quantised_pages.py). What is left
to show here is that the engine reaches it, that each narrower tier costs
more than the one above, and that its memory is what the layout says.
"""

import os
from pathlib import Path

import numpy as np
import pytest
from test_paged_engine import decode

from conftest import require_model
from microinfer import Engine, ModelConfig, _microinfer, model, nvml, paged_cache_bytes
from microinfer.gate import KL_MAX, TOP1_MIN, run_gate
from microinfer.golden import GoldenError, GoldenSet

REPO = Path(__file__).resolve().parent.parent
MODEL = "qwen2.5-0.5b-instruct"
TIERS = Engine.KV_TIERS
P = _microinfer.device.page_tokens
MIB = 2**20


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


def at_tier(engine, tier):
    """The engine with its caches at `tier` for the duration of a with-block:
    one set of weights serves every tier."""
    class Switch:
        def __enter__(self):
            self.before, engine.kv_tier = engine.kv_tier, tier
            return engine

        def __exit__(self, *exc):
            engine.kv_tier = self.before

    return Switch()


def test_the_tier_is_chosen_by_configuration():
    path = require_model(MODEL)
    assert Engine(path).kv_tier == "FP16"
    assert Engine(path, kv_tier="INT4").kv_tier == "INT4"
    with pytest.raises(ValueError, match="kv_tier"):
        Engine(path, kv_tier="INT3")
    with pytest.raises(ValueError, match="contiguous"):
        Engine(path, kv_cache="contiguous", kv_tier="INT8")


def test_every_tier_runs_and_each_narrower_one_costs_more(engine, golden):
    """The gate, run at every tier. FP16 passes it, as ADR-0006 requires. The
    others are not held to it, since quantisation is meant to cost something;
    what must hold is that it costs more the narrower the tier: KL against the
    reference rises strictly from FP16 through INT8 and INT4 to INT2, and top-1
    agreement never rises."""
    reports = {}
    for tier in TIERS:
        with at_tier(engine, tier):
            reports[tier] = run_gate(engine, golden)
    print("\n" + "\n".join(f"  {t:<5} top-1 {r.top1:.4f}  mean KL {r.kl_mean:.3e}"
                           for t, r in reports.items()))
    fp16 = reports["FP16"]
    assert fp16.top1 >= TOP1_MIN and fp16.kl_mean < KL_MAX, fp16.render()
    kl = [reports[t].kl_mean for t in TIERS]
    top1 = [reports[t].top1 for t in TIERS]
    assert all(a < b for a, b in zip(kl, kl[1:])), kl
    assert all(a >= b for a, b in zip(top1, top1[1:])), top1


@pytest.mark.parametrize("tier", TIERS[1:])
def test_no_page_changes_tier_during_a_generation(engine, golden, tier):
    """A real prefill and decode at a quantised tier, with the allocator read
    after every step: every page of positions is at the cache's tier, the only
    FP16 pages are the open pages, one per layer, and no page once seen is ever
    at another tier."""
    cfg = engine.config
    cache = model.PagedCache(cfg, getattr(_microinfer.Tier, tier))
    ids = golden["medium-02"].token_ids
    seen = {}

    def check(step):
        allocator = cache.allocator
        now = {key: t for t in _microinfer.Tier.__members__.values() for key in allocator.pages(t)}
        for key, was in seen.items():
            assert now.get(key) == was, f"{key} was at {was}, now {now.get(key)} (step {step})"
        seen.update(now)
        full = cache.length // P
        assert len(allocator.pages(cache.tier)) == cfg.num_hidden_layers * full
        assert sorted(allocator.pages(_microinfer.Tier.FP16)) == [
            (layer, _microinfer.device.open_page) for layer in range(cfg.num_hidden_layers)]

    decode(engine, ids, 3 * P, cache, between_steps=check)
    check("end")


@pytest.mark.parametrize("tier", TIERS)
def test_the_footprint_at_each_tier_is_what_the_layout_computes(tier):
    """Qwen2.5-1.5B's whole 32K window at each tier, allocated but not run: what
    this process holds by the driver's own account, which no other process
    moves, against paged_cache_bytes, within 1%. The RoPE table is covered
    before the reading, so the pages alone are measured."""
    cfg = ModelConfig.from_card("qwen2.5-1.5b-instruct")
    tokens = cfg.max_position_embeddings
    cache = model.PagedCache(cfg, getattr(_microinfer.Tier, tier))
    cache.rope.cover(tokens)
    before = nvml.own_used_bytes()
    cache.pages.reserve(tokens)
    measured = nvml.own_used_bytes() - before
    computed = paged_cache_bytes(cfg, tokens, tier)
    print(f"\n{tier}: measured {measured / MIB:.1f} MiB, computed {computed / MIB:.1f} MiB")
    assert abs(measured - computed) <= 0.01 * computed
    del cache


@pytest.mark.skipif(os.environ.get("MICROINFER_LONG_TESTS") != "1",
                    reason="tens of minutes; set MICROINFER_LONG_TESTS=1 to run")
@pytest.mark.parametrize("tier", TIERS[1:])
def test_a_32k_prompt_prefills_at_each_quantised_tier_on_the_1_5b_model(tier):
    """ADR-0003's experimental configuration at each quantised tier: the whole
    context window of Qwen2.5-1.5B, prefilled in chunks into a cache at the
    tier, then one token chosen. FP16 is test_chunked_prefill.py's. At the
    peak the cache holds what paged_cache_bytes says for the window, and
    beside it the RoPE table, which covers every position."""
    e = Engine(require_model("qwen2.5-1.5b-instruct"), kv_tier=tier)
    e.load_weights()
    window = e.config.max_position_embeddings
    ids = np.random.default_rng(15).integers(1000, 100_000, window).astype(np.int32)
    e.reset_peak()
    out = e.generate(ids, 1, stop_at_eos=False)
    assert out.shape == (1,) and 0 <= out[0] < e.config.vocab_size
    peak = e.peak_footprint()
    print(f"\n{tier}: " + peak.render())
    table = peak.kv_cache - paged_cache_bytes(e.config, window, tier)
    per_position = e.config.head_dim * 4  # {cos, sin} in fp32 for head_dim / 2 frequencies
    assert table % per_position == 0 and table // per_position >= window, table
