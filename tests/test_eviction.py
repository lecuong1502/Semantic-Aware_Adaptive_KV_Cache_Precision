"""Eviction at equal bytes, the H2O-style baseline (#160).

An evicted page is freed, its memory back to the driver, and no query
attends to its positions again. The proof is equality, not tolerance:
attention over a cache with pages evicted is, to the bit, attention over
the same cache with those pages masked, kept and read by no query; and a
masked page is masked, attention near what it is over the keys without
its positions.
"""

from dataclasses import asdict

import numpy as np
import pytest
from conftest import load_tool, require_model
from test_kv_pages import CFG, LAYERS, PAGE_TOKENS, normal, put
from test_quantised_pages import attend, sealed, sealing_at_fp16, store_in_runs

from microinfer import Engine, _microinfer, benchlog, controller, needles, rq3, rq3_summary
from microinfer._microinfer import PagedKVCache, Tier
from microinfer.model import tier_page_bytes

device = _microinfer.device


def stored(page_tokens, seq, seed=160):
    """Two caches at FP16 that seal, holding the same keys and values in
    every layer, and those keys and values."""
    rng = np.random.default_rng(seed)
    kv_heads, hd = CFG.num_key_value_heads, CFG.head_dim
    k = [normal(rng, seq, kv_heads, hd) for _ in range(LAYERS)]
    v = [normal(rng, seq, kv_heads, hd) for _ in range(LAYERS)]
    caches = []
    for _ in range(2):
        allocator, cache = sealing_at_fp16(page_tokens)
        runs = [page_tokens + 5, seq - page_tokens - 5]
        for layer in range(LAYERS):
            start = store_in_runs(cache, layer, k[layer], v[layer], runs)
        caches.append((allocator, cache))
    return caches, k, v, start


def test_attention_over_evicted_pages_is_attention_with_them_masked_to_the_bit():
    """Pages evicted from one cache and masked in its twin, page 0 among
    them, at a quantised tier and at FP16: every query's output is the
    same to the bit, with and without the key bias, and for a decode
    step's per-page mass too, which gives the pages none. The masked
    cache's output is near attention over the keys without the pages'
    positions, and far from the cache's with nothing masked."""
    heads, kv_heads, hd = CFG.num_attention_heads, CFG.num_key_value_heads, CFG.head_dim

    def evicted(page_tokens):
        seq = 5 * page_tokens + 3
        (held, evicting), k, v, start = stored(page_tokens, seq)
        whole = attend(CFG, held[1], 0, normal(np.random.default_rng(1), seq - start, heads,
                                                 hd), k[0], v[0], None, start)
        evicting[1].downgrade(0, 2, Tier.INT4)
        held[1].downgrade(0, 2, Tier.INT4)
        for page in (0, 2, 3):
            evicting[1].evict(0, page)
            held[1].mask(0, page)
        evicting[1].evict(1, 1)
        held[1].mask(1, 1)
        rng = np.random.default_rng(2)
        for layer in range(LAYERS):
            for bias in (None, normal(rng, kv_heads, hd) * 30):
                q = normal(rng, seq - start, heads, hd)
                got = attend(CFG, evicting[1], layer, q, k[layer], v[layer], bias, start)
                masked = attend(CFG, held[1], layer, q, k[layer], v[layer], bias, start)
                np.testing.assert_array_equal(got, masked, err_msg=f"layer {layer}")

        # Masked is masked: the queries read the keys whose pages are read.
        q = normal(np.random.default_rng(1), seq - start, heads, hd)
        masked = attend(CFG, held[1], 0, q, k[0], v[0], None, start)
        assert not np.allclose(masked, whole, atol=1e-2)
        kept = np.ones(seq, bool)
        for page in (0, 2, 3):
            kept[page * page_tokens:(page + 1) * page_tokens] = False
        reference = without(held, k[0], v[0], q, kept, start, page_tokens)
        np.testing.assert_allclose(masked, reference, atol=2e-3)

        # A decode step's per-page mass gives an unread page none.
        one = normal(rng, 1, heads, hd)
        q_row, k_row, v_row = one, normal(rng, 1, kv_heads, hd), normal(rng, 1, kv_heads, hd)
        masses = []
        for _, cache in (evicting, held):
            cache.reserve(seq + 1)
            cache.store(0, put(k_row), put(v_row), seq, 1)
            mass = device.empty_f32(heads * (seq // page_tokens + 1))
            out = device.empty(heads * hd)
            rope = device.RopeTable(hd, CFG.rope_theta)
            rope.cover(seq + 1)
            cache.attention(0, put(q_row), None, rope, out, 1, seq + 1, heads, kv_heads, hd,
                            put(k_row), put(v_row), mass)
            masses.append((out.to_numpy(), mass.to_numpy().reshape(heads, -1)))
        np.testing.assert_array_equal(masses[0][0], masses[1][0])
        np.testing.assert_array_equal(masses[0][1], masses[1][1])
        assert (masses[0][1][:, [0, 2, 3]] == 0).all()
        np.testing.assert_allclose(masses[0][1].sum(axis=1), 1.0, rtol=1e-5)

    for page_tokens in PAGE_TOKENS:
        evicted(page_tokens)


def without(pair, k, v, q, kept, start, page_tokens):
    """Each query alone over the keys it may read, those of the pages before
    its own that `kept` marks, and its own page's, from a cache holding
    them all: the masked cache's own reference. A query's own page is
    always read."""
    allocator, _ = pair
    heads, kv_heads, hd = CFG.num_attention_heads, CFG.num_key_value_heads, CFG.head_dim
    sealed_k, sealed_v = sealed(allocator, CFG, 0, len(k) // page_tokens, page_tokens)
    rope = device.RopeTable(hd, CFG.rope_theta)
    rope.cover(len(k))
    rows = []
    for i, pos in enumerate(range(start, len(k))):
        own = pos // page_tokens * page_tokens
        read = kept[:pos + 1] | (np.arange(pos + 1) >= own)
        kk = np.concatenate([sealed_k[:own], k[own:pos + 1]])[read]
        vv = np.concatenate([sealed_v[:own], v[own:pos + 1]])[read]
        out = device.empty(heads * hd)
        device.attention(put(q[i:i + 1]), put(kk), put(vv), None, rope, out, 1, len(kk),
                         heads, kv_heads, hd)
        rows.append(out.to_numpy())
    return np.concatenate(rows)


def test_an_evicted_pages_memory_returns_to_the_driver():
    """Evicting pages that fill granules of an FP16 range gives the
    granules back: the range maps all but what the pages left need, the
    evicted pages are no longer in the page table, and the rest hold what
    they held."""
    page_tokens = PAGE_TOKENS[0]
    seq = 640 * page_tokens
    (pair, _), k, v, _ = stored(page_tokens, seq)
    allocator, cache = pair
    before = allocator.mapped_bytes(Tier.FP16)
    held = {key: allocator.read(*key) for key in allocator.pages(Tier.FP16)}
    gone = [(layer, page) for layer in range(LAYERS) for page in range(400)]
    for layer, page in gone:
        cache.evict(layer, page)
    granule = allocator.granule_bytes
    left = len(allocator.pages(Tier.FP16)) * allocator.page_bytes(Tier.FP16)
    assert allocator.mapped_bytes(Tier.FP16) == -(-left // granule) * granule < before
    assert not set(gone) & set(allocator.pages(Tier.FP16))
    for key in allocator.pages(Tier.FP16):
        np.testing.assert_array_equal(allocator.read(*key), held[key])
    assert cache.evicted()[:, :400].all() and not cache.evicted()[:, 400:].any()
    tiers = cache.page_tiers()
    assert (tiers[:, :400] == 4).all() and (tiers[:, 400:] == int(Tier.FP16)).all()


def test_what_an_eviction_refuses():
    """A page not sealed, one evicted already, or masked, is not evicted;
    an evicted page neither moves nor is masked; and a cache that does not
    seal evicts nothing."""
    page_tokens = PAGE_TOKENS[0]
    seq = 3 * page_tokens + 5
    ((_, cache), (_, twin)), _, _, _ = stored(page_tokens, seq)
    cache.evict(0, 1)
    twin.mask(0, 1)
    for call in (lambda: cache.evict(0, 3), lambda: cache.evict(0, 1),
                 lambda: cache.mask(0, 1), lambda: cache.downgrade(0, 1, Tier.INT4),
                 lambda: cache.upgrade(0, 1, Tier.FP16), lambda: twin.evict(0, 1),
                 lambda: cache.evict(LAYERS, 0)):
        with pytest.raises((ValueError, IndexError)):
            call()
    with pytest.raises(Exception, match="not in the page table"):
        cache.page_tier(0, 1)
    twin.downgrade(0, 1, Tier.INT4)  # a masked page still moves
    capacity = [0] * 4
    capacity[int(Tier.FP16)] = 64
    allocator = PagedKVCache(tier_page_bytes(CFG, page_tokens), capacity)
    plain = device.KVPages(allocator, LAYERS, page_tokens, CFG.num_key_value_heads,
                           CFG.head_dim, Tier.FP16)
    plain.reserve(2 * page_tokens)
    with pytest.raises(ValueError, match="seals"):
        plain.evict(0, 0)


# -- the controller: which pages, for which bytes --------------------------------------


PAGE_BYTES = {"FP16": 1000, "INT8": 550, "INT4": 300, "INT2": 170}
ERRORS = {"FP16": 0.0, "INT8": 0.001, "INT4": 0.01, "INT2": 0.1}
MOVES = {(a, b): 1e-5 for a in controller.TIERS for b in controller.TIERS if a != b}


def test_an_eviction_takes_the_pages_the_plan_downgrades_most_for_its_bytes():
    """The plan's pages, those it leaves lowest first, and at one tier in
    the order it took them there; until the bytes they hold meet the
    target, the last one meeting it. Pages a plan never moves are not
    taken, and a target they cannot meet is short."""
    tiers = np.zeros((2, 6), dtype=np.int8)
    scores = np.array([[0.1, 5.0, 0.2, 3.0, 9.0, 9.0], [0.3, 4.0, 2.0, 0.05, 9.0, 9.0]])
    plan = controller.target_plan_arrays(3000, controller.RED, scores, tiers, PAGE_BYTES,
                                         ERRORS, move_seconds=MOVES, positions=6 * 32,
                                         page_tokens=32, recency_floor=64)
    lowest = {}
    for i in range(len(plan)):
        lowest[(int(plan.layers[i]), int(plan.pages[i]))] = (int(plan.lower[i]), i)
    want = sorted(lowest, key=lambda p: (-lowest[p][0], lowest[p][1]))
    eviction = controller.eviction_plan(2500, plan, tiers, PAGE_BYTES)
    assert list(zip(eviction.layers.tolist(), eviction.pages.tolist())) == want[:3]
    assert eviction.reclaimed_bytes == 3000 and not eviction.short
    assert (eviction.recency_floor, eviction.options) == (64, plan.options)
    assert controller.eviction_plan(0, plan, tiers, PAGE_BYTES).reclaimed_bytes == 0
    short = controller.eviction_plan(10**6, plan, tiers, PAGE_BYTES)
    assert short.short and len(short) == len(want)


def test_an_evicted_page_holds_nothing_and_no_plan_moves_it():
    """In a tier map, an evicted page holds no bytes, counts toward no
    share and is no candidate; the bytes between two maps count what it
    held as given back."""
    tiers = np.zeros((1, 6), dtype=np.int8)
    after = tiers.copy()
    after[0, 1] = controller.EVICTED
    assert controller.bytes_given_back(tiers, after, PAGE_BYTES) == 1000
    assert not controller.candidates(after, 6 * 32, 32, 0)[0, 1]
    assert controller.full_pages(after, 6 * 32, 32, 0)[0, 1]
    assert controller.share_target_bytes(0.5, after, PAGE_BYTES, positions=6 * 32,
                                         page_tokens=32, recency_floor=0) == 2500
    assert controller.PLACES[controller.EVICTED] == "evicted"


# -- through a conversation -------------------------------------------------------------

NAME = "qwen2.5-0.5b-instruct"


def prompt(n, seed):
    return np.random.default_rng(seed).integers(1000, 100_000, n).astype(np.int32)


@pytest.fixture(scope="module")
def engine():
    e = Engine(require_model(NAME), kv_adaptive=True, kv_scoring=True, prefill_chunk=None)
    e.load_weights()
    return e


@pytest.mark.slow
def test_logits_after_an_eviction_are_those_with_its_pages_masked_to_the_bit(engine):
    """2048 positions, the last 48 decoded one by one so that the scorer
    has scores, then an eviction at 80% in one conversation and the same
    pages masked in another from the same snapshot: the next tokens'
    logits are the same to the bit, prefilled together and decoded one by
    one, and not those of the cache with nothing masked. The eviction
    meets the semantic plan's target."""
    with engine.conversation(for_snapshot=True) as chat:
        chat.force(prompt(2000, 160))
        chat.force(prompt(48, 1599), one_by_one=True)
        snapshot = chat.snapshot()
    logits, evicted = [], None
    for how in ("evicted", "masked", "whole"):
        with engine.conversation(start=snapshot) as chat:
            if how == "evicted":
                record = chat.evict(0.8)
                evicted = chat._cache.pages.evicted()
            elif how == "masked":
                for layer, page in zip(*np.nonzero(evicted)):
                    chat._cache.pages.mask(int(layer), int(page))
            logits.append([chat.force(prompt(40, 1600), logits=True),
                           chat.force(prompt(8, 1601), one_by_one=True, logits=True)])
    for got, want, whole in zip(*logits):
        np.testing.assert_array_equal(got, want)
        assert not np.array_equal(got, whole)
    c = record.controlled
    assert c["evicts"] and c["source"] == "semantic" and c["target_met"]
    assert c["target_bytes"] <= c["reclaimed_bytes"] < c["target_bytes"] + engine.page_bytes["FP16"]
    assert record.ended == "applied" and len(record.plan) == int(evicted.sum()) > 0
    assert c["candidate_effective_bits"] <= 16 * 0.2 + 1e-9


@pytest.mark.slow
def test_an_evictions_memory_returns_to_the_driver(engine):
    """The bytes the allocator gives back are those the driver sees given
    back, to a granule, and the cache holds less by them."""
    engine.measure_plans = True
    try:
        with engine.conversation() as chat:
            chat.turn(prompt(4096, 161), 8, stop_at_eos=False)
            record = chat.evict(0.8)
            chat.turn(prompt(8, 1610), 2, stop_at_eos=False)  # the cache reads on after it
    finally:
        engine.measure_plans = False
    (batch,) = record.batches
    returned = batch.cache_bytes_before - batch.cache_bytes_after
    seen = batch.own_bytes_before - batch.own_bytes_after
    assert returned >= record.controlled["reclaimed_bytes"] - _microinfer.granule_bytes()
    assert abs(seen - returned) <= _microinfer.granule_bytes()


def test_an_eviction_condition_runs_a_needle_and_logs_its_evicted_pages(engine):
    """evict-80 holds turn 1 as the source conditions do, and logs the
    eviction as a plan, its pages evicted among the tiers' shares."""
    assert rq3.shares_turn_1(rq3.condition("evict-80"))
    sample = needles.needle_set(load_tool("perplexity").held_out_text(), engine.tokenizer,
                                seed=160, samples_per_depth=1, haystack_tokens=1024)[5]
    (_, results), = rq3.runs(engine, sample, [rq3.condition("evict-80")], "question-blind")
    assert results["plan"]["evicts"] and results["plan"]["target_met"]
    assert results["plan"]["moves"] > 0 and "evicted" in results["tier_shares"]
    assert results["candidate_effective_bits"] <= 16 * 0.2 + 1e-9


# -- the summaries and the tools -------------------------------------------------------

KEPT = ["d00-0", "d10-0", "d20-0", "d30-0"]
SET = ("m", 140, 7936)


def logged_run(log, sample_id, name, correct, *, bits=3.4, reclaimed=1000):
    benchlog.append(
        "rq3-needle", model="m", context_length=8000, precision_tiers=None,
        config={"sample_id": sample_id, "condition": asdict(rq3.condition(name)),
                "experiment": "question-blind", "needle_seed": 140, "plan_seed": 0,
                "haystack_tokens": 7936},
        results={"correct": correct, "effective_bits": bits, "candidate_effective_bits": bits,
                 "plan": {"short": False, "target_met": True, "reclaimed_bytes": reclaimed}},
        log=log)


def test_eviction_is_given_against_the_semantic_source_at_each_level(tmp_path):
    """At 80%, semantic answers all four and eviction one, at the same bytes;
    at 70%, eviction answers the one semantic misses. Each pair is by
    sample, with McNemar's test, the change and its interval, and each
    condition's bits and bytes."""
    log = tmp_path / "benchmark.jsonl"
    benchlog.append("needle-screening", model="m", context_length=None,
                    precision_tiers={"FP16": 1.0},
                    config={"needle_seed": 140, "draws": 5, "haystack_tokens": 7936},
                    results={"kept": KEPT, "rejected": [], "complete": True}, log=log)
    for i, s in enumerate(KEPT):
        logged_run(log, s, "semantic-80", True)
        logged_run(log, s, "evict-80", i == 0, bits=3.2, reclaimed=1010)
        logged_run(log, s, "semantic-70", i != 0, bits=5.0)
        logged_run(log, s, "evict-70", True, bits=4.8)
    with pytest.raises(ValueError, match=r"no run of 4 .*--evictions"):
        rq3_summary.summarise_evictions(benchlog.read(log)[:-4], *SET)
    summary = rq3_summary.summarise_evictions(benchlog.read(log), *SET)
    at_80 = summary["levels"]["80%"]
    assert at_80["eviction"]["name"] == "evict-80" and at_80["semantic"]["name"] == "semantic-80"
    assert (at_80["eviction"]["correct"], at_80["semantic"]["correct"]) == (1, 4)
    assert (at_80["eviction_only"], at_80["semantic_only"]) == (0, 3)
    assert at_80["change"] == pytest.approx(-0.75)
    assert at_80["mcnemar_p"] == pytest.approx(rq3_summary.mcnemar(0, 3))
    assert at_80["eviction"]["reclaimed_bytes"] == 1010
    assert at_80["eviction"]["effective_bits"] == pytest.approx(3.2)
    at_70 = summary["levels"]["70%"]
    assert (at_70["eviction_only"], at_70["semantic_only"], at_70["change"]) == (1, 0, 0.25)
    assert len(summary["entries"]) == 16


@pytest.mark.slowest
def test_the_tools_run_and_summarise_eviction_on_the_needles_and_the_dialogue(tmp_path,
                                                                              monkeypatch):
    """--evictions runs the semantic source and eviction at both levels on a
    screened needle, and again none; its summary is logged. The anchor-fact
    tool does the same on its screened dialogues, and refuses --conditions
    beside --evictions."""
    needle_tool, summarise, anchor_tool = (load_tool(n) for n in
                                           ("rq3", "summarise_rq3", "anchors"))
    log = tmp_path / "benchmark.jsonl"
    real = benchlog.environment  # this test runs on a tree being changed
    for t in (needle_tool, summarise, anchor_tool):
        monkeypatch.setattr(t.benchlog, "environment",
                            lambda log=benchlog.DEFAULT_LOG: {**real(log), "git_dirty": False})
    shape = ["--issue", "160", "--model", NAME, "--needle-seed", "160",
             "--haystack-tokens", "1024", "--log", str(log)]
    benchlog.append("needle-screening", model=NAME, context_length=None,
                    precision_tiers={"FP16": 1.0},
                    config={"needle_seed": 160, "draws": 1, "haystack_tokens": 1024},
                    results={"kept": ["d50-0"], "rejected": [], "complete": True}, log=log)
    assert needle_tool.main([*shape, "--evictions"]) == 0
    runs = [e for e in benchlog.read(log) if e["kind"] == "rq3-needle"]
    assert [e["config"]["condition"]["name"] for e in runs] == list(rq3_summary.EVICTION_NAMES)
    assert all(e["results"]["plan"]["target_met"] for e in runs)
    assert needle_tool.main([*shape, "--evictions"]) == 0
    assert len([e for e in benchlog.read(log) if e["kind"] == "rq3-needle"]) == len(runs)
    assert summarise.main([*shape, "--evictions"]) == 0
    (summary,) = [e for e in benchlog.read(log)
                  if e["kind"] == rq3_summary.EVICTION_SUMMARY_KIND]
    assert list(summary["results"]["levels"]) == ["70%", "80%"]

    dialogue = ["--model", NAME, "--turns", "2", "--passage-tokens", "96", "--log", str(log)]
    assert anchor_tool.main(["--issue", "160", "--screen", "--per-gap", "1", "--draws", "3",
                             *dialogue]) == 0
    with pytest.raises(SystemExit):
        anchor_tool.main(["--issue", "160", "--evictions", "--conditions", "evict-80",
                          *dialogue])
    assert anchor_tool.main(["--issue", "160", "--evictions", *dialogue]) == 0
    assert anchor_tool.main(["--issue", "160", "--evictions", "--summarise", *dialogue]) == 0
    (anchor_summary,) = [e for e in benchlog.read(log) if e["kind"] == "anchor-summary"]
    levels = anchor_summary["results"]["evictions"]
    assert set(levels) == {"70%", "80%"} and levels["80%"]["eviction"]["n"] == 2
