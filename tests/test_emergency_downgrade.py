"""An allocation that fails: an emergency downgrade, then a graceful stop
(#107, Seam A).

When reserving a step's pages raises OutOfMemory, an adaptive engine makes a
RED plan for no headroom at all, applies it at once, and retries the step
once. Reserving is all or nothing (KVPages::reserve), so a failed step left
the cache as it was and may be run again. If the plan frees nothing, as when
every page is within the recency floor, or the retry fails too, the generation
stops gracefully: generate returns the tokens it kept, `ending` says
"exhausted" and the position reached, and the cache is released.

The failures here are made where the allocator would raise them, in the
cache's reserve, on the step the test chooses; the downgrades that answer
them are real. In the last two the device is full, another allocator in
the process having taken every granule the driver would give, and the
refusal is the driver's. They show what the spare granules an adaptive
cache keeps are for.
"""

import numpy as np
import pytest

from conftest import require_model
from microinfer import Engine, _microinfer, model, nvml
from microinfer import engine as engine_module
from microinfer.engine import COMPLETE, EXHAUSTED

MIB = 2**20
NAME = "qwen2.5-0.5b-instruct"


@pytest.fixture(scope="module")
def engine() -> Engine:
    e = Engine(require_model(NAME), kv_adaptive=True, prefill_chunk=None)
    e.load_weights()
    return e


def failing_at(monkeypatch, tokens, times):
    """Make reserving `tokens` positions or more fail, `times` times in all
    (None for every time); returns the list of the sizes refused."""
    refused = []
    real = model.PagedCache.reserve

    def reserve(cache, n):
        if n >= tokens and (times is None or len(refused) < times):
            refused.append(n)
            raise _microinfer.OutOfMemory(f"simulated: reserving {n} positions")
        real(cache, n)

    monkeypatch.setattr(model.PagedCache, "reserve", reserve)
    return refused


def prompt(n, seed):
    return np.random.default_rng(seed).integers(1000, 100_000, n).astype(np.int32)


def test_a_failed_allocation_is_recovered_by_an_emergency_plan(engine, monkeypatch):
    """The 20th decode step's pages are refused once: the engine downgrades
    every page outside the recency floor, at once, retries the step, and
    the generation completes with every token asked for."""
    ids = prompt(1024, 307)
    refused = failing_at(monkeypatch, len(ids) + 20, times=1)
    engine.plans.clear()
    out = engine.generate(ids, 40, stop_at_eos=False)
    assert refused == [len(ids) + 20]
    assert len(out) == 40
    assert engine.ending.state == COMPLETE and engine.ending.positions == len(ids) + 39
    emergency = [p for p in engine.plans if p.emergency]
    assert len(emergency) == 1
    record = emergency[0]
    assert record.ended == "applied" and record.applied == len(record.plan) > 0
    assert record.batches[0].cache_bytes_after < record.batches[0].cache_bytes_before
    assert "simulated" in record.emergency


def test_a_retry_that_fails_too_ends_the_generation_exhausted_with_its_tokens(engine,
                                                                             monkeypatch):
    """Refused again after the emergency plan: the generation ends "exhausted"
    at the position it reached, with the tokens decoded before the failure,
    those an unfailed generation decodes first, and its memory released."""
    ids = prompt(1024, 308)
    want = engine.generate(ids, 40, stop_at_eos=False)
    before = nvml.settled_own_used_bytes()
    refused = failing_at(monkeypatch, len(ids) + 20, times=None)
    out = engine.generate(ids, 40, stop_at_eos=False)
    assert refused == [len(ids) + 20] * 2  # the step, and its one retry
    np.testing.assert_array_equal(out, want[:20])
    ending = engine.ending
    assert ending.state == EXHAUSTED and ending.positions == len(ids) + 19
    np.testing.assert_array_equal(ending.tokens, out)
    assert "simulated" in ending.reason
    assert engine._cache is None
    assert nvml.settled_own_used_bytes() <= before + _microinfer.granule_bytes()


def test_with_nothing_to_downgrade_the_generation_stops_without_a_retry(engine, monkeypatch):
    """Every page within the recency floor: the emergency plan has nothing
    to downgrade, so the step is not retried and the generation is exhausted."""
    ids = prompt(64, 309)
    refused = failing_at(monkeypatch, len(ids) + 5, times=None)
    out = engine.generate(ids, 40, stop_at_eos=False)
    assert refused == [len(ids) + 5]
    assert len(out) == 5 and engine.ending.state == EXHAUSTED
    assert "nothing to downgrade" in engine.ending.reason
    assert engine.plans[-1].emergency and engine.plans[-1].ended == "nothing to downgrade"


def test_a_prefill_chunk_refused_ends_the_generation_with_no_token(monkeypatch):
    """A failure while prefilling, recovered from or not, is answered as one
    while decoding: here the chunk is refused again, and the generation ends
    exhausted where the prefill reached, with no token."""
    engine = Engine(require_model(NAME), kv_adaptive=True, prefill_chunk=256)
    engine.load_weights()
    ids = prompt(1024, 310)
    refused = failing_at(monkeypatch, 768, times=None)
    out = engine.generate(ids, 8, stop_at_eos=False)
    assert refused == [768, 768]
    assert len(out) == 0
    assert engine.ending.state == EXHAUSTED and engine.ending.positions == 512
    assert [p.emergency is not None for p in engine.plans] == [True]


def test_out_of_memory_outside_a_step_s_pages_ends_the_generation_exhausted(engine,
                                                                            monkeypatch):
    """A step's logits refused memory, not its pages: no emergency plan
    answers it, and the generation ends exhausted, as gracefully. A
    generation refused before it begins leaves the last ending in place."""
    ids = prompt(256, 312)
    real = model.Model.greedy_last
    calls = []

    def greedy_last(self, ws, n):
        calls.append(n)
        if len(calls) == 6:
            raise _microinfer.OutOfMemory("simulated: the logits")
        return real(self, ws, n)

    monkeypatch.setattr(model.Model, "greedy_last", greedy_last)
    out = engine.generate(ids, 16, stop_at_eos=False)
    assert len(out) == 5 and engine.ending.state == EXHAUSTED
    assert engine.ending.reason == "simulated: the logits"
    with pytest.raises(ValueError):
        engine.generate(np.full(engine.config.max_position_embeddings, 1000, np.int32), 4)
    assert engine.ending.state == EXHAUSTED
    monkeypatch.undo()
    engine.generate(ids, 2, stop_at_eos=False)
    assert engine.ending.state == COMPLETE


def test_an_engine_that_does_not_adapt_raises_as_it_did(monkeypatch):
    engine = Engine(require_model(NAME))
    engine.load_weights()
    failing_at(monkeypatch, 70, times=None)
    with pytest.raises(_microinfer.OutOfMemory):
        engine.generate(prompt(64, 311), 16, stop_at_eos=False)


def device_full(engine, new, monkeypatch):
    """Generate `new` tokens after a 1024-token prompt, the device filled
    at the 64th decode step: before it reserves its pages, another
    allocator takes every granule the driver will give, and holds them
    until the generation ends. The tokens, and what this process held by NVML
    before the generation and after it, the filler given back."""
    ids = prompt(1024, 312)
    engine.generate(ids, 8, stop_at_eos=False)  # warm
    before = nvml.settled_own_used_bytes()
    engine.plans.clear()
    g = _microinfer.granule_bytes()
    held = [_microinfer.PagedKVCache([g] * 4, [nvml.memory().total // g] + [0] * 3)]
    real = model.PagedCache.reserve

    def reserve(cache, n):
        if n == len(ids) + 64:
            try:
                for i in range(nvml.memory().total // g):
                    held[0].allocate(0, i, _microinfer.Tier.FP16)
            except _microinfer.OutOfMemory:
                pass
        real(cache, n)

    monkeypatch.setattr(model.PagedCache, "reserve", reserve)
    try:
        out = engine.generate(ids, new, stop_at_eos=False)
    finally:
        held.clear()  # the filler, and every granule it took
    ending = engine.ending
    print(f"\n{ending.state} at {ending.positions} positions with {len(out)} tokens; "
          f"{ending.reason[:300]}; plans {[(len(p.plan), p.ended[:80]) for p in engine.plans]}")
    return out, before, nvml.settled_own_used_bytes()


@pytest.mark.slow
def test_a_full_device_is_recovered_from_by_an_emergency_plan(engine, monkeypatch):
    """The device filled mid decode, the next granule the cache needs is
    refused by the driver: the emergency plan's first downgrades go into
    the spare granules the adaptive cache keeps, its later ones into the
    room the FP16 pages it freed gave back, and the generation completes."""
    out, _, _ = device_full(engine, 384, monkeypatch)
    emergency = [p for p in engine.plans if p.emergency]
    assert emergency and "CUDA_ERROR_OUT_OF_MEMORY" in emergency[0].emergency
    record = emergency[0]
    assert record.ended == "applied" and record.applied > 0
    assert record.batches[0].cache_bytes_after < record.batches[0].cache_bytes_before
    assert engine.ending.state == COMPLETE and len(out) == 384


@pytest.mark.slow
def test_a_full_device_with_no_spares_ends_the_generation_exhausted(engine, monkeypatch):
    """Without the spare granules the emergency plan cannot start, its first
    downgrade refused a granule: the generation ends exhausted, with its
    tokens, no error raised, and gives back what it held."""
    monkeypatch.setattr(engine_module, "EMERGENCY_SPARES", 0)
    out, before, after = device_full(engine, 384, monkeypatch)
    ending = engine.ending
    assert ending.state == EXHAUSTED and 0 < len(out) < 384
    np.testing.assert_array_equal(ending.tokens, out)
    assert "freed nothing" in ending.reason
    assert after <= before + _microinfer.granule_bytes()
