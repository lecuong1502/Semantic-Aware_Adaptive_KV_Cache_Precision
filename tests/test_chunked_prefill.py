"""Chunked prefill (#15): a long prompt prefilled a chunk at a time, so the
workspace is sized to a chunk and not to the prompt.

**What "the same output" means here, and why not bits.** Attention is
row-invariant: a query's output does not depend on which other queries share
its launch, to the bit. cuBLAS is not. It picks its algorithm, tiling and
split-K from the GEMM's shape, so the same row of a projection can round
differently in a 37-row GEMM than in a 1,090-row one (measured: some sizes
agree, some do not). Chunked and single-shot prefill therefore agree to
within a bound, not to the bit. The same is already true of prefill (many
rows) against decode (one) in every engine that uses cuBLAS. The bound was
measured on the 23 prompts other than adversarial-00, at chunks of 1, 37, 128
and 512: logits within 0.096, KL within 7.5e-5, and the argmax the same at
every position except two near-ties. The assertions below allow twice the
logit difference, and allow an argmax to change only where single-shot's top
two logits are closer than that.

adversarial-00 is left out, and ADR-0010 says why. It amplifies any error of
its embedding about a thousand times, so the rounding differences between
GEMM algorithms become logit differences of up to 9 on it. That is a
property of the prompt, and it is measured there.
"""

import os
from pathlib import Path

import numpy as np
import pytest

from conftest import require_model, stable_free_bytes
from microinfer import Engine, model
from microinfer.gate import kl_divergence
from microinfer.golden import GoldenError, GoldenSet

REPO = Path(__file__).resolve().parent.parent
MODEL = "qwen2.5-0.5b-instruct"
MIB = 2**20

#: Twice the largest logit difference measured between chunked and single-shot
#: prefill on the well-conditioned prompts (0.096; module docstring).
LOGIT_BOUND = 0.2
KL_BOUND = 2e-4


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


def forward_with(engine, ids, chunk):
    before, engine.prefill_chunk = engine.prefill_chunk, chunk
    try:
        return engine.forward(ids)
    finally:
        engine.prefill_chunk = before


def assert_same_output(single, chunked, what):
    np.testing.assert_array_less(np.abs(single - chunked).max(), LOGIT_BOUND, err_msg=what)
    assert kl_divergence(single, chunked).max() < KL_BOUND, what
    top_two = np.sort(single, -1)[:, -2:]
    near_tie = top_two[:, 1] - top_two[:, 0] < LOGIT_BOUND
    flipped = single.argmax(-1) != chunked.argmax(-1)
    assert not np.any(flipped & ~near_tie), f"{what}: an argmax changed away from a near-tie"


# -- the same output -------------------------------------------------------------


@pytest.mark.parametrize("prompt_id,chunks", [
    ("medium-02", (1, 16)),        # 37 tokens: one position at a time, and a ragged last chunk
    ("adversarial-01", (37, 128)),  # 501 tokens
    ("long-01", (128, 512)),        # 1,090 tokens, the longest
])
def test_chunked_prefill_matches_single_shot(engine, golden, prompt_id, chunks):
    ids = golden[prompt_id].token_ids
    single = forward_with(engine, ids, None)
    for chunk in chunks:
        assert_same_output(single, forward_with(engine, ids, chunk), f"{prompt_id} chunk {chunk}")


def test_a_boundary_mid_sentence_leaves_positions_and_masking_alone(engine):
    """The boundary falls inside a sentence. The keys cached for the positions
    after it are what single-shot prefill caches, to within fp16 rounding.
    That is a check on RoPE: a position off by one rotates the fastest pair of
    dimensions by a whole radian, which would move a key by about its own
    size. The logits after the boundary, which see both chunks through the
    causal mask, agree within the bound."""
    text = ("The committee met on Tuesday to review the budget, and after a long "
            "discussion about the costs of the new building they agreed to delay it.")
    ids = engine.encode(text)
    boundary = len(ids) // 2  # mid-sentence, by construction of the text
    cfg, m = engine.config, engine._model

    def cached_keys(chunk):
        cache = model.ContiguousCache(cfg, capacity=len(ids))
        ws = model.Workspace(cfg, rows=chunk)
        for start in range(0, len(ids), chunk):
            m.run(ws, cache, ids[start:start + chunk])
        return np.stack([k.to_numpy() for k in cache.keys]).reshape(cfg.num_hidden_layers, len(ids), -1)

    single, chunked = cached_keys(len(ids)), cached_keys(boundary)
    after = slice(boundary, len(ids))
    scale = np.abs(single[:, after]).max()
    assert np.abs(single[:, after] - chunked[:, after]).max() < 0.05 * scale
    assert_same_output(forward_with(engine, ids, None)[after],
                       forward_with(engine, ids, boundary)[after], "after the boundary")


def test_generation_after_a_chunked_prefill_matches(engine, golden):
    ids = golden["long-03"].token_ids
    before = engine.prefill_chunk
    try:
        engine.prefill_chunk = None
        single = engine.generate(ids, 16, stop_at_eos=False)
        engine.prefill_chunk = 100
        chunked = engine.generate(ids, 16, stop_at_eos=False)
    finally:
        engine.prefill_chunk = before
    np.testing.assert_array_equal(chunked, single)


# -- memory -------------------------------------------------------------------------


def test_prefill_workspace_does_not_grow_with_the_prompt(engine):
    """Measured, not only computed: across prompts from 1 to 4 chunks long, the
    workspace the engine reports is one chunk's, and what the driver lost
    beyond the KV cache stays the same, to within the noise other processes
    put on the reading (ADR-0007, notes from #10 and #12)."""
    chunk = engine.prefill_chunk
    beyond_cache = []
    for chunks in (1, 2, 4):
        ids = np.arange(100, 100 + chunks * chunk, dtype=np.int32)
        engine.forward(ids)  # warm: cuBLAS loads kernels for new shapes once
        engine.reset_peak()
        before = stable_free_bytes()
        engine.forward(ids)
        peak = engine.peak_footprint()
        assert peak.workspace == model.Workspace(engine.config, rows=chunk).nbytes
        beyond_cache.append(before - peak.device_free - peak.kv_cache)
    print("\ndriver memory beyond the KV cache, MiB:", [round(b / MIB, 1) for b in beyond_cache])
    assert max(beyond_cache) - min(beyond_cache) < 8 * MIB


def test_a_sequence_past_the_context_window_is_refused_before_any_work(engine):
    window = engine.config.max_position_embeddings
    with pytest.raises(ValueError, match="context window"):
        engine.generate(np.zeros(window, np.int32), 2)
    with pytest.raises(ValueError, match="context window"):
        engine.forward(np.zeros(window + 1, np.int32))


def test_prefill_chunk_is_configured_and_checked():
    with pytest.raises(ValueError, match="prefill_chunk"):
        Engine(require_model(MODEL), prefill_chunk=0)
    assert Engine(require_model(MODEL), prefill_chunk=64).prefill_chunk == 64
    assert Engine(require_model(MODEL)).prefill_chunk == Engine.DEFAULT_PREFILL_CHUNK


@pytest.mark.skipif(os.environ.get("MICROINFER_LONG_TESTS") != "1",
                    reason="tens of minutes; set MICROINFER_LONG_TESTS=1 to run")
def test_a_32k_prompt_prefills_on_the_1_5b_model():
    """ADR-0003's experimental configuration on this 6 GiB card: the whole
    context window of Qwen2.5-1.5B, prefilled in chunks, then one token
    chosen. Weights alone are 2.88 GiB; the benchmark log has the time it
    takes and the memory it left (prefill-throughput, 32768 tokens)."""
    e = Engine(require_model("qwen2.5-1.5b-instruct"))
    e.load_weights()
    window = e.config.max_position_embeddings
    ids = np.random.default_rng(15).integers(1000, 100_000, window).astype(np.int32)
    e.reset_peak()
    out = e.generate(ids, 1, stop_at_eos=False)
    assert out.shape == (1,) and 0 <= out[0] < e.config.vocab_size
    peak = e.peak_footprint()
    assert peak.workspace == model.Workspace(e.config, rows=e.prefill_chunk).nbytes
