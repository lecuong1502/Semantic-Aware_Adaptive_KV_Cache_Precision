"""The device surface the engine's forward pass runs on (Seam B, #12).

Each existing kernel is reached two ways: through the NumPy wrappers, which the
per-kernel tests hold to ADR-0006's gate, and through `_microinfer.device`,
which the engine calls with operands already on the device. Both are the same
launcher (`device_ops.h`), so the device surface is tested by showing it is
bit-identical to the tested one, not by a second numerics test of the same
arithmetic.

The kernels new here are tested for what they are:
- `embed` is a gather, and must be exact.
- `add` is one fp32 addition rounded once.
- `argmax` is a selection, with the tie rule argmax conventionally has.
- `logits` is the LM head with an fp32 output.

Every dimension comes from the model cards, both models' (CONTRIBUTING), with
one exception, VOCAB_SLICE. The input-validation tests at the end use literals,
because what they check is a shape the model never has.
"""

import numpy as np
import pytest
from ulp_gate import fp16_exact, fp32_accumulation_bound

from microinfer import _microinfer
from microinfer.config import ModelConfig
from microinfer.models import VERIFIED

device = _microinfer.device
MODELS = sorted(VERIFIED)
RNG = np.random.default_rng(12)

#: Rows of the LM head used where a test needs one. The kernels take the
#: vocabulary size as a parameter and never assume it; the real 151,936 rows
#: would be 545 MiB of random fp32 per test for no extra coverage. A slice
#: that is not a multiple of any block size keeps the ragged edge in play.
VOCAB_SLICE = 1021


@pytest.fixture(params=MODELS)
def cfg(request) -> ModelConfig:
    return ModelConfig.from_card(request.param)


def put(host: np.ndarray):
    return _microinfer.upload_fp16(np.ascontiguousarray(host, np.float32).reshape(-1))


def scratch(rows, vocab):
    return device.empty(device.scratch_elements(rows, vocab))


def get(tensor, shape) -> np.ndarray:
    return tensor.to_numpy().reshape(shape)


def normal(*shape, scale=1.0):
    return fp16_exact(RNG.standard_normal(shape).astype(np.float32) * scale)


# -- the existing kernels, reached from the device --------------------------


def test_rmsnorm_is_the_tested_kernel(cfg):
    rows, hidden = 7, cfg.hidden_size
    x, w = normal(rows, hidden), normal(hidden)
    out = device.empty(rows * hidden)
    device.rmsnorm(put(x), put(w), out, rows, hidden, cfg.rms_norm_eps)
    np.testing.assert_array_equal(get(out, (rows, hidden)),
                                  _microinfer.rmsnorm(x, w, cfg.rms_norm_eps))


@pytest.mark.parametrize("bias", [False, True])
def test_linear_is_the_tested_kernel(cfg, bias):
    rows, n_in, n_out = 5, cfg.hidden_size, cfg.num_key_value_heads * cfg.head_dim
    x, w = normal(rows, n_in), normal(n_out, n_in, scale=n_in**-0.5)
    b = normal(n_out) if bias else None
    out = device.empty(rows * n_out)
    device.linear(put(x), put(w), put(b) if bias else None, out, rows, n_in, n_out)
    np.testing.assert_array_equal(get(out, (rows, n_out)), _microinfer.linear(x, w, b))


def test_rope_is_the_tested_kernel(cfg):
    seq, heads, hd = 6, cfg.num_attention_heads, cfg.head_dim
    x = normal(seq, heads, hd)
    positions = np.array([0, 1, 2, 31, 2048, cfg.max_position_embeddings - 1], np.int32)
    out = device.empty(x.size)
    device.rope(put(x), device.index(positions), out, seq, heads, hd, cfg.rope_theta)
    np.testing.assert_array_equal(get(out, x.shape),
                                  _microinfer.rope(x, positions, cfg.rope_theta))


def test_swiglu_is_the_tested_kernel(cfg):
    gate, up = normal(3, cfg.intermediate_size), normal(3, cfg.intermediate_size)
    out = device.empty(gate.size)
    device.swiglu(put(gate), put(up), out, gate.size)
    np.testing.assert_array_equal(get(out, gate.shape), _microinfer.swiglu(gate, up))


def test_attention_reads_the_first_seq_k_rows_of_a_larger_cache(cfg):
    """The engine's KV cache is sized for the whole generation and filled as it
    goes, so attention is always handed a buffer longer than seq_k. What lies
    past seq_k must not be read: it is filled with NaN here, and a single read
    of it would poison every output."""
    heads, kv_heads, hd = cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim
    seq_q, seq_k, capacity = 3, 40, 64
    q, k, v = normal(seq_q, heads, hd), normal(seq_k, kv_heads, hd), normal(seq_k, kv_heads, hd)
    pad = np.full((capacity - seq_k, kv_heads, hd), np.nan, np.float32)
    out = device.empty(q.size)
    device.attention(put(q), put(np.concatenate([k, pad])), put(np.concatenate([v, pad])),
                     None, None, out, seq_q, seq_k, heads, kv_heads, hd)
    np.testing.assert_array_equal(get(out, q.shape), _microinfer.attention(q, k, v))


def test_a_view_writes_only_its_own_elements(cfg):
    """The KV cache is written through views, one step's rows at an offset."""
    rows, n_in, n_out = 2, cfg.hidden_size, cfg.num_key_value_heads * cfg.head_dim
    x, w = normal(rows, n_in), normal(n_out, n_in, scale=n_in**-0.5)
    before = normal(5 * rows * n_out)
    target = put(before)
    offset = 3 * rows * n_out
    device.linear(put(x), put(w), None, device.view(target, offset, rows * n_out),
                  rows, n_in, n_out)
    after = target.to_numpy()
    np.testing.assert_array_equal(after[offset:offset + rows * n_out],
                                  _microinfer.linear(x, w).reshape(-1))
    np.testing.assert_array_equal(after[:offset], before[:offset])
    np.testing.assert_array_equal(after[offset + rows * n_out:], before[offset + rows * n_out:])


# -- the kernels new here ---------------------------------------------------


def test_embed_is_an_exact_gather(cfg):
    vocab, hidden = VOCAB_SLICE, cfg.hidden_size
    table = normal(vocab, hidden)
    ids = np.array([3, 0, vocab - 1, 3, 17], np.int32)
    out = device.empty(len(ids) * hidden)
    device.embed(device.index(ids), put(table), out, hidden, vocab)
    np.testing.assert_array_equal(get(out, (len(ids), hidden)), table[ids])


def test_add_is_one_fp32_addition_rounded_once_and_may_update_in_place(cfg):
    """The residual stream's update: three tokens' worth."""
    a, b = normal(3 * cfg.hidden_size), normal(3 * cfg.hidden_size)
    expected = (a.astype(np.float32) + b.astype(np.float32)).astype(np.float16).astype(np.float32)
    ta = put(a)
    device.add(ta, put(b), ta, a.size)
    np.testing.assert_array_equal(ta.to_numpy(), expected)


def test_logits_are_fp32_and_within_fp32_accumulation_of_float64(cfg):
    rows, hidden, vocab = 4, cfg.hidden_size, VOCAB_SLICE
    x, head = normal(rows + 2, hidden), normal(vocab, hidden, scale=hidden**-0.5)
    got = device.logits(put(x), put(head), scratch(rows, vocab), 2, rows, hidden, vocab)
    assert got.dtype == np.float32 and got.shape == (rows, vocab)
    x64, h64 = x[2:].astype(np.float64), head.astype(np.float64)
    ref, terms = x64 @ h64.T, np.abs(x64) @ np.abs(h64).T
    # There is no fp16 store here to round the difference away, so the output
    # is held to the accumulation bound itself, in fp32's own units.
    bound = fp32_accumulation_bound(terms, hidden)
    assert np.all(np.abs(got - ref) <= bound), np.max(np.abs(got - ref) / bound)


def test_greedy_is_the_argmax_of_the_logits(cfg):
    rows, hidden, vocab = 6, cfg.hidden_size, VOCAB_SLICE
    x, head = normal(rows, hidden), normal(vocab, hidden, scale=hidden**-0.5)
    got = device.greedy(put(x), put(head), scratch(rows, vocab), 0, rows, hidden, vocab)
    np.testing.assert_array_equal(got, device.logits(put(x), put(head), scratch(rows, vocab), 0, rows, hidden, vocab).argmax(-1))


def test_greedy_breaks_ties_to_the_lowest_index(cfg):
    """Duplicate rows in the head give bit-identical logits, a tie by
    construction. NumPy's argmax, and HuggingFace's through torch, take the
    first; so must the device."""
    hidden, vocab = cfg.hidden_size, VOCAB_SLICE
    head = normal(vocab, hidden, scale=hidden**-0.5)
    x = normal(1, hidden)
    best = int(device.logits(put(x), put(head), scratch(1, vocab), 0, 1, hidden, vocab).argmax())
    for late in (vocab - 1, best + 1 if best + 1 < vocab else best - 1):
        tied = head.copy()
        tied[late] = head[best]
        first = min(best, late)
        assert device.greedy(put(x), put(tied), scratch(1, vocab), 0, 1, hidden, vocab)[0] == first


# -- bounds -----------------------------------------------------------------


def test_an_operand_too_small_is_rejected_before_launch():
    small = device.empty(10)
    with pytest.raises(ValueError, match="out holds 10"):
        device.rmsnorm(put(normal(2, 8)), put(normal(8)), small, 2, 8, 1e-6)


def test_a_view_cannot_reach_past_its_tensor():
    t = device.empty(10)
    with pytest.raises(ValueError, match="view"):
        device.view(t, 8, 3)


def test_logits_refuse_scratch_too_small_for_them():
    hidden, vocab = 8, 100
    small = device.empty(device.scratch_elements(1, vocab) - 1)
    with pytest.raises(ValueError, match="scratch"):
        device.logits(put(normal(1, hidden)), put(normal(vocab, hidden)), small, 0, 1, hidden, vocab)


def test_a_negative_first_row_is_refused_rather_than_wrapped():
    """first_row + rows can be positive while first_row is not; the pointer
    offset computed from it would then wrap to anywhere."""
    hidden, vocab = 8, 100
    x, head = put(normal(4, hidden)), put(normal(vocab, hidden))
    for call in (device.logits, device.greedy):
        with pytest.raises(ValueError, match="first_row"):
            call(x, head, scratch(3, vocab), -1, 3, hidden, vocab)


def test_greedy_refuses_a_row_with_no_finite_logit():
    """No argmax exists; the kernel's answer is one past the vocabulary, and it
    must not reach the caller as a token."""
    hidden, vocab = 8, 100
    head = put(np.full((vocab, hidden), np.nan, np.float32))
    with pytest.raises(RuntimeError, match="no finite logit"):
        device.greedy(put(normal(1, hidden)), head, scratch(1, vocab), 0, 1, hidden, vocab)


def test_rope_refuses_to_write_over_its_input():
    x = put(normal(1, 1, 8))
    with pytest.raises(ValueError, match="over its input"):
        device.rope(x, device.index(np.zeros(1, np.int32)), x, 1, 1, 8, 1e4)


def test_swiglu_refuses_to_write_over_its_input():
    gate, up = put(normal(8)), put(normal(8))
    with pytest.raises(ValueError, match="over its input"):
        device.swiglu(gate, up, gate, 8)


def test_attention_refuses_more_queries_than_keys():
    q = put(normal(3, 1, 8))
    kv = put(normal(2, 1, 8))
    with pytest.raises(ValueError, match="seq_q"):
        device.attention(q, kv, kv, None, None, device.empty(24), 3, 2, 1, 1, 8)
