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
"""

import numpy as np
import pytest
from ulp_gate import FP32_REL_ULP, fp16_exact

from microinfer import _microinfer
from microinfer.config import ModelConfig
from microinfer.models import VERIFIED

device = _microinfer.device
CFG = ModelConfig.from_card(sorted(VERIFIED)[0])
RNG = np.random.default_rng(12)


def put(host: np.ndarray):
    return _microinfer.upload_fp16(np.ascontiguousarray(host, np.float32).reshape(-1))


def scratch(rows, vocab):
    return device.empty(device.scratch_elements(rows, vocab))


def get(tensor, shape) -> np.ndarray:
    return tensor.to_numpy().reshape(shape)


def normal(*shape, scale=1.0):
    return fp16_exact(RNG.standard_normal(shape).astype(np.float32) * scale)


# -- the existing kernels, reached from the device --------------------------


def test_rmsnorm_is_the_tested_kernel():
    rows, hidden = 7, CFG.hidden_size
    x, w = normal(rows, hidden), normal(hidden)
    out = device.empty(rows * hidden)
    device.rmsnorm(put(x), put(w), out, rows, hidden, CFG.rms_norm_eps)
    np.testing.assert_array_equal(get(out, (rows, hidden)),
                                  _microinfer.rmsnorm(x, w, CFG.rms_norm_eps))


@pytest.mark.parametrize("bias", [False, True])
def test_linear_is_the_tested_kernel(bias):
    rows, n_in, n_out = 5, CFG.hidden_size, CFG.num_key_value_heads * CFG.head_dim
    x, w = normal(rows, n_in), normal(n_out, n_in, scale=n_in**-0.5)
    b = normal(n_out) if bias else None
    out = device.empty(rows * n_out)
    device.linear(put(x), put(w), put(b) if bias else None, out, rows, n_in, n_out)
    np.testing.assert_array_equal(get(out, (rows, n_out)), _microinfer.linear(x, w, b))


def test_rope_is_the_tested_kernel():
    seq, heads, hd = 6, CFG.num_attention_heads, CFG.head_dim
    x = normal(seq, heads, hd)
    positions = np.array([0, 1, 2, 31, 2048, 9000], np.int32)
    out = device.empty(x.size)
    device.rope(put(x), device.index(positions), out, seq, heads, hd, CFG.rope_theta)
    np.testing.assert_array_equal(get(out, x.shape),
                                  _microinfer.rope(x, positions, CFG.rope_theta))


def test_swiglu_is_the_tested_kernel():
    gate, up = normal(3, CFG.intermediate_size), normal(3, CFG.intermediate_size)
    out = device.empty(gate.size)
    device.swiglu(put(gate), put(up), out, gate.size)
    np.testing.assert_array_equal(get(out, gate.shape), _microinfer.swiglu(gate, up))


def test_attention_reads_the_first_seq_k_rows_of_a_larger_cache():
    """The engine's KV cache is sized for the whole generation and filled as it
    goes, so attention is always handed a buffer longer than seq_k. What lies
    past seq_k must not be read: it is filled with NaN here, and a single read
    of it would poison every output."""
    heads, kv_heads, hd = CFG.num_attention_heads, CFG.num_key_value_heads, CFG.head_dim
    seq_q, seq_k, capacity = 3, 40, 64
    q, k, v = normal(seq_q, heads, hd), normal(seq_k, kv_heads, hd), normal(seq_k, kv_heads, hd)
    pad = np.full((capacity - seq_k, kv_heads, hd), np.nan, np.float32)
    out = device.empty(q.size)
    device.attention(put(q), put(np.concatenate([k, pad])), put(np.concatenate([v, pad])),
                     None, out, seq_q, seq_k, heads, kv_heads, hd, 0.0)
    np.testing.assert_array_equal(get(out, q.shape), _microinfer.attention(q, k, v))


def test_a_view_writes_only_its_own_elements():
    """The KV cache is written through views, one step's rows at an offset."""
    rows, n_in, n_out = 2, 64, 32
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


def test_embed_is_an_exact_gather():
    vocab, hidden = 50, CFG.hidden_size
    table = normal(vocab, hidden)
    ids = np.array([3, 0, 49, 3, 17], np.int32)
    out = device.empty(len(ids) * hidden)
    device.embed(device.index(ids), put(table), out, hidden, vocab)
    np.testing.assert_array_equal(get(out, (len(ids), hidden)), table[ids])


def test_add_is_one_fp32_addition_rounded_once_and_may_update_in_place():
    a, b = normal(1000), normal(1000)
    expected = (a.astype(np.float32) + b.astype(np.float32)).astype(np.float16).astype(np.float32)
    ta = put(a)
    device.add(ta, put(b), ta, a.size)
    np.testing.assert_array_equal(ta.to_numpy(), expected)


def test_logits_are_fp32_and_within_fp32_accumulation_of_float64():
    rows, hidden, vocab = 4, CFG.hidden_size, 3000
    x, head = normal(rows + 2, hidden), normal(vocab, hidden, scale=hidden**-0.5)
    got = device.logits(put(x), put(head), scratch(rows, vocab), 2, rows, hidden, vocab)
    assert got.dtype == np.float32 and got.shape == (rows, vocab)
    x64, h64 = x[2:].astype(np.float64), head.astype(np.float64)
    ref, terms = x64 @ h64.T, np.abs(x64) @ np.abs(h64).T
    # fp32 accumulation's probabilistic bound (tests/ulp_gate.py), in fp32's own
    # units: there is no fp16 store here to round away the difference.
    bound = np.sqrt(hidden) * FP32_REL_ULP * terms
    assert np.all(np.abs(got - ref) <= bound), np.max(np.abs(got - ref) / bound)


def test_greedy_is_the_argmax_of_the_logits():
    rows, hidden, vocab = 6, CFG.hidden_size, 5000
    x, head = normal(rows, hidden), normal(vocab, hidden, scale=hidden**-0.5)
    got = device.greedy(put(x), put(head), scratch(rows, vocab), 0, rows, hidden, vocab)
    np.testing.assert_array_equal(got, device.logits(put(x), put(head), scratch(rows, vocab), 0, rows, hidden, vocab).argmax(-1))


def test_greedy_breaks_ties_to_the_lowest_index():
    """Duplicate rows in the head give bit-identical logits, a tie by
    construction. NumPy's argmax, and HuggingFace's through torch, take the
    first; so must the device."""
    hidden, vocab = 64, 1024
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
        device.attention(q, kv, kv, None, device.empty(24), 3, 2, 1, 1, 8, 0.0)
