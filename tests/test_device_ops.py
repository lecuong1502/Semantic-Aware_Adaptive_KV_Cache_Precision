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
from conftest import each
from ulp_gate import assert_within_gate, fp16_exact, fp32_accumulation_bound

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


CFGS = [ModelConfig.from_card(name) for name in MODELS]


def put(host: np.ndarray):
    return _microinfer.upload_fp16(np.ascontiguousarray(host, np.float32).reshape(-1))


def scratch(rows, vocab):
    return device.empty(device.scratch_elements(rows, vocab))


def get(tensor, shape) -> np.ndarray:
    return tensor.to_numpy().reshape(shape)


def normal(*shape, scale=1.0):
    return fp16_exact(RNG.standard_normal(shape).astype(np.float32) * scale)


# -- the existing kernels, reached from the device --------------------------


def rmsnorm_is_the_tested_kernel(cfg):
    rows, hidden = 7, cfg.hidden_size
    x, w = normal(rows, hidden), normal(hidden)
    out = device.empty(rows * hidden)
    device.rmsnorm(put(x), put(w), out, rows, hidden, cfg.rms_norm_eps)
    np.testing.assert_array_equal(get(out, (rows, hidden)),
                                  _microinfer.rmsnorm(x, w, cfg.rms_norm_eps))


def linear_is_the_tested_kernel(cfg, bias):
    rows, n_in, n_out = 5, cfg.hidden_size, cfg.num_key_value_heads * cfg.head_dim
    x, w = normal(rows, n_in), normal(n_out, n_in, scale=n_in**-0.5)
    b = normal(n_out) if bias else None
    out = device.empty(rows * n_out)
    device.linear(put(x), put(w), put(b) if bias else None, out, rows, n_in, n_out)
    np.testing.assert_array_equal(get(out, (rows, n_out)), _microinfer.linear(x, w, b))


def rope_is_the_tested_kernel(cfg):
    seq, heads, hd = 6, cfg.num_attention_heads, cfg.head_dim
    x = normal(seq, heads, hd)
    positions = np.array([0, 1, 2, 31, 2048, cfg.max_position_embeddings - 1], np.int32)
    out = device.empty(x.size)
    device.rope(put(x), device.index(positions), out, seq, heads, hd, cfg.rope_theta)
    np.testing.assert_array_equal(get(out, x.shape),
                                  _microinfer.rope(x, positions, cfg.rope_theta))


def swiglu_is_the_tested_kernel(cfg):
    gate, up = normal(3, cfg.intermediate_size), normal(3, cfg.intermediate_size)
    out = device.empty(gate.size)
    device.swiglu(put(gate), put(up), out, gate.size)
    np.testing.assert_array_equal(get(out, gate.shape), _microinfer.swiglu(gate, up))


def attention_reads_the_first_seq_k_rows_of_a_larger_cache(cfg):
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


def a_view_writes_only_its_own_elements(cfg):
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


def embed_is_an_exact_gather(cfg):
    vocab, hidden = VOCAB_SLICE, cfg.hidden_size
    table = normal(vocab, hidden)
    ids = np.array([3, 0, vocab - 1, 3, 17], np.int32)
    out = device.empty(len(ids) * hidden)
    device.embed(device.index(ids), put(table), out, hidden, vocab)
    np.testing.assert_array_equal(get(out, (len(ids), hidden)), table[ids])


def add_is_one_fp32_addition_rounded_once_and_may_update_in_place(cfg):
    """The residual stream's update: three tokens' worth."""
    a, b = normal(3 * cfg.hidden_size), normal(3 * cfg.hidden_size)
    expected = (a.astype(np.float32) + b.astype(np.float32)).astype(np.float16).astype(np.float32)
    ta = put(a)
    device.add(ta, put(b), ta, a.size)
    np.testing.assert_array_equal(ta.to_numpy(), expected)


def logits_are_fp32_and_within_fp32_accumulation_of_float64(cfg):
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


def greedy_is_the_argmax_of_the_logits(cfg):
    rows, hidden, vocab = 6, cfg.hidden_size, VOCAB_SLICE
    x, head = normal(rows, hidden), normal(vocab, hidden, scale=hidden**-0.5)
    got = device.greedy(put(x), put(head), scratch(rows, vocab), 0, rows, hidden, vocab)
    np.testing.assert_array_equal(got, device.logits(put(x), put(head), scratch(rows, vocab), 0, rows, hidden, vocab).argmax(-1))


def greedy_breaks_ties_to_the_lowest_index(cfg):
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


# -- the fp32 residual stream (ADR-0010) ----------------------------------


def the_embedding_widens_exactly_to_fp32(cfg):
    vocab, hidden = VOCAB_SLICE, cfg.hidden_size
    table = normal(vocab, hidden)
    ids = np.array([3, 0, vocab - 1], np.int32)
    out = device.empty_f32(len(ids) * hidden)
    device.embed_f32(device.index(ids), put(table), out, hidden, vocab)
    np.testing.assert_array_equal(out.to_numpy().reshape(len(ids), hidden), table[ids])


def rmsnorm_of_fp32_is_the_tested_kernel_on_fp16_exact_input(cfg):
    """Every element is widened to fp32 on read, so an fp16-exact input gives
    the fp16 path's bits exactly."""
    rows, hidden = 5, cfg.hidden_size
    x, w = normal(rows, hidden), normal(hidden)
    x32 = device.upload_f32(x)
    out = device.empty(x.size)
    device.rmsnorm_f32(x32, put(w), out, rows, hidden, cfg.rms_norm_eps)
    np.testing.assert_array_equal(get(out, x.shape), _microinfer.rmsnorm(x, w, cfg.rms_norm_eps))


def rmsnorm_of_fp32_meets_the_gate_on_input_fp16_cannot_hold(cfg):
    """The point of the fp32 residual: inputs with more precision than fp16,
    normalised without being rounded to fp16 first. Held to ADR-0006's gate
    against float64, as a product-shaped kernel, with no floor."""
    rows, hidden = 4, cfg.hidden_size
    x = (RNG.standard_normal((rows, hidden)) * 300).astype(np.float32)
    x[:, 0] += 1700  # a massive activation, as the residual carries from layer 3 on
    w = normal(hidden)
    x32 = device.upload_f32(x)
    out = device.empty(x.size)
    device.rmsnorm_f32(x32, put(w), out, rows, hidden, cfg.rms_norm_eps)
    x64 = x.astype(np.float64)
    ref = x64 / np.sqrt((x64 * x64).mean(-1, keepdims=True) + cfg.rms_norm_eps) * w
    assert_within_gate(get(out, x.shape), ref)


def a_projection_accumulates_into_fp32_without_rounding(cfg):
    """out += x W^T in fp32, twice, as the o_proj and down_proj updates of one
    layer do; within fp32 accumulation of float64, with the terms of both the
    sums and the residual they are added to."""
    rows, n_in, n_out = 3, cfg.intermediate_size, cfg.hidden_size
    x, w = normal(rows, n_in), normal(n_out, n_in, scale=n_in**-0.5)
    r = (RNG.standard_normal((rows, n_out)) * 500).astype(np.float32)
    out = device.upload_f32(r)
    device.linear_accumulate(put(x), put(w), out, rows, n_in, n_out)
    device.linear_accumulate(put(x), put(w), out, rows, n_in, n_out)
    x64, w64 = x.astype(np.float64), w.astype(np.float64)
    ref = r + 2 * (x64 @ w64.T)
    terms = np.abs(r) + 2 * (np.abs(x64) @ np.abs(w64).T)
    bound = fp32_accumulation_bound(terms, 2 * n_in + 1)
    got = out.to_numpy().reshape(rows, n_out)
    assert np.all(np.abs(got - ref) <= bound), np.max(np.abs(got - ref) / bound)


# -- bounds -----------------------------------------------------------------


def an_operand_too_small_is_rejected_before_launch():
    small = device.empty(10)
    with pytest.raises(ValueError, match="out holds 10"):
        device.rmsnorm(put(normal(2, 8)), put(normal(8)), small, 2, 8, 1e-6)


def a_view_cannot_reach_past_its_tensor():
    t = device.empty(10)
    with pytest.raises(ValueError, match="view"):
        device.view(t, 8, 3)


def logits_refuse_scratch_too_small_for_them():
    hidden, vocab = 8, 100
    small = device.empty(device.scratch_elements(1, vocab) - 1)
    with pytest.raises(ValueError, match="scratch"):
        device.logits(put(normal(1, hidden)), put(normal(vocab, hidden)), small, 0, 1, hidden, vocab)


def a_negative_first_row_is_refused_rather_than_wrapped():
    """first_row + rows can be positive while first_row is not; the pointer
    offset computed from it would then wrap to anywhere."""
    hidden, vocab = 8, 100
    x, head = put(normal(4, hidden)), put(normal(vocab, hidden))
    for call in (device.logits, device.greedy):
        with pytest.raises(ValueError, match="first_row"):
            call(x, head, scratch(3, vocab), -1, 3, hidden, vocab)


def greedy_refuses_a_row_with_no_finite_logit():
    """No argmax exists; the kernel's answer is one past the vocabulary, and it
    must not reach the caller as a token."""
    hidden, vocab = 8, 100
    head = put(np.full((vocab, hidden), np.nan, np.float32))
    with pytest.raises(RuntimeError, match="no finite logit"):
        device.greedy(put(normal(1, hidden)), head, scratch(1, vocab), 0, 1, hidden, vocab)


def rope_refuses_to_write_over_its_input():
    x = put(normal(1, 1, 8))
    with pytest.raises(ValueError, match="over its input"):
        device.rope(x, device.index(np.zeros(1, np.int32)), x, 1, 1, 8, 1e4)


def swiglu_refuses_to_write_over_its_input():
    gate, up = put(normal(8)), put(normal(8))
    with pytest.raises(ValueError, match="over its input"):
        device.swiglu(gate, up, gate, 8)


def attention_refuses_more_queries_than_keys():
    q = put(normal(3, 1, 8))
    kv = put(normal(2, 1, 8))
    with pytest.raises(ValueError, match="seq_q"):
        device.attention(q, kv, kv, None, None, device.empty(24), 3, 2, 1, 1, 8)


# -- the tests: each check above, for both models -----------------------------


def _named(case):
    check, *args = case
    return " ".join([check.__name__, *(getattr(a, "name", str(a)) for a in args)])


def test_the_device_surface_is_the_tested_kernels():
    """Each existing kernel reached from the device is bit-identical to the
    one the per-kernel tests hold to the gate, at both models' shapes; the
    linear with and without its bias. Attention reads the first seq_k rows of
    a larger cache, and a view writes only its own elements."""
    each([(check, cfg) for check in (rmsnorm_is_the_tested_kernel, rope_is_the_tested_kernel, swiglu_is_the_tested_kernel, attention_reads_the_first_seq_k_rows_of_a_larger_cache, a_view_writes_only_its_own_elements) for cfg in CFGS]
         + [(linear_is_the_tested_kernel, cfg, bias) for cfg in CFGS for bias in (False, True)],
         lambda check, *args: check(*args), name=_named)


def test_the_kernels_new_to_the_device():
    """What each is, at both models' shapes: embed an exact gather; add one
    fp32 addition rounded once, in place too; logits fp32 and within fp32
    accumulation of float64; greedy the argmax of the logits, ties to the
    lowest index. And the fp32 residual stream (ADR-0010): the embedding
    widens exactly; RMSNorm of fp32 is the tested kernel on fp16-exact input
    and meets the gate on input fp16 cannot hold; a projection accumulates
    into fp32 without rounding."""
    each([(check, cfg) for check in (embed_is_an_exact_gather, add_is_one_fp32_addition_rounded_once_and_may_update_in_place, logits_are_fp32_and_within_fp32_accumulation_of_float64, greedy_is_the_argmax_of_the_logits, greedy_breaks_ties_to_the_lowest_index, the_embedding_widens_exactly_to_fp32, rmsnorm_of_fp32_is_the_tested_kernel_on_fp16_exact_input, rmsnorm_of_fp32_meets_the_gate_on_input_fp16_cannot_hold, a_projection_accumulates_into_fp32_without_rounding) for cfg in CFGS],
         lambda check, cfg: check(cfg), name=_named)


def test_bounds_are_checked_before_a_launch():
    """With literals, since what they check is a shape the model never has:
    an operand too small, a view past its tensor, logits' scratch too small,
    a negative first row, a row with no finite logit, RoPE or SwiGLU writing
    over their input, more queries than keys."""
    each([(check,) for check in (an_operand_too_small_is_rejected_before_launch, a_view_cannot_reach_past_its_tensor, logits_refuse_scratch_too_small_for_them, a_negative_first_row_is_refused_rather_than_wrapped, greedy_refuses_a_row_with_no_finite_logit, rope_refuses_to_write_over_its_input, swiglu_refuses_to_write_over_its_input, attention_refuses_more_queries_than_keys)], lambda check: check(), name=_named)
