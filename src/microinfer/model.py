"""The Qwen2 forward pass, as a sequence of device kernel calls.

Python sequences and does no arithmetic (ADR-0002): every line below that
touches a tensor is one launch on `_microinfer.device`, and the only values
computed here are sizes and offsets.

The KV cache here is the simple one the Milestone 0 checkpoint asks for (#12):
one contiguous FP16 buffer per layer for keys and one for values, sized for the
whole sequence up front. Paging replaces it in #14, and must be proven
equivalent to it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import _microinfer
from .config import ModelConfig

device = _microinfer.device

#: Rows of logits computed per call. The LM head's fp32 output is 608 KiB a row
#: for Qwen2.5's vocabulary, so a whole 890-token prompt's worth would be
#: 541 MiB of device scratch at once. In chunks of 64 it is 39 MiB.
LOGIT_CHUNK_ROWS = 64


@dataclass(frozen=True)
class Weights:
    """The checkpoint's tensors on the device, by role."""

    embed: _microinfer.DeviceTensor
    layers: list[dict[str, _microinfer.DeviceTensor]]
    norm: _microinfer.DeviceTensor
    head: _microinfer.DeviceTensor

    @classmethod
    def from_tensors(cls, cfg: ModelConfig, tensors: dict) -> Weights:
        prefix = "model.layers."
        layers = [
            {name[len(f"{prefix}{i}."):]: t for name, t in tensors.items()
             if name.startswith(f"{prefix}{i}.")}
            for i in range(cfg.num_hidden_layers)
        ]
        embed = tensors["model.embed_tokens.weight"]
        # Qwen2.5-0.5B ties the LM head to the embedding and stores no
        # lm_head.weight; 1.5B does not tie.
        head = embed if cfg.tie_word_embeddings else tensors["lm_head.weight"]
        return cls(embed=embed, layers=layers, norm=tensors["model.norm.weight"], head=head)


class KVCache:
    """Contiguous FP16 keys and values, one buffer each per layer.

    Row t of a layer's buffer is token t's keys (or values) for every KV head,
    which is the layout attention reads: the first `length` rows are exactly its
    `(seq_k, kv_heads, head_dim)` operand.
    """

    def __init__(self, cfg: ModelConfig, capacity: int):
        self.capacity = capacity
        self.row = cfg.num_key_value_heads * cfg.head_dim
        self.keys = [device.empty(capacity * self.row) for _ in range(cfg.num_hidden_layers)]
        self.values = [device.empty(capacity * self.row) for _ in range(cfg.num_hidden_layers)]
        self.length = 0

    @property
    def nbytes(self) -> int:
        return sum(t.nbytes for t in self.keys + self.values)

    def rows(self, buffers: list, layer: int, start: int, count: int):
        """A view of `count` rows from `start`: where a step's keys or values go."""
        if start + count > self.capacity:
            raise ValueError(f"the cache holds {self.capacity} tokens; "
                             f"writing {count} at {start} would pass its end")
        return device.view(buffers[layer], start * self.row, count * self.row)


class Workspace:
    """Every activation buffer one step needs, for `rows` tokens at a time.

    Allocated once per step size and reused by every layer, so a forward pass
    makes a handful of allocations rather than one per kernel call.
    """

    def __init__(self, cfg: ModelConfig, rows: int):
        self.rows = rows
        q_dim = cfg.num_attention_heads * cfg.head_dim
        kv_dim = cfg.num_key_value_heads * cfg.head_dim
        h, i = cfg.hidden_size, cfg.intermediate_size
        self.residual = device.empty(rows * h)
        self.normed = device.empty(rows * h)
        self.q_raw = device.empty(rows * q_dim)
        self.q = device.empty(rows * q_dim)
        self.k_raw = device.empty(rows * kv_dim)
        self.attended = device.empty(rows * q_dim)
        self.projected = device.empty(rows * h)
        self.gate = device.empty(rows * i)
        self.up = device.empty(rows * i)
        self.activated = device.empty(rows * i)
        self.scratch = device.empty(
            device.scratch_elements(min(rows, LOGIT_CHUNK_ROWS), cfg.vocab_size))

    @property
    def nbytes(self) -> int:
        return sum(t.nbytes for t in vars(self).values() if isinstance(t, _microinfer.DeviceTensor))


def run(cfg: ModelConfig, weights: Weights, ws: Workspace, cache: KVCache,
        token_ids: np.ndarray, capture: list | None = None) -> None:
    """Feed `token_ids` through the model after the cache's current contents.

    Leaves the final-normed hidden states for these tokens in `ws.normed`, and
    extends the cache by them. With `capture`, appends each layer's residual
    stream to it as a host array, in HuggingFace's `output_hidden_states` order:
    the embedding output, the output of every layer but the last, and then the
    last layer's output *after the final norm*. That is the order the golden
    reference stores, so index i compares with index i.
    """
    n = len(token_ids)
    start = cache.length
    heads, kv_heads, hd = cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim
    h, inter = cfg.hidden_size, cfg.intermediate_size
    q_dim, kv_dim = heads * hd, kv_heads * hd
    eps, layers = cfg.rms_norm_eps, cfg.num_hidden_layers

    ids = device.index(token_ids)
    positions = device.index(np.arange(start, start + n, dtype=np.int32))

    def keep(tensor, width):
        if capture is not None:
            capture.append(tensor.to_numpy()[: n * width].reshape(n, width))

    device.embed(ids, weights.embed, ws.residual, h, cfg.vocab_size)
    keep(ws.residual, h)

    for i, w in enumerate(weights.layers):
        # Attention block. K is rotated straight into the cache, and V
        # projected straight into it: neither needs a buffer of its own.
        #
        # K goes into the cache *without* its bias (ADR-0009). Qwen2.5's key
        # bias reaches 147 against an input-dependent part of 5 to 11, and at
        # that magnitude fp16's ulp is as large as what distinguishes one key
        # from the next. Attention adds the bias back, rotated, in fp32.
        k_rows = cache.rows(cache.keys, i, start, n)
        v_rows = cache.rows(cache.values, i, start, n)
        device.rmsnorm(ws.residual, w["input_layernorm.weight"], ws.normed, n, h, eps)
        device.linear(ws.normed, w["self_attn.q_proj.weight"], w["self_attn.q_proj.bias"],
                      ws.q_raw, n, h, q_dim)
        device.linear(ws.normed, w["self_attn.k_proj.weight"], None, ws.k_raw, n, h, kv_dim)
        device.linear(ws.normed, w["self_attn.v_proj.weight"], w["self_attn.v_proj.bias"],
                      v_rows, n, h, kv_dim)
        device.rope(ws.q_raw, positions, ws.q, n, heads, hd, cfg.rope_theta)
        device.rope(ws.k_raw, positions, k_rows, n, kv_heads, hd, cfg.rope_theta)
        device.attention(ws.q, cache.keys[i], cache.values[i], w["self_attn.k_proj.bias"],
                         ws.attended, n, start + n, heads, kv_heads, hd, cfg.rope_theta)
        device.linear(ws.attended, w["self_attn.o_proj.weight"], None, ws.projected, n, q_dim, h)
        device.add(ws.residual, ws.projected, ws.residual, n * h)

        # MLP block.
        device.rmsnorm(ws.residual, w["post_attention_layernorm.weight"], ws.normed, n, h, eps)
        device.linear(ws.normed, w["mlp.gate_proj.weight"], None, ws.gate, n, h, inter)
        device.linear(ws.normed, w["mlp.up_proj.weight"], None, ws.up, n, h, inter)
        device.swiglu(ws.gate, ws.up, ws.activated, n * inter)
        device.linear(ws.activated, w["mlp.down_proj.weight"], None, ws.projected, n, inter, h)
        device.add(ws.residual, ws.projected, ws.residual, n * h)

        if i < layers - 1:
            keep(ws.residual, h)

    device.rmsnorm(ws.residual, weights.norm, ws.normed, n, h, eps)
    keep(ws.normed, h)
    cache.length = start + n


def logits(cfg: ModelConfig, weights: Weights, ws: Workspace, n: int) -> np.ndarray:
    """fp32 logits for the n rows in `ws.normed`, in chunks through the scratch."""
    out = np.empty((n, cfg.vocab_size), dtype=np.float32)
    for first in range(0, n, LOGIT_CHUNK_ROWS):
        rows = min(LOGIT_CHUNK_ROWS, n - first)
        out[first:first + rows] = device.logits(ws.normed, weights.head, ws.scratch, first,
                                                rows, cfg.hidden_size, cfg.vocab_size)
    return out


def greedy_last(cfg: ModelConfig, weights: Weights, ws: Workspace, n: int) -> int:
    """The argmax token after the last of the n rows in `ws.normed`."""
    return int(device.greedy(ws.normed, weights.head, ws.scratch, n - 1, 1,
                             cfg.hidden_size, cfg.vocab_size)[0])
