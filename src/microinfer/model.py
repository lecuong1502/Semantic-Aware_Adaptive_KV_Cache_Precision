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

from dataclasses import dataclass, fields

import numpy as np

from . import _microinfer
from .config import ModelConfig

device = _microinfer.device
Tensor = _microinfer.DeviceTensor

#: Rows of logits computed per call. The LM head's fp32 output is 608 KiB a row
#: for Qwen2.5's vocabulary, so a whole 890-token prompt's worth would be
#: 541 MiB of device scratch at once. In chunks of 64 it is 39 MiB.
LOGIT_CHUNK_ROWS = 64


@dataclass(frozen=True)
class LayerWeights:
    """One decoder layer's tensors, each under the checkpoint's own name with
    the `model.layers.<i>.` prefix removed and dots as underscores. Built when
    the weights are loaded, so a missing or misspelt tensor fails there, not
    halfway through a forward pass."""

    input_layernorm_weight: Tensor
    self_attn_q_proj_weight: Tensor
    self_attn_q_proj_bias: Tensor
    self_attn_k_proj_weight: Tensor
    self_attn_k_proj_bias: Tensor
    self_attn_v_proj_weight: Tensor
    self_attn_v_proj_bias: Tensor
    self_attn_o_proj_weight: Tensor
    post_attention_layernorm_weight: Tensor
    mlp_gate_proj_weight: Tensor
    mlp_up_proj_weight: Tensor
    mlp_down_proj_weight: Tensor

    @classmethod
    def from_tensors(cls, tensors: dict[str, Tensor], layer: int) -> LayerWeights:
        prefix = f"model.layers.{layer}."
        found = {name[len(prefix):].replace(".", "_"): t
                 for name, t in tensors.items() if name.startswith(prefix)}
        wanted = {f.name for f in fields(cls)}
        if missing := wanted - found.keys():
            raise KeyError(f"layer {layer} has no {', '.join(sorted(missing))}")
        return cls(**{name: found[name] for name in wanted})


@dataclass(frozen=True)
class Weights:
    """The checkpoint's tensors on the device, by role."""

    embed: Tensor
    layers: list[LayerWeights]
    norm: Tensor
    head: Tensor

    @classmethod
    def from_tensors(cls, cfg: ModelConfig, tensors: dict[str, Tensor]) -> Weights:
        embed = tensors["model.embed_tokens.weight"]
        # Qwen2.5-0.5B ties the LM head to the embedding and stores no
        # lm_head.weight; 1.5B does not tie.
        head = embed if cfg.tie_word_embeddings else tensors["lm_head.weight"]
        layers = [LayerWeights.from_tensors(tensors, i) for i in range(cfg.num_hidden_layers)]
        return cls(embed=embed, layers=layers, norm=tensors["model.norm.weight"], head=head)


class KVCache:
    """Contiguous FP16 keys and values, one buffer each per layer.

    Row t of a layer's buffer is token t's keys (or values) for every KV head,
    which is the layout attention reads: the first `length` rows are exactly its
    `(seq_k, kv_heads, head_dim)` operand. Row t is also position t, which
    attention relies on to rotate the key bias it adds back (ADR-0009).
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

    def key_rows(self, layer: int, start: int, count: int):
        """Where `count` tokens' keys go, from row `start`."""
        return self._rows(self.keys[layer], start, count)

    def value_rows(self, layer: int, start: int, count: int):
        return self._rows(self.values[layer], start, count)

    def _rows(self, buffer, start: int, count: int):
        if start + count > self.capacity:
            raise ValueError(f"the cache holds {self.capacity} tokens; "
                             f"writing {count} at {start} would pass its end")
        return device.view(buffer, start * self.row, count * self.row)


class Workspace:
    """Every activation buffer one step needs, for `rows` tokens at a time.

    Allocated once per step size and reused by every layer, so a forward pass
    makes a handful of allocations rather than one per kernel call.
    """

    def __init__(self, cfg: ModelConfig, rows: int):
        self.rows = rows
        q_width = cfg.num_attention_heads * cfg.head_dim
        kv_width = cfg.num_key_value_heads * cfg.head_dim
        hidden, intermediate = cfg.hidden_size, cfg.intermediate_size
        self.residual = device.empty(rows * hidden)
        self.normed = device.empty(rows * hidden)
        self.q_raw = device.empty(rows * q_width)
        self.q = device.empty(rows * q_width)
        self.k_raw = device.empty(rows * kv_width)
        self.attended = device.empty(rows * q_width)
        self.projected = device.empty(rows * hidden)
        self.gate = device.empty(rows * intermediate)
        self.up = device.empty(rows * intermediate)
        self.activated = device.empty(rows * intermediate)
        self.scratch = device.empty(
            device.scratch_elements(min(rows, LOGIT_CHUNK_ROWS), cfg.vocab_size))

    @property
    def nbytes(self) -> int:
        return sum(t.nbytes for t in vars(self).values() if isinstance(t, Tensor))


@dataclass(frozen=True)
class Model:
    """A configuration and its weights on the device: what a forward pass runs."""

    cfg: ModelConfig
    weights: Weights

    def run(self, ws: Workspace, cache: KVCache, token_ids: np.ndarray,
            hidden_states: list | None = None) -> None:
        """Feed `token_ids` through the model after the cache's current contents.

        Leaves the final-normed hidden states for these tokens in `ws.normed`,
        and extends the cache by them. With `hidden_states`, appends each
        layer's residual stream to it as a host array, in HuggingFace's
        `output_hidden_states` order: the embedding output, the output of every
        layer but the last, and then the last layer's output *after the final
        norm*. That is the order the golden reference stores, so index i
        compares with index i.
        """
        cfg, n, start = self.cfg, len(token_ids), cache.length
        heads, kv_heads, head_dim = cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim
        hidden, intermediate = cfg.hidden_size, cfg.intermediate_size
        q_width, kv_width = heads * head_dim, kv_heads * head_dim
        eps = cfg.rms_norm_eps

        ids = device.index(token_ids)
        positions = device.index(np.arange(start, start + n, dtype=np.int32))

        def capture_state(tensor):
            if hidden_states is not None:
                hidden_states.append(tensor.to_numpy()[: n * hidden].reshape(n, hidden))

        device.embed(ids, self.weights.embed, ws.residual, hidden, cfg.vocab_size)
        capture_state(ws.residual)

        for i, w in enumerate(self.weights.layers):
            # Attention block. K is rotated straight into the cache, and V
            # projected straight into it: neither needs a buffer of its own.
            #
            # K goes into the cache *without* its bias (ADR-0009). Qwen2.5's key
            # bias reaches 147 against an input-dependent part of 5 to 11, and
            # at that magnitude fp16's ulp is as large as what distinguishes one
            # key from the next. Attention adds the bias back, rotated, in fp32.
            device.rmsnorm(ws.residual, w.input_layernorm_weight, ws.normed, n, hidden, eps)
            device.linear(ws.normed, w.self_attn_q_proj_weight, w.self_attn_q_proj_bias,
                          ws.q_raw, n, hidden, q_width)
            device.linear(ws.normed, w.self_attn_k_proj_weight, None, ws.k_raw, n, hidden, kv_width)
            device.linear(ws.normed, w.self_attn_v_proj_weight, w.self_attn_v_proj_bias,
                          cache.value_rows(i, start, n), n, hidden, kv_width)
            device.rope(ws.q_raw, positions, ws.q, n, heads, head_dim, cfg.rope_theta)
            device.rope(ws.k_raw, positions, cache.key_rows(i, start, n), n, kv_heads, head_dim,
                        cfg.rope_theta)
            device.attention(ws.q, cache.keys[i], cache.values[i], w.self_attn_k_proj_bias,
                             ws.attended, n, start + n, heads, kv_heads, head_dim, cfg.rope_theta)
            device.linear(ws.attended, w.self_attn_o_proj_weight, None, ws.projected,
                          n, q_width, hidden)
            device.add(ws.residual, ws.projected, ws.residual, n * hidden)

            # MLP block.
            device.rmsnorm(ws.residual, w.post_attention_layernorm_weight, ws.normed, n, hidden, eps)
            device.linear(ws.normed, w.mlp_gate_proj_weight, None, ws.gate, n, hidden, intermediate)
            device.linear(ws.normed, w.mlp_up_proj_weight, None, ws.up, n, hidden, intermediate)
            device.swiglu(ws.gate, ws.up, ws.activated, n * intermediate)
            device.linear(ws.activated, w.mlp_down_proj_weight, None, ws.projected,
                          n, intermediate, hidden)
            device.add(ws.residual, ws.projected, ws.residual, n * hidden)

            if i < cfg.num_hidden_layers - 1:
                capture_state(ws.residual)

        device.rmsnorm(ws.residual, self.weights.norm, ws.normed, n, hidden, eps)
        capture_state(ws.normed)
        cache.length = start + n

    def logits(self, ws: Workspace, n: int) -> np.ndarray:
        """fp32 logits for the n rows in `ws.normed`, in chunks through the scratch."""
        cfg = self.cfg
        out = np.empty((n, cfg.vocab_size), dtype=np.float32)
        for first in range(0, n, LOGIT_CHUNK_ROWS):
            rows = min(LOGIT_CHUNK_ROWS, n - first)
            out[first:first + rows] = device.logits(ws.normed, self.weights.head, ws.scratch,
                                                    first, rows, cfg.hidden_size, cfg.vocab_size)
        return out

    def greedy_last(self, ws: Workspace, n: int) -> int:
        """The argmax token after the last of the n rows in `ws.normed`."""
        return int(device.greedy(ws.normed, self.weights.head, ws.scratch, n - 1, 1,
                                 self.cfg.hidden_size, self.cfg.vocab_size)[0])
