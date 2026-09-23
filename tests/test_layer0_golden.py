"""Decoder layer 0, built from the engine's own kernels, against the reference.

The first check against golden tensors rather than a synthetic reference (#9).
What it is really after is grouped-query attention on the real model, but the
reference does not store an attention output: it stores the residual stream
after each layer (`tools/gen_golden.py`). So the whole of layer 0 is composed
here — RMSNorm, the q/k/v projections with their biases, RoPE, attention, the
output projection, the residual, and the MLP — and compared where the
reference has something to compare with: `hidden_states[1]`, at the sampled
positions.

That makes this a test of every kernel at once, which is weaker than it sounds
for any one of them and stronger than it sounds for the mapping. Measured at
the time of writing, every prompt reaches a cosine similarity of 0.9998 or
better against the 0.999 threshold. Reading KV heads in interleaved order
instead of HuggingFace's contiguous one drops it to between 0.01 and 0.91, so
the threshold would notice the mistake GQA is most likely to make, and
`test_the_threshold_rejects_interleaved_kv_heads` keeps it that way.

The layer's embedding input is looked up from the checkpoint, not read from
`hidden_states[0]`, because attention needs every position and the reference
keeps only a sample. Where the two overlap they agree exactly: the lookup is a
widening of bf16, with no arithmetic in it.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

from conftest import require_model
from microinfer import _microinfer
from microinfer.config import ModelConfig
from microinfer.golden import GoldenError, GoldenSet
from microinfer.models import VERIFIED
from microinfer.weights import read_tensor

MODELS = sorted(VERIFIED)
GOLDEN = Path(__file__).resolve().parent / "golden"

#: The ticket's own number: cosine similarity per position.
MIN_COSINE = 0.999

LAYER_0 = [
    "input_layernorm.weight",
    "self_attn.q_proj.weight", "self_attn.q_proj.bias",
    "self_attn.k_proj.weight", "self_attn.k_proj.bias",
    "self_attn.v_proj.weight", "self_attn.v_proj.bias",
    "self_attn.o_proj.weight",
    "post_attention_layernorm.weight",
    "mlp.gate_proj.weight", "mlp.up_proj.weight", "mlp.down_proj.weight",
]


@dataclass
class Layer0:
    cfg: ModelConfig
    checkpoint: Path
    w: dict[str, np.ndarray]

    def embed(self, token_ids: np.ndarray) -> np.ndarray:
        return read_tensor(self.checkpoint, "model.embed_tokens.weight", rows=token_ids)

    def forward(self, h: np.ndarray, kv_order: np.ndarray | None = None) -> np.ndarray:
        """h: (seq, hidden), the embeddings of a whole prompt from position 0.

        `kv_order`, when given, expands k and v to one head per query head in
        that order before attention, so a test can stand in a wrong mapping
        without a wrong kernel."""
        cfg, w = self.cfg, self.w
        seq = len(h)
        heads, kv_heads, head_dim = cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim
        positions = np.arange(seq, dtype=np.int32)

        x = _microinfer.rmsnorm(h, w["input_layernorm.weight"], cfg.rms_norm_eps)

        def project(name, n_heads):
            y = _microinfer.linear(x, w[f"self_attn.{name}.weight"], w[f"self_attn.{name}.bias"])
            return y.reshape(seq, n_heads, head_dim)

        q = _microinfer.rope(project("q_proj", heads), positions, cfg.rope_theta)
        k = _microinfer.rope(project("k_proj", kv_heads), positions, cfg.rope_theta)
        v = project("v_proj", kv_heads)
        if kv_order is not None:
            k, v = np.ascontiguousarray(k[:, kv_order]), np.ascontiguousarray(v[:, kv_order])

        attended = _microinfer.attention(q, k, v).reshape(seq, heads * head_dim)
        h = h + _microinfer.linear(attended, w["self_attn.o_proj.weight"])

        x = _microinfer.rmsnorm(h, w["post_attention_layernorm.weight"], cfg.rms_norm_eps)
        gated = _microinfer.swiglu(_microinfer.linear(x, w["mlp.gate_proj.weight"]),
                                   _microinfer.linear(x, w["mlp.up_proj.weight"]))
        return h + _microinfer.linear(gated, w["mlp.down_proj.weight"])


@pytest.fixture(scope="module", params=MODELS)
def layer0(request) -> Layer0:
    name = request.param
    checkpoint = require_model(name) / "model.safetensors"
    w = {n: read_tensor(checkpoint, f"model.layers.0.{n}") for n in LAYER_0}
    return Layer0(ModelConfig.from_card(name), checkpoint, w)


def golden_with_hidden_states(cfg: ModelConfig):
    try:
        golden = GoldenSet(GOLDEN / cfg.name)
    except GoldenError as exc:
        pytest.skip(str(exc))
    items = [item for item in golden if item.hidden_states is not None]
    assert items, "no prompt carries hidden states; regenerate the reference"
    return items


def cosine(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a, b = a.astype(np.float64), b.astype(np.float64)
    return (a * b).sum(-1) / (np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1))


def test_the_embedding_lookup_is_the_reference_s_layer_input(layer0):
    for item in golden_with_hidden_states(layer0.cfg):
        h = layer0.embed(item.token_ids)
        np.testing.assert_array_equal(h[item.logit_positions], item.hidden_states[0])


def test_layer_0_matches_the_reference_at_every_sampled_position(layer0):
    """Every prompt that carries hidden states: the short ones, and the
    adversarial ones up to 890 tokens, so the causal tiles are crossed many
    times over."""
    worst = {}
    for item in golden_with_hidden_states(layer0.cfg):
        got = layer0.forward(layer0.embed(item.token_ids))
        worst[item.prompt_id] = cosine(got[item.logit_positions], item.hidden_states[1]).min()
    failing = {k: v for k, v in worst.items() if not v > MIN_COSINE}
    assert not failing, f"cosine similarity at or below {MIN_COSINE}: {failing}"


def test_the_threshold_rejects_interleaved_kv_heads(layer0):
    """Query head h reading KV head h % kv_heads, the plausible mistake, must
    fail the same comparison, or the test above is not measuring the mapping."""
    cfg = layer0.cfg
    interleaved = np.arange(cfg.num_attention_heads) % cfg.num_key_value_heads
    for item in golden_with_hidden_states(cfg):
        got = layer0.forward(layer0.embed(item.token_ids), kv_order=interleaved)
        assert cosine(got[item.logit_positions], item.hidden_states[1]).min() < MIN_COSINE, item.prompt_id
