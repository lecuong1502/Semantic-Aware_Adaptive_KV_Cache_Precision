#!/usr/bin/env python3
"""How much a prompt amplifies small errors, measured in float64 (#36).

    .venv/bin/python tools/conditioning.py adversarial-00 medium-01

For each prompt, runs the model in float64 with NumPy, three times: once
exactly, and twice with the embedding perturbed by random relative noise the
size of one ulp of fp16 (2^-11) and of fp32 (2^-24). The reported gain is how
far the worst hidden state moves, relative to how far the embedding was moved.

A gain near 1 means the model carries an error forward at the size it was
made. A large gain means the prompt is ill-conditioned: an engine whose
activations are rounded at fp16 cannot follow the reference on it, whichever
rounding is removed, because every one of them is amplified. ADR-0010 has
the reasoning.

NumPy only. No torch, so this runs in the engine's environment; it imports
nothing of the engine but the config and weight readers. Appends one entry
per prompt to the benchmark log. Slow by design: a whole model in float64 on
the CPU, three times per prompt, about a minute each for the 0.5B model.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from microinfer import benchlog  # noqa: E402
from microinfer.config import ModelConfig  # noqa: E402
from microinfer.golden import GoldenSet  # noqa: E402
from microinfer.weights import read_tensor  # noqa: E402

PERTURBATIONS = {"fp16_ulp": 2.0**-11, "fp32_ulp": 2.0**-24}
SEED = 0


def forward64(cfg: ModelConfig, checkpoint: Path, ids: np.ndarray, embed_noise: float) -> np.ndarray:
    """Hidden states (layers + 1, seq, hidden) in float64, in HuggingFace's
    order: embedding, every layer's output, the last after the final norm.
    The embedding is scaled by (1 + embed_noise * N(0, 1)) per element."""
    n = len(ids)
    heads, kv_heads, hd = cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim
    group = heads // kv_heads
    angle = np.arange(n)[:, None] * cfg.rope_theta ** (-np.arange(hd // 2) * 2.0 / hd)
    cos, sin = np.cos(angle)[:, None, :], np.sin(angle)[:, None, :]

    def rope(x):
        a, b = x[..., : hd // 2], x[..., hd // 2:]
        return np.concatenate([a * cos - b * sin, b * cos + a * sin], -1)

    def rms(x, w):
        return x / np.sqrt((x * x).mean(-1, keepdims=True) + cfg.rms_norm_eps) * w

    mask = np.triu(np.full((n, n), -np.inf), 1)
    h = read_tensor(checkpoint, "model.embed_tokens.weight", rows=ids).astype(np.float64)
    h = h * (1 + embed_noise * np.random.default_rng(SEED).standard_normal(h.shape))
    states = [h]
    for layer in range(cfg.num_hidden_layers):
        def w(name):
            return read_tensor(checkpoint, f"model.layers.{layer}.{name}").astype(np.float64)

        x = rms(h, w("input_layernorm.weight"))
        q = rope((x @ w("self_attn.q_proj.weight").T + w("self_attn.q_proj.bias")).reshape(n, heads, hd))
        k = rope((x @ w("self_attn.k_proj.weight").T + w("self_attn.k_proj.bias")).reshape(n, kv_heads, hd))
        v = (x @ w("self_attn.v_proj.weight").T + w("self_attn.v_proj.bias")).reshape(n, kv_heads, hd)
        s = np.einsum("qhd,khd->hqk", q, np.repeat(k, group, 1)) / np.sqrt(hd) + mask
        s = np.exp(s - s.max(-1, keepdims=True))
        s /= s.sum(-1, keepdims=True)
        attended = np.einsum("hqk,khd->qhd", s, np.repeat(v, group, 1)).reshape(n, heads * hd)
        h = h + attended @ w("self_attn.o_proj.weight").T
        x = rms(h, w("post_attention_layernorm.weight"))
        gate = x @ w("mlp.gate_proj.weight").T
        h = h + (gate / (1 + np.exp(-gate)) * (x @ w("mlp.up_proj.weight").T)) @ w("mlp.down_proj.weight").T
        last = layer == cfg.num_hidden_layers - 1
        states.append(rms(h, read_tensor(checkpoint, "model.norm.weight").astype(np.float64)) if last else h)
    return np.stack(states)


def relative(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Per state and position: |a - b| / |b|."""
    return np.linalg.norm(a - b, axis=-1) / np.linalg.norm(b, axis=-1)


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("prompts", nargs="+")
    parser.add_argument("--model", default="qwen2.5-0.5b-instruct")
    args = parser.parse_args(argv)

    cfg = ModelConfig.from_card(args.model)
    checkpoint = REPO / "models" / args.model / "model.safetensors"
    golden = GoldenSet(REPO / "tests" / "golden" / args.model)
    for prompt in args.prompts:
        ids = golden[prompt].token_ids
        exact = forward64(cfg, checkpoint, ids, 0.0)
        results = {}
        for name, size in PERTURBATIONS.items():
            moved = relative(forward64(cfg, checkpoint, ids, size), exact)
            results[name] = {"embedding": float(moved[0].max()),
                             "worst_state": float(moved[1:].max()),
                             "gain": float(moved[1:].max() / moved[0].max())}
        print(prompt, results)
        benchlog.append("conditioning", model=args.model, context_length=len(ids),
                        precision_tiers=None,
                        config={"prompt": prompt, "arithmetic": "float64, NumPy",
                                "perturbation": "embedding * (1 + size * N(0, 1))",
                                "seed": SEED, "sizes": PERTURBATIONS, "issue": 36},
                        results=results)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
