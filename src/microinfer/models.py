"""The constants this repository was written against, verified upstream.

Every value below was read from the model's own `config.json` on 2026-09-22 and
matches what ADR-0003 assumed. The cards under `model_cards/` are committed
copies of those files; this module is the expectation checked against them, kept
separate on purpose — a copy that checks itself proves nothing.
"""

from __future__ import annotations

from pathlib import Path

from .config import CARDS

#: Development target and experimental target from ADR-0003. One parameterised
#: kernel serves both: same family, same RoPE, same two KV heads, same 32K
#: context — only `hidden_size` and the derived `head_dim` differ.
VERIFIED: dict[str, dict] = {
    "qwen2.5-0.5b-instruct": {
        "num_hidden_layers": 24,
        "hidden_size": 896,
        "num_attention_heads": 14,
        "num_key_value_heads": 2,
        "intermediate_size": 4864,
        "vocab_size": 151936,
        "max_position_embeddings": 32768,
        "rope_theta": 1000000.0,
        "rms_norm_eps": 1e-06,
        "tie_word_embeddings": True,
        "torch_dtype": "bfloat16",
        "hidden_act": "silu",
        "rope_scaling": None,
        "head_dim": 64,
    },
    "qwen2.5-1.5b-instruct": {
        "num_hidden_layers": 28,
        "hidden_size": 1536,
        "num_attention_heads": 12,
        "num_key_value_heads": 2,
        "intermediate_size": 8960,
        "vocab_size": 151936,
        "max_position_embeddings": 32768,
        "rope_theta": 1000000.0,
        "rms_norm_eps": 1e-06,
        "tie_word_embeddings": True,
        "torch_dtype": "bfloat16",
        "hidden_act": "silu",
        "rope_scaling": None,
        "head_dim": 128,
    },
}

#: Upstream repository for each card, so a reader can re-derive it.
UPSTREAM = {name: f"Qwen/{name.replace('qwen2.5', 'Qwen2.5').replace('b-instruct', 'B-Instruct')}"
            for name in VERIFIED}


def verified_card(name: str) -> Path:
    return CARDS / f"{name}.json"
