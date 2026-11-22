"""Model configuration, verified rather than assumed.

ADR-0003 recorded its constants from prior knowledge and said in as many words
that they must be checked against each model's `config.json` before a kernel is
written against them. A silent mismatch would present as a numerical bug, which
is an expensive way to discover a typo.
"""

import json

import pytest

from microinfer.config import ConfigMismatch, ModelConfig
from microinfer.models import VERIFIED, verified_card


def test_both_adr_0003_models_are_recorded_with_the_constants_expected():
    """The card is a copy of the upstream config.json; the expectation is
    written out separately. They must agree, or one of them is a typo."""
    assert set(VERIFIED) == {"qwen2.5-0.5b-instruct", "qwen2.5-1.5b-instruct"}
    for name in sorted(VERIFIED):
        ModelConfig.from_card(name).verify(VERIFIED[name])


def test_derived_head_dim_matches_hidden_over_heads():
    cfg = ModelConfig.from_card("qwen2.5-0.5b-instruct")
    assert cfg.head_dim == cfg.hidden_size // cfg.num_attention_heads == 64


def test_the_two_models_share_a_kernel_shape():
    """ADR-0003 chose one family so that one parameterised kernel serves both.
    If these ever diverge, that reasoning is void."""
    small = ModelConfig.from_card("qwen2.5-0.5b-instruct")
    large = ModelConfig.from_card("qwen2.5-1.5b-instruct")
    assert small.num_key_value_heads == large.num_key_value_heads == 2
    assert small.rope_theta == large.rope_theta
    assert small.rope_scaling is large.rope_scaling is None
    assert small.head_dim != large.head_dim  # 64 vs 128 — nothing may hardcode it


def test_a_mismatch_names_the_field_the_expected_and_the_actual(tmp_path):
    cfg = ModelConfig.from_card("qwen2.5-0.5b-instruct")
    wrong = dict(VERIFIED["qwen2.5-0.5b-instruct"], num_hidden_layers=999)
    with pytest.raises(ConfigMismatch) as exc:
        cfg.verify(wrong)
    message = str(exc.value)
    assert "num_hidden_layers" in message
    assert "999" in message
    assert "24" in message


def test_a_mismatch_reports_every_bad_field_not_only_the_first():
    cfg = ModelConfig.from_card("qwen2.5-1.5b-instruct")
    wrong = dict(VERIFIED["qwen2.5-1.5b-instruct"], num_hidden_layers=1, hidden_size=2)
    with pytest.raises(ConfigMismatch) as exc:
        cfg.verify(wrong)
    assert "num_hidden_layers" in str(exc.value)
    assert "hidden_size" in str(exc.value)


def test_weights_are_stored_as_bfloat16():
    """Not fp16. The engine's kernels are fp16 throughout, so every load crosses
    a format boundary; recording it here keeps the assumption from going quiet
    again."""
    for name in VERIFIED:
        assert ModelConfig.from_card(name).torch_dtype == "bfloat16"


def test_qwen25_has_no_rope_scaling():
    """ADR-0003 chose this family partly to avoid llama3 rope scaling, which is
    an error-prone surface in a hand-written kernel."""
    for name in VERIFIED:
        assert ModelConfig.from_card(name).rope_scaling is None


def test_card_is_a_faithful_copy_of_upstream():
    """The card is committed so the build does not depend on the network. It is
    only trustworthy if nothing has been edited into it."""
    raw = json.loads(verified_card("qwen2.5-0.5b-instruct").read_text())
    assert raw["architectures"] == ["Qwen2ForCausalLM"]
    assert raw["vocab_size"] == 151936


# -- a second family (#209) ----------------------------------------------------

#: Llama-3.2-1B-Instruct's config.json as ADR-0003 expects it, until its card
#: is committed with the checkpoint; the card then replaces this copy.
LLAMA = {
    "architectures": ["LlamaForCausalLM"], "attention_bias": False, "head_dim": 64,
    "hidden_act": "silu", "hidden_size": 2048, "intermediate_size": 8192,
    "max_position_embeddings": 131072, "mlp_bias": False, "model_type": "llama",
    "num_attention_heads": 32, "num_hidden_layers": 16, "num_key_value_heads": 8,
    "rms_norm_eps": 1e-05, "rope_theta": 500000.0, "tie_word_embeddings": True,
    "rope_scaling": {"factor": 32.0, "high_freq_factor": 4.0, "low_freq_factor": 1.0,
                     "original_max_position_embeddings": 8192, "rope_type": "llama3"},
    "torch_dtype": "bfloat16", "vocab_size": 128256,
}


def test_qwen2_carries_a_qkv_bias_and_llama_says_whether_it_does():
    """Qwen2's config has no attention_bias: the architecture always has one.
    Llama's says, and Llama-3.2's has none, so ADR-0009's bias path goes
    unused there."""
    qwen = ModelConfig.from_card("qwen2.5-0.5b-instruct")
    assert qwen.architecture == "Qwen2ForCausalLM" and qwen.attention_bias is True
    llama = ModelConfig.from_dict("llama-3.2-1b-instruct", LLAMA)
    assert llama.architecture == "LlamaForCausalLM" and llama.attention_bias is False
    assert llama.head_dim == 64 and llama.kv_bytes_per_token == 32 * 1024
    assert llama.rope_scaling["rope_type"] == "llama3"


def test_an_architecture_the_engine_does_not_run_is_refused():
    with pytest.raises(ConfigMismatch, match="GemmaForCausalLM"):
        ModelConfig.from_dict("gemma", dict(LLAMA, architectures=["GemmaForCausalLM"]))


def test_a_stated_head_dim_other_than_hidden_over_heads_is_refused():
    """The engine sizes the projections by hidden_size / num_attention_heads;
    a model whose head_dim says otherwise is not the model it would run."""
    with pytest.raises(ConfigMismatch, match="head_dim"):
        ModelConfig.from_dict("odd", dict(LLAMA, head_dim=128))


def test_rope_scaling_other_than_llama3_is_refused():
    """Only llama3 scaling is implemented (rope_scaling.h); any other kind
    would run as though it were plain RoPE, which is a silent error. The
    older spelling, `type`, is read as `rope_type` is."""
    for scaling in ({"rope_type": "linear", "factor": 2.0}, {"type": "dynamic", "factor": 2.0},
                    {"rope_type": "yarn", "factor": 4.0}):
        with pytest.raises(ConfigMismatch, match="rope_scaling"):
            ModelConfig.from_dict("scaled", dict(LLAMA, rope_scaling=scaling))
