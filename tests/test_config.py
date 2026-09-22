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


def test_both_adr_0003_models_are_recorded():
    assert set(VERIFIED) == {"qwen2.5-0.5b-instruct", "qwen2.5-1.5b-instruct"}


@pytest.mark.parametrize("name", sorted(VERIFIED))
def test_recorded_card_matches_the_expected_constants(name):
    """The card is a copy of the upstream config.json; the expectation is
    written out separately. They must agree, or one of them is a typo."""
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
