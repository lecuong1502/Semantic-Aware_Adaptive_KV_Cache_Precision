"""Loading a model and knowing what it cost.

The arithmetic tests here need no checkpoint: they check that the figures
ADR-0003 was argued from can be re-derived from config.json alone. The tests
against a real checkpoint skip when it is absent.
"""

import sys
from pathlib import Path

import pytest

from conftest import require_model
from microinfer import ConfigMismatch, Engine, ModelConfig, expected_weight_bytes, kv_cache_bytes

MODELS = Path(__file__).resolve().parent.parent / "models"
MIB = 1024 * 1024


def cfg(name):
    return ModelConfig.from_card(name)


# -- arithmetic, no checkpoint needed ---------------------------------------

@pytest.mark.parametrize(
    "name,per_token",
    [("qwen2.5-0.5b-instruct", 12 * 1024), ("qwen2.5-1.5b-instruct", 28 * 1024)],
)
def test_kv_bytes_per_token_matches_adr_0003(name, per_token):
    """12 KiB and 28 KiB are the figures the whole model choice was argued
    from. If they are wrong, ADR-0003 is wrong."""
    assert cfg(name).kv_bytes_per_token == per_token
    assert kv_cache_bytes(cfg(name), 1) == per_token


def test_kv_cache_at_32k_matches_the_table_in_adr_0003():
    assert kv_cache_bytes(cfg("qwen2.5-1.5b-instruct"), 32768) == 896 * MIB
    assert kv_cache_bytes(cfg("qwen2.5-0.5b-instruct"), 32768) == 384 * MIB


def test_kv_cache_scales_with_precision():
    """The one term the engine can renegotiate mid-session."""
    c = cfg("qwen2.5-1.5b-instruct")
    fp16 = kv_cache_bytes(c, 32768, bytes_per_element=2)
    assert kv_cache_bytes(c, 32768, bytes_per_element=1) == fp16 // 2
    assert kv_cache_bytes(c, 32768, bytes_per_element=0.5) == fp16 // 4


def test_expected_weight_bytes_is_in_the_right_region():
    """Cross-checked against the published parameter counts: 0.49B and 1.54B,
    two bytes each. Tight enough to catch a missing term, loose enough not to
    depend on how upstream rounds."""
    small = expected_weight_bytes(cfg("qwen2.5-0.5b-instruct"))
    large = expected_weight_bytes(cfg("qwen2.5-1.5b-instruct"))
    assert 0.9e9 < small < 1.1e9
    assert 2.9e9 < large < 3.3e9


# -- configuration gate ------------------------------------------------------

CARD = Path(__file__).resolve().parent.parent / "src/microinfer/model_cards"


def stage_config(tmp_path: Path, card: str = "qwen2.5-0.5b-instruct") -> Path:
    """A model directory whose name — and so its identity — is the temp dir's."""
    (tmp_path / "config.json").write_text((CARD / f"{card}.json").read_text())
    return tmp_path


def test_an_unknown_model_is_refused_by_name(tmp_path):
    """Identity comes from the directory name. An unrecognised one has no
    recorded constants to check against, and running blind is the failure this
    ticket exists to prevent."""
    with pytest.raises(ConfigMismatch, match="no verified constants"):
        Engine(stage_config(tmp_path))


def test_verification_can_be_waived_deliberately(tmp_path):
    engine = Engine(stage_config(tmp_path), verify=False)
    assert engine.config.num_hidden_layers == 24


# -- against the real checkpoint --------------------------------------------

@pytest.mark.parametrize("name", ["qwen2.5-0.5b-instruct", "qwen2.5-1.5b-instruct"])
def test_loads_and_reports_weights_within_one_percent_of_config(name):
    engine = Engine(require_model(name))
    engine.load_weights()

    measured = engine.footprint().weights
    predicted = engine.expected_weight_bytes()
    error = abs(measured - predicted) / predicted

    assert error < 0.01, (
        f"measured {measured / MIB:.1f} MiB on device, config.json implies "
        f"{predicted / MIB:.1f} MiB — {error:.2%} apart"
    )


def test_footprint_separates_weights_cache_and_workspace():
    engine = Engine(require_model("qwen2.5-0.5b-instruct"))
    engine.load_weights()
    fp = engine.footprint(context_length=8192)

    assert fp.weights > 0
    assert fp.kv_cache == kv_cache_bytes(engine.config, 8192)
    assert fp.workspace == 0  # nothing allocates workspace yet
    assert fp.engine_total == fp.weights + fp.kv_cache + fp.workspace
    assert fp.device_total > fp.device_free > 0
    assert "weights" in fp.render() and "MiB" in fp.render()


def test_torch_is_absent_after_loading_a_model():
    """ADR-0002, checked at the one moment it would be most tempting to reach
    for transformers."""
    engine = Engine(require_model("qwen2.5-0.5b-instruct"))
    engine.load_weights()
    assert "torch" not in sys.modules
    assert "transformers" not in sys.modules
