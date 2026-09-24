"""Loading a model and knowing what it cost.

The arithmetic tests here need no checkpoint: they check that the figures
ADR-0003 was argued from can be re-derived from config.json alone. The tests
against a real checkpoint skip when it is absent.
"""

import gc
import sys
from pathlib import Path

import numpy as np
import pytest
from test_weights import write_safetensors

from conftest import require_model, stable_free_bytes
from microinfer import (ConfigMismatch, Engine, ModelConfig, _microinfer, expected_weight_bytes,
                        expected_weight_shapes, kv_cache_bytes, nvml, weight_layout)
from microinfer.weights import WeightError, describe

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


def test_raw_cache_bytes_scale_linearly_with_element_size():
    """Element size is the one term the engine can renegotiate mid-session.

    These are *raw* bytes with scale metadata excluded, which is why the ratios
    are clean powers of two. They are not compression ratios: ADR-0005 puts
    INT4 at 4.63 effective bits once its 1280 bytes per page are counted, and
    CONTRIBUTING forbids quoting 4x for it.
    """
    c = cfg("qwen2.5-1.5b-instruct")
    fp16 = kv_cache_bytes(c, 32768, bytes_per_element=2)
    assert kv_cache_bytes(c, 32768, bytes_per_element=1) == fp16 // 2
    assert kv_cache_bytes(c, 32768, bytes_per_element=0.5) == fp16 // 4
    assert isinstance(kv_cache_bytes(c, 32768, bytes_per_element=0.5), int)


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
    fp = engine.footprint()

    assert fp.weights > 0
    assert fp.kv_cache == 0  # nothing allocates a cache yet; #14 will
    assert fp.workspace == 0
    assert fp.engine_total == fp.weights + fp.kv_cache + fp.workspace
    assert fp.device_total > fp.device_free > 0
    assert "weights" in fp.render() and "MiB" in fp.render()


def test_unaccounted_stays_positive_and_does_not_move_with_a_projection():
    """`unaccounted` is measurement minus measurement.

    An earlier version folded the projected cache into the same object, so this
    number shrank as the projection grew and went negative past a large enough
    context — a figure whose whole job is to separate causes, mixing them.
    """
    engine = Engine(require_model("qwen2.5-0.5b-instruct"))
    engine.load_weights()
    before = engine.footprint().unaccounted
    assert before > 0

    engine.project_kv_cache(32768)  # a projection must not touch the account
    assert engine.footprint().unaccounted == pytest.approx(before, rel=0.05)


def test_reported_weights_match_what_the_driver_says_was_taken():
    """Turns a self-report into a measurement.

    `footprint.weights` sums what the engine believes it uploaded. This checks
    that belief against what the driver took for this process across the
    load: its own account (nvml.own_used_bytes), which no other process moves,
    where device free memory drifts by more than the bound checked here.

    The driver may round up and never down, so the claim can only understate.
    It understated by 27% while each of the 290 tensors had an allocation of
    its own (#5), because the driver's overhead tracks the number of
    allocations rather than their size. In one arena (#23) it is one
    allocation, rounded to the driver's granularity: under 2%.
    """
    engine = Engine(require_model("qwen2.5-0.5b-instruct"))
    gc.collect()
    before = nvml.own_used_bytes()
    engine.load_weights()
    taken = nvml.own_used_bytes() - before
    claimed = engine.footprint().weights

    assert taken >= claimed
    assert taken < claimed * 1.02, (
        f"driver took {taken / MIB:.1f} MiB against a claim of "
        f"{claimed / MIB:.1f} MiB: more overhead than one allocation's granularity "
        f"explains")


@pytest.mark.parametrize("name", ["qwen2.5-0.5b-instruct", "qwen2.5-1.5b-instruct"])
def test_config_implies_exactly_the_checkpoints_tensors(name):
    """The arena is laid out from config.json before the checkpoint is read, so
    config.json must name every tensor the checkpoint holds, with its shape,
    and nothing else."""
    path = require_model(name) / "model.safetensors"
    assert expected_weight_shapes(cfg(name)) == {i.name: i.shape for i in describe(path)}


def test_weights_occupy_one_allocation_sized_from_config():
    """One arena, of the size weight_layout computes from config.json; every
    weight inside it, at the offset the layout gave it, on a multiple of
    weight_alignment."""
    engine = Engine(require_model("qwen2.5-0.5b-instruct"))
    engine.load_weights()
    arena = engine.weight_arena
    offsets, size = weight_layout(engine.config)
    assert arena.nbytes == size
    for name, tensor in engine.tensors.items():
        at = tensor.address - arena.address
        assert at == offsets[name], name
        assert at % _microinfer.weight_alignment == 0 and at + tensor.nbytes <= size, name
    assert engine.footprint().weights == sum(t.nbytes for t in engine.tensors.values())


def test_freeing_the_engine_returns_the_whole_arena():
    """Asserted against cudaMemGetInfo, as ADR-0007's reclaim is, within what
    other processes move it by; and against this process's own account,
    exactly."""
    gc.collect()
    free_before, own_before = stable_free_bytes(), nvml.own_used_bytes()
    engine = Engine(require_model("qwen2.5-0.5b-instruct"))
    engine.load_weights()
    held = engine.weight_arena.nbytes
    assert free_before - stable_free_bytes() > 0.95 * held
    del engine
    gc.collect()
    assert abs(free_before - stable_free_bytes()) < 0.02 * held
    assert nvml.own_used_bytes() == own_before


def test_a_checkpoint_config_does_not_describe_is_refused_before_anything_is_allocated(tmp_path):
    """A tensor config.json does not imply, one it implies that is missing, and
    one of the wrong shape, all named; and no arena taken."""
    staged = stage_config(tmp_path)
    write_safetensors(staged / "model.safetensors", {
        "model.norm.weight": ("F32", np.ones(3, np.float32)),
        "model.rotary_emb.inv_freq": ("F32", np.ones(4, np.float32)),
    })
    engine = Engine(staged, verify=False)
    gc.collect()
    before = nvml.own_used_bytes()
    with pytest.raises(WeightError) as caught:
        engine.load_weights()
    message = str(caught.value)
    assert "model.embed_tokens.weight: missing" in message
    assert "model.rotary_emb.inv_freq: not implied by config.json" in message
    assert "model.norm.weight: shape (3,)" in message
    assert "nothing was allocated" in message
    assert nvml.own_used_bytes() == before and engine.weight_arena is None


def test_torch_is_absent_after_loading_a_model():
    """ADR-0002, checked at the one moment it would be most tempting to reach
    for transformers."""
    engine = Engine(require_model("qwen2.5-0.5b-instruct"))
    engine.load_weights()
    assert "torch" not in sys.modules
    assert "transformers" not in sys.modules
