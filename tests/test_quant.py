"""Quantisation of a page at INT8, INT4 and INT2, and the layout they share
(#16, #17), at Seam B.

Keys are quantised per (head, channel) across the page's P positions, values
per (head, token), both asymmetric (ADR-0005). The layout, and its 1280 bytes
of scale metadata on the 1.5B model, is every quantised tier's; below 8 bits
codes are packed, 8/bits to a byte, the first in the lowest bits. Every test
of the quantiser runs at every tier, with nothing special-cased.

The kernel is held to a float64 NumPy reference in two ways that do not
overlap. Its codes and metadata are the reference's own, exactly, but for a
code whose unrounded value sits within a hair of a rounding tie. And its
dequantised output is the reference's q * scale + zero to within ADR-0006's
per-kernel gate. What quantisation itself loses is a third thing, and not a
kernel tolerance: each element comes back within half a step of what went in,
a bound derived below from the scale the page stores. That bound is asserted
on random pages and on real keys and values from the golden prompts; the
error it allows is measured and logged by tools/quant_roundtrip.py, and it
grows as the tiers narrow: INT8 < INT4 < INT2.
"""

from pathlib import Path

import numpy as np
import pytest

from conftest import require_model
from microinfer import Engine, _microinfer
from microinfer.config import ModelConfig
from microinfer.golden import GoldenError, GoldenSet
from ulp_gate import assert_within_gate, fp16_exact

REPO = Path(__file__).resolve().parent.parent
Tier = _microinfer.Tier
P = _microinfer.device.page_tokens
INT8 = Tier.INT8
QUANTISED = (Tier.INT8, Tier.INT4, Tier.INT2)
BITS = {Tier.INT8: 8, Tier.INT4: 4, Tier.INT2: 2}

#: (kv_heads, head_dim) of each model the kernels serve (ADR-0003).
SHAPES = {name: (cfg.num_key_value_heads, cfg.head_dim)
          for name in ("qwen2.5-0.5b-instruct", "qwen2.5-1.5b-instruct")
          for cfg in [ModelConfig.from_card(name)]}


def random_page(heads, head_dim, seed=0):
    """Keys with a few outlier channels, as KIVI found real keys to have, and
    values with a few outlier tokens; everything fp16-exact, as a page is."""
    rng = np.random.default_rng(seed)
    keys = rng.normal(0, 1, (P, heads, head_dim))
    keys[:, :, rng.choice(head_dim, 3, replace=False)] *= 40
    keys += rng.normal(0, 4, (1, heads, head_dim))  # a channel's offset: asymmetric
    values = rng.normal(0, 1, (P, heads, head_dim))
    values[rng.choice(P, 2, replace=False)] *= 25
    return fp16_exact(keys), fp16_exact(values)


# -- the page, taken apart --------------------------------------------------------


def split(page, tier, heads, head_dim):
    """A quantised page's regions, read back through the layout the extension
    reports: codes (P, heads, head_dim) and fp16 scales and zero-points."""
    layout = _microinfer.quantised_page_layout(tier, heads, head_dim)
    bits = layout["bits"]
    per_byte = 8 // bits

    def codes(offset):
        raw = page[offset:offset + P * heads * head_dim // per_byte]
        shifts = np.arange(per_byte, dtype=np.uint8) * bits
        unpacked = (raw[:, None] >> shifts) & ((1 << bits) - 1)
        return unpacked.reshape(P, heads, head_dim).astype(np.int64)

    def halves(offset, shape):
        n = int(np.prod(shape))
        return page[offset:offset + 2 * n].view(np.float16).astype(np.float64).reshape(shape)

    return {"key_codes": codes(layout["key_codes"]),
            "value_codes": codes(layout["value_codes"]),
            "key_scales": halves(layout["key_scales"], (heads, head_dim)),
            "key_zeros": halves(layout["key_zeros"], (heads, head_dim)),
            "value_scales": halves(layout["value_scales"], (P, heads)),
            "value_zeros": halves(layout["value_zeros"], (P, heads))}


def reference(x, bits, axis):
    """Asymmetric quantisation in float64, over `axis`: the zero-point is the
    minimum, which an fp16 input holds exactly, and the scale is the range over
    2^bits - 1 levels, rounded up to fp16, the smallest the page can store that
    still reaches the maximum. Codes are formed against the stored scale. Returns the scale, the
    zero-point, the codes, and the unrounded code value, whose distance from a
    rounding tie decides whether an fp32 kernel may round the other way."""
    levels = (1 << bits) - 1
    lo, hi = x.min(axis, keepdims=True), x.max(axis, keepdims=True)
    exact_scale = (hi - lo) / levels
    nearest = exact_scale.astype(np.float16)
    scale = np.where(nearest.astype(np.float64) < exact_scale,
                     np.nextafter(nearest, np.float16(np.inf)), nearest).astype(np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        exact = np.where(scale > 0, (x - lo) / scale, 0.0)
    codes = np.clip(np.rint(exact), 0, levels)
    return scale, lo, codes, exact


def near_tie(exact):
    return np.abs(exact - np.floor(exact) - 0.5) < 1e-3


def round_trip_bound(x, got, scales):
    """What quantisation may lose, per element, derived rather than chosen:
    - half a step of the stored scale, which is rounded up, so that no
      element lies past the last level and none is clamped;
    - half an fp16 ulp for the dequantised value's own rounding, at the larger
      of what went in and what came out;
    - an fp32 ulp of the element, for the kernel's division and its fused
      multiply-add."""
    larger = np.maximum(np.abs(x), np.abs(got))
    out_ulp = np.spacing(larger.astype(np.float16)).astype(np.float64)
    return scales / 2 + out_ulp / 2 + 2.0**-23 * larger


def assert_round_trip_within_bound(keys, values, tier=INT8):
    heads, head_dim = keys.shape[1:]
    page = _microinfer.quantise_page(keys, values, tier)
    got_k, got_v = _microinfer.dequantise_page(page, tier, heads, head_dim)
    parts = split(page, tier, heads, head_dim)
    k_bound = round_trip_bound(keys, got_k, parts["key_scales"][None])
    v_bound = round_trip_bound(values, got_v, parts["value_scales"][..., None])
    k_err, v_err = np.abs(got_k - keys), np.abs(got_v - values)
    assert np.all(k_err <= k_bound), f"key error {k_err.max():.3g} past its bound"
    assert np.all(v_err <= v_bound), f"value error {v_err.max():.3g} past its bound"
    return k_err, v_err, parts


# -- the layout ---------------------------------------------------------------------


def test_metadata_is_1280_bytes_per_page_for_the_1_5b_model():
    heads, head_dim = SHAPES["qwen2.5-1.5b-instruct"]
    for tier in QUANTISED:
        assert _microinfer.quantised_page_layout(tier, heads, head_dim)["metadata_bytes"] == 1280


def test_the_regions_tile_the_page_in_the_documented_order():
    """Key codes, value codes, then key scales, key zero-points, value scales,
    value zero-points, back to back. The metadata is the same at every tier;
    only the codes shrink."""
    for heads, head_dim in SHAPES.values():
        width = heads * head_dim
        for tier in QUANTISED:
            layout = _microinfer.quantised_page_layout(tier, heads, head_dim)
            codes = P * width * BITS[tier] // 8
            assert layout["bits"] == BITS[tier]
            regions = [("key_codes", codes), ("value_codes", codes),
                       ("key_scales", 2 * width), ("key_zeros", 2 * width),
                       ("value_scales", 2 * P * heads), ("value_zeros", 2 * P * heads)]
            offset = 0
            for name, size in regions:
                assert layout[name] == offset, (tier, name)
                offset += size
            assert layout["page_bytes"] == offset
            assert layout["metadata_bytes"] == offset - 2 * codes


def test_effective_bits_count_the_metadata():
    """What a compression ratio must quote (ADR-0005): on the 1.5B model INT8
    is 8.63 bits per element, not 8; INT4 is 4.63 and INT2 2.63."""
    heads, head_dim = SHAPES["qwen2.5-1.5b-instruct"]
    got = {tier: _microinfer.quantised_page_layout(tier, heads, head_dim)["effective_bits"]
           for tier in QUANTISED}
    assert got == {Tier.INT8: 8.625, Tier.INT4: 4.625, Tier.INT2: 2.625}
    layout = _microinfer.quantised_page_layout(INT8, heads, head_dim)
    assert layout["effective_bits"] == 8 * layout["page_bytes"] / (2 * P * heads * head_dim)


def test_fp16_has_no_quantised_layout():
    with pytest.raises(ValueError, match="FP16"):
        _microinfer.quantised_page_layout(Tier.FP16, 2, 64)


# -- the kernel against float64 --------------------------------------------------------


@pytest.mark.parametrize("tier", QUANTISED, ids=lambda t: t.name)
@pytest.mark.parametrize("model", sorted(SHAPES))
def test_codes_and_metadata_are_the_float64_reference(model, tier):
    heads, head_dim = SHAPES[model]
    keys, values = random_page(heads, head_dim)
    parts = split(_microinfer.quantise_page(keys, values, tier), tier, heads, head_dim)

    bits = BITS[tier]
    k_scale, k_zero, k_codes, k_exact = reference(keys.astype(np.float64), bits, axis=0)
    v_scale, v_zero, v_codes, v_exact = reference(values.astype(np.float64), bits, axis=2)
    np.testing.assert_array_equal(parts["key_scales"], k_scale[0])
    np.testing.assert_array_equal(parts["key_zeros"], k_zero[0])
    np.testing.assert_array_equal(parts["value_scales"], v_scale[..., 0])
    np.testing.assert_array_equal(parts["value_zeros"], v_zero[..., 0])
    for got, want, exact in ((parts["key_codes"], k_codes, k_exact),
                             (parts["value_codes"], v_codes, v_exact)):
        differ = got != want
        assert not np.any(differ & ~near_tie(exact)), "a code differs away from a tie"
        assert np.all(np.abs(got - want) <= 1)


@pytest.mark.parametrize("tier", QUANTISED, ids=lambda t: t.name)
@pytest.mark.parametrize("model", sorted(SHAPES))
def test_dequantised_values_are_the_float64_reference(model, tier):
    """q * scale + zero from the page's own codes and metadata, to within the
    per-kernel gate. It is a single fused multiply-add, so there is no
    cancellation to floor: fp32 rounds its result, not its terms."""
    heads, head_dim = SHAPES[model]
    keys, values = random_page(heads, head_dim, seed=1)
    page = _microinfer.quantise_page(keys, values, tier)
    parts = split(page, tier, heads, head_dim)
    got_k, got_v = _microinfer.dequantise_page(page, tier, heads, head_dim)
    assert_within_gate(got_k, parts["key_codes"] * parts["key_scales"][None]
                       + parts["key_zeros"][None])
    assert_within_gate(got_v, parts["value_codes"] * parts["value_scales"][..., None]
                       + parts["value_zeros"][..., None])


@pytest.mark.parametrize("tier", QUANTISED, ids=lambda t: t.name)
def test_codes_are_packed_with_no_padding_first_in_the_lowest_bits(tier):
    """Inputs placed exactly on the levels, so every code is known without
    rounding: each key channel and each value group holds 0 and 2^bits - 1,
    making its scale 1 and its zero-point 0, and its codes the inputs
    themselves. The code regions must then be those codes packed by hand,
    8/bits to a byte with the first in the lowest bits, byte for byte, and
    exactly P * W * bits / 8 bytes each: two codes to a byte at INT4, four at
    INT2, and nothing between them."""
    heads, head_dim = SHAPES["qwen2.5-1.5b-instruct"]
    bits, top = BITS[tier], (1 << BITS[tier]) - 1
    rng = np.random.default_rng(6)
    keys = rng.integers(0, top + 1, (P, heads, head_dim)).astype(np.float32)
    keys[0], keys[1] = 0, top
    values = rng.integers(0, top + 1, (P, heads, head_dim)).astype(np.float32)
    values[:, :, 0], values[:, :, 1] = 0, top
    page = _microinfer.quantise_page(keys, values, tier)
    layout = _microinfer.quantised_page_layout(tier, heads, head_dim)

    def packed(codes):
        per_byte = 8 // bits
        grouped = codes.astype(np.uint8).reshape(-1, per_byte)
        shifts = np.arange(per_byte, dtype=np.uint8) * bits
        return np.bitwise_or.reduce(grouped << shifts, axis=1)

    size = P * heads * head_dim * bits // 8
    assert layout["value_codes"] - layout["key_codes"] == size
    assert layout["key_scales"] - layout["value_codes"] == size
    np.testing.assert_array_equal(page[layout["key_codes"]:layout["value_codes"]], packed(keys))
    np.testing.assert_array_equal(page[layout["value_codes"]:layout["key_scales"]], packed(values))
    got_k, got_v = _microinfer.dequantise_page(page, tier, heads, head_dim)
    np.testing.assert_array_equal(got_k, keys)
    np.testing.assert_array_equal(got_v, values)


# -- what quantisation loses -------------------------------------------------------


@pytest.mark.parametrize("tier", QUANTISED, ids=lambda t: t.name)
@pytest.mark.parametrize("model", sorted(SHAPES))
@pytest.mark.parametrize("seed", range(3))
def test_a_round_trip_loses_at_most_half_a_step(model, seed, tier):
    heads, head_dim = SHAPES[model]
    assert_round_trip_within_bound(*random_page(heads, head_dim, seed), tier)


@pytest.mark.parametrize("tier", QUANTISED, ids=lambda t: t.name)
def test_keys_are_scaled_per_channel_and_values_per_token(tier):
    """The granularity, asserted by independence: an outlier planted in one
    key channel changes no other channel's round trip by a single bit, and an
    outlier in one value group, a (head, token) pair, changes no other group's.
    Only the group that holds the outlier coarsens."""
    heads, head_dim = SHAPES["qwen2.5-0.5b-instruct"]
    keys, values = random_page(heads, head_dim, seed=2)
    base_k, base_v = _microinfer.dequantise_page(
        _microinfer.quantise_page(keys, values, tier), tier, heads, head_dim)

    k_out, v_out = keys.copy(), values.copy()
    k_out[5, 1, 7] = 3000.0  # one channel: (head 1, channel 7)
    v_out[9, 0, 3] = -3000.0  # one group: (token 9, head 0)
    got_k, got_v = _microinfer.dequantise_page(
        _microinfer.quantise_page(k_out, v_out, tier), tier, heads, head_dim)

    other_channels = np.ones((heads, head_dim), bool)
    other_channels[1, 7] = False
    np.testing.assert_array_equal(got_k[:, other_channels], base_k[:, other_channels])
    other_groups = np.ones((P, heads), bool)
    other_groups[9, 0] = False
    np.testing.assert_array_equal(got_v[other_groups], base_v[other_groups])
    assert not np.array_equal(got_k[:, 1, 7], base_k[:, 1, 7])
    assert not np.array_equal(got_v[9, 0], base_v[9, 0])


@pytest.mark.parametrize("tier", QUANTISED, ids=lambda t: t.name)
def test_the_quantiser_is_asymmetric(tier):
    """A channel far from zero keeps a step of its own range, not of its
    magnitude: at INT8 [100, 101] quantises in steps of 1/255, where a
    symmetric quantiser would step by 101/127 and lose almost everything."""
    levels = (1 << BITS[tier]) - 1
    heads, head_dim = SHAPES["qwen2.5-0.5b-instruct"]
    keys, values = random_page(heads, head_dim, seed=3)
    keys[:, 0, 0] = fp16_exact(np.linspace(100, 101, P))
    values[4, 1] = fp16_exact(np.linspace(-51, -50, head_dim))
    k_err, v_err, parts = assert_round_trip_within_bound(keys, values, tier)
    assert parts["key_zeros"][0, 0] == 100 and parts["value_zeros"][4, 1] == -51
    assert k_err[:, 0, 0].max() <= 1 / levels and v_err[4, 1].max() <= 1 / levels


@pytest.mark.parametrize("tier", QUANTISED, ids=lambda t: t.name)
def test_a_group_too_narrow_for_a_normal_fp16_scale_keeps_its_maximum(tier):
    """A key channel from layer 0 of the 1.5B model on adversarial-00: a
    range of 1.26e-4, so its scale, 4.9e-7, is below fp16's smallest normal,
    where the spacing is 6e-8. Rounded to nearest it fell an eighth short, the
    last level missed the maximum by 8 steps, and the clamp lost them. The
    scale is rounded up instead, and the maximum comes back within half a
    step."""
    heads, head_dim = SHAPES["qwen2.5-1.5b-instruct"]
    keys, values = random_page(heads, head_dim, seed=5)
    keys[:, 1, 58] = fp16_exact(np.linspace(0.00554656982421875, 0.005672454833984375, P))
    k_err, _, parts = assert_round_trip_within_bound(keys, values, tier)
    scale, levels = parts["key_scales"][1, 58], (1 << BITS[tier]) - 1
    assert scale < 2.0**-14 and levels * scale >= keys[:, 1, 58].max() - keys[:, 1, 58].min()
    assert k_err[:, 1, 58].max() <= scale / 2 + np.spacing(np.float16(0.0057)) / 2


@pytest.mark.parametrize("tier", QUANTISED, ids=lambda t: t.name)
def test_a_constant_group_round_trips_exactly(tier):
    heads, head_dim = SHAPES["qwen2.5-0.5b-instruct"]
    keys, values = random_page(heads, head_dim, seed=4)
    keys[:, 0, 5] = -2.5
    values[3, 1] = 0.75
    k_err, v_err, parts = assert_round_trip_within_bound(keys, values, tier)
    assert parts["key_scales"][0, 5] == 0 and k_err[:, 0, 5].max() == 0
    assert parts["value_scales"][3, 1] == 0 and v_err[3, 1].max() == 0


# -- what may not be quantised -------------------------------------------------------


def test_the_page_being_filled_cannot_be_quantised():
    """A key channel's scale spans all P positions, so a page with fewer has
    none to compute: the open page stays at FP16 (ADR-0005)."""
    heads, head_dim = SHAPES["qwen2.5-0.5b-instruct"]
    keys, values = random_page(heads, head_dim)
    for filled in (1, P - 1):
        with pytest.raises(_microinfer.OpenPage, match=f"{filled} of {P} positions.*FP16"):
            _microinfer.quantise_page(keys[:filled], values[:filled], INT8)
    assert issubclass(_microinfer.OpenPage, ValueError)


def test_malformed_pages_are_refused():
    heads, head_dim = SHAPES["qwen2.5-0.5b-instruct"]
    keys, values = random_page(heads, head_dim)
    with pytest.raises(ValueError, match="positions"):
        _microinfer.quantise_page(np.concatenate([keys, keys]),
                                  np.concatenate([values, values]), INT8)
    with pytest.raises(ValueError, match="shape"):
        _microinfer.quantise_page(keys, values[:, :1], INT8)
    with pytest.raises(ValueError, match="FP16"):
        _microinfer.quantise_page(keys, values, Tier.FP16)
    page = _microinfer.quantise_page(keys, values, INT8)
    with pytest.raises(ValueError, match="bytes"):
        _microinfer.dequantise_page(page[:-1], INT8, heads, head_dim)


# -- real keys and values ------------------------------------------------------------


@pytest.fixture(scope="module")
def engine() -> Engine:
    e = Engine(require_model("qwen2.5-0.5b-instruct"))
    e.load_weights()
    return e


def test_real_keys_and_values_from_golden_data_round_trip_within_the_bound(engine):
    """Every full page of every layer, for two golden prompts, at every tier:
    the keys as the cache holds them, without their bias (ADR-0009), and the
    values. Real keys carry the channel outliers KIVI describes, which random
    data only imitates. Across the whole set, the relative error grows as the
    tiers narrow, for keys and for values alike: INT8 < INT4 < INT2."""
    try:
        golden = GoldenSet(REPO / "tests" / "golden" / "qwen2.5-0.5b-instruct")
    except GoldenError as exc:
        pytest.skip(str(exc))
    pages = [(k[start:start + P], v[start:start + P])
             for prompt_id in ("long-01", "adversarial-01")
             for keys, values in [engine.cached_kv(golden[prompt_id].token_ids)]
             for k, v in zip(keys, values)
             for start in range(0, keys.shape[1] // P * P, P)]
    assert len(pages) > 1000

    relative = {}
    for tier in QUANTISED:
        sq_err, sq = np.zeros(2), np.zeros(2)
        for k, v in pages:
            k_err, v_err, _ = assert_round_trip_within_bound(k, v, tier)
            sq_err += [np.square(k_err).sum(), np.square(v_err).sum()]
            sq += [np.square(k.astype(np.float64)).sum(), np.square(v.astype(np.float64)).sum()]
        relative[tier.name] = np.sqrt(sq_err / sq)
    print(f"\n{len(pages)} real pages; relative RMS error (keys, values): {relative}")
    for half in (0, 1):
        assert relative["INT8"][half] < relative["INT4"][half] < relative["INT2"][half]
