"""Reading safetensors, and the bf16 to fp16 crossing.

The synthetic tests here run anywhere. The tests against the real checkpoint
skip when it is absent, with instructions — a 4 GiB download is not a
precondition for running the suite.
"""

import json
import struct
from pathlib import Path

import numpy as np
import pytest

from conftest import require_model
from microinfer import weights

MODELS = Path(__file__).resolve().parent.parent / "models"


def write_safetensors(path: Path, tensors: dict[str, tuple[str, np.ndarray]]) -> None:
    """Build a file by hand, so the reader is tested against the format rather
    than against the library that produced it."""
    header, blob, offset = {}, bytearray(), 0
    for name, (dtype, arr) in tensors.items():
        raw = arr.tobytes()
        header[name] = {"dtype": dtype, "shape": list(arr.shape),
                        "data_offsets": [offset, offset + len(raw)]}
        blob += raw
        offset += len(raw)
    encoded = json.dumps(header).encode()
    path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + bytes(blob))


def test_bf16_widening_is_exact():
    """bf16 is float32 with the low mantissa bits cut off, so widening is a
    shift and must be lossless. If this is ever approximate, something is wrong
    with the decoder, not with floating point."""
    original = np.array([1.0, -2.5, 0.0, 1e30, -1e-30, 3.5], dtype=np.float32)
    truncated = (original.view(np.uint32) >> 16).astype(np.uint16)  # to bf16
    widened = (truncated.astype(np.uint32) << 16).view(np.float32)
    round_tripped = (widened.view(np.uint32) >> 16).astype(np.uint16)
    np.testing.assert_array_equal(truncated, round_tripped)


def test_reads_bf16_f16_and_f32(tmp_path):
    f32 = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    bf16 = (f32.view(np.uint32) >> 16).astype(np.uint16)
    f16 = f32.astype(np.float16)
    path = tmp_path / "m.safetensors"
    write_safetensors(path, {"a": ("F32", f32), "b": ("BF16", bf16), "c": ("F16", f16)})

    got = dict(weights.iter_tensors(path))
    assert set(got) == {"a", "b", "c"}
    np.testing.assert_array_equal(got["a"], f32)
    np.testing.assert_array_equal(got["b"], f32)  # exact — these values are bf16-representable
    np.testing.assert_array_equal(got["c"], f32)
    assert got["a"].shape == (2, 2)


def test_unknown_dtype_is_an_error_not_a_guess(tmp_path):
    path = tmp_path / "m.safetensors"
    write_safetensors(path, {"a": ("I64", np.array([1, 2], dtype=np.int64))})
    with pytest.raises(weights.WeightError, match="I64"):
        list(weights.iter_tensors(path))


def test_a_truncated_download_says_so(tmp_path):
    """A part-downloaded checkpoint would otherwise surface as a reshape error
    deep in the reader, reading like a parser bug."""
    f32 = np.arange(64, dtype=np.float32).reshape(8, 8)
    path = tmp_path / "m.safetensors"
    write_safetensors(path, {"a": ("F32", f32)})
    whole = path.read_bytes()
    path.write_bytes(whole[: len(whole) - 32])
    with pytest.raises(weights.WeightError, match="truncated"):
        list(weights.iter_tensors(path))


def test_a_file_that_is_not_safetensors_fails_clearly(tmp_path):
    path = tmp_path / "junk.safetensors"
    path.write_bytes(b"\xff" * 64)
    with pytest.raises(weights.WeightError, match="not a safetensors file"):
        weights.read_header(path)


def test_fp16_range_check_catches_what_bf16_can_hold_and_fp16_cannot():
    assert weights.check_fp16_range("ok", np.array([1.0, -65000.0], dtype=np.float32)) is None
    over = weights.check_fp16_range("bad", np.array([1.0, 1e30], dtype=np.float32))
    assert over is not None and over > weights.FP16_MAX


# --- against the real checkpoint -------------------------------------------

@pytest.mark.parametrize("name", ["qwen2.5-0.5b-instruct", "qwen2.5-1.5b-instruct"])
def test_real_checkpoint_is_bf16_throughout(name):
    infos = weights.describe(require_model(name) / "model.safetensors")
    assert infos, "no tensors in the checkpoint"
    assert {i.dtype for i in infos} == {"BF16"}


@pytest.mark.parametrize("name", ["qwen2.5-0.5b-instruct", "qwen2.5-1.5b-instruct"])
def test_no_real_weight_overflows_fp16(name):
    """The engine stores fp16. bf16 reaches far past fp16's 65504, so this is
    the check that the whole conversion is safe for these particular weights —
    not a general guarantee about bf16 checkpoints."""
    offenders = {
        n: over
        for n, v in weights.iter_tensors(require_model(name) / "model.safetensors")
        if (over := weights.check_fp16_range(n, v)) is not None
    }
    assert not offenders, f"tensors exceeding fp16 range: {offenders}"
