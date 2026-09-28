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

from conftest import each, require_model
from microinfer import weights



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


def test_reads_bf16_f16_and_f32_whole_or_by_rows(tmp_path):
    """bf16 is float32 with the low mantissa bits cut off, so widening is a
    shift and must be lossless: if it is ever approximate, the decoder is
    wrong, not floating point. All three dtypes read back exactly. Rows are
    picked before decoding, correct only if the row stride is the stored
    dtype's and not float32's, which BF16 and F16 are where it shows; out of
    order and repeated, as token ids are."""
    original = np.array([1.0, -2.5, 0.0, 1e30, -1e-30, 3.5], dtype=np.float32)
    truncated = (original.view(np.uint32) >> 16).astype(np.uint16)  # to bf16
    widened = (truncated.astype(np.uint32) << 16).view(np.float32)
    np.testing.assert_array_equal(truncated, (widened.view(np.uint32) >> 16).astype(np.uint16))

    f32 = np.arange(24, dtype=np.float32).reshape(6, 2, 2)
    for dtype, stored in (("F32", f32), ("BF16", (f32.view(np.uint32) >> 16).astype(np.uint16)),
                          ("F16", f32.astype(np.float16))):
        path = tmp_path / f"{dtype}.safetensors"
        write_safetensors(path, {"other": ("F32", np.zeros(3, np.float32)), "t": (dtype, stored)})
        got = dict(weights.iter_tensors(path))
        np.testing.assert_array_equal(got["t"], f32, err_msg=dtype)
        np.testing.assert_array_equal(weights.read_tensor(path, "t"), f32, err_msg=dtype)
        rows = np.array([4, 0, 4, 5])
        np.testing.assert_array_equal(weights.read_tensor(path, "t", rows=rows), f32[rows],
                                      err_msg=dtype)
        with pytest.raises(weights.WeightError, match="missing"):
            weights.read_tensor(path, "missing")


def test_what_is_not_a_whole_checkpoint_of_known_dtypes_is_an_error_not_a_guess(tmp_path):
    """An unknown dtype; a part-downloaded checkpoint, which would otherwise
    surface as a reshape error deep in the reader, reading like a parser bug;
    a file that is not safetensors at all."""
    path = tmp_path / "m.safetensors"
    write_safetensors(path, {"a": ("I64", np.array([1, 2], dtype=np.int64))})
    with pytest.raises(weights.WeightError, match="I64"):
        list(weights.iter_tensors(path))

    write_safetensors(path, {"a": ("F32", np.arange(64, dtype=np.float32).reshape(8, 8))})
    whole = path.read_bytes()
    path.write_bytes(whole[: len(whole) - 32])
    with pytest.raises(weights.WeightError, match="truncated"):
        list(weights.iter_tensors(path))

    path.write_bytes(b"\xff" * 64)
    with pytest.raises(weights.WeightError, match="not a safetensors file"):
        weights.read_header(path)


def test_the_fp16_range_check_catches_what_bf16_can_hold_and_fp16_cannot():
    """Past 65504, and infinities; and NaN, which every comparison lets
    through, `peak > FP16_MAX` alone included: the exact silent corruption
    the check exists to stop, invisible to the obvious test."""
    assert weights.check_fp16_range("ok", np.array([1.0, -65000.0], dtype=np.float32)) is None
    assert weights.check_fp16_range("empty", np.array([], dtype=np.float32)) is None
    for bad in (1e30, np.inf, -np.inf):
        reason = weights.check_fp16_range("bad", np.array([1.0, bad], dtype=np.float32))
        assert reason is not None and "exceeds" in reason
    reason = weights.check_fp16_range("bad", np.array([1.0, np.nan, np.nan], dtype=np.float32))
    assert reason is not None and "NaN" in reason and "2" in reason


# --- against the real checkpoint -------------------------------------------

def test_real_checkpoints_are_bf16_throughout_and_fit_fp16():
    """The engine stores fp16, and bf16 reaches far past fp16's 65504: this is
    the check that the conversion is safe for these particular weights, not a
    general guarantee about bf16 checkpoints."""
    def fits(name):
        path = require_model(name) / "model.safetensors"
        infos = weights.describe(path)
        assert infos and {i.dtype for i in infos} == {"BF16"}
        offenders = {n: over for n, v in weights.iter_tensors(path)
                     if (over := weights.check_fp16_range(n, v)) is not None}
        assert not offenders, f"tensors exceeding fp16 range: {offenders}"

    each(["qwen2.5-0.5b-instruct", "qwen2.5-1.5b-instruct"], fits)
