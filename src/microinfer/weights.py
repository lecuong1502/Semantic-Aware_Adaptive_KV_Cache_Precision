"""Reading safetensors without PyTorch, and the bf16 problem.

Qwen2.5 ships `torch_dtype: bfloat16`. NumPy has no bfloat16, and
`safetensors.numpy` cannot load one for the same reason — which is why this
module parses the container itself rather than taking the dependency.

The conversion is exact and cheap. bfloat16 *is* the top sixteen bits of a
float32: same 8-bit exponent, mantissa truncated from 23 bits to 7. Widening it
back is a shift, with no rounding anywhere.

Going on to fp16 is not free, and in the opposite direction from what people
expect. fp16 has *more* mantissa (10 bits against 7), so precision is gained;
what is lost is range, because bf16 carries float32's exponent and reaches far
beyond fp16's largest finite value of 65504. Every tensor is checked for that on
the way in rather than silently becoming an infinity.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np

FP16_MAX = 65504.0

#: Only the dtypes this project can encounter. An unknown one is an error, not
#: something to guess at.
_DECODERS = {
    "F32": lambda raw: raw.view(np.float32),
    "F16": lambda raw: raw.view(np.float16).astype(np.float32),
    # bf16 is float32 with the low 16 mantissa bits cut off; put them back.
    "BF16": lambda raw: (raw.view(np.uint16).astype(np.uint32) << 16).view(np.float32),
}


class WeightError(RuntimeError):
    pass


@dataclass(frozen=True)
class TensorInfo:
    name: str
    dtype: str
    shape: tuple[int, ...]
    numel: int


def read_header(path: Path) -> tuple[dict, int]:
    """Return the safetensors JSON header and the offset its data starts at."""
    with open(path, "rb") as f:
        size = int.from_bytes(f.read(8), "little")
        if size <= 0 or size > 100_000_000:
            raise WeightError(f"{path}: implausible header length {size}; not a safetensors file")
        header = json.loads(f.read(size))
    return header, 8 + size


def check_complete(path: Path) -> None:
    """Fail clearly on a truncated file.

    A part-downloaded checkpoint otherwise surfaces as a reshape error deep in
    the reader, which reads like a bug in the parser rather than an incomplete
    download. Cheap to check, and it happens.
    """
    header, base = read_header(path)
    needed = base + max(
        (meta["data_offsets"][1] for name, meta in header.items() if name != "__metadata__"),
        default=0,
    )
    actual = path.stat().st_size
    if actual < needed:
        raise WeightError(
            f"{path.name} is truncated: {actual:,} bytes on disk, {needed:,} expected "
            f"from the header. The download is incomplete."
        )


def iter_tensors(path: Path) -> Iterator[tuple[str, np.ndarray]]:
    """Yield every tensor as float32, in file order.

    Memory-mapped: a 3 GiB checkpoint is never held in host RAM in full, which
    matters more than usual on a machine chosen for having too little memory.
    """
    check_complete(path)
    header, base = read_header(path)
    mapped = np.memmap(path, dtype=np.uint8, mode="r")

    for name, meta in header.items():
        if name == "__metadata__":
            continue
        dtype = meta["dtype"]
        if dtype not in _DECODERS:
            raise WeightError(
                f"{name}: dtype {dtype} is not supported. This project reads "
                f"{', '.join(sorted(_DECODERS))}."
            )
        start, end = meta["data_offsets"]
        raw = mapped[base + start : base + end]
        values = _DECODERS[dtype](raw).reshape(meta["shape"])
        yield name, values


def describe(path: Path) -> list[TensorInfo]:
    header, _ = read_header(path)
    return [
        TensorInfo(name, meta["dtype"], tuple(meta["shape"]), int(np.prod(meta["shape"])))
        for name, meta in header.items()
        if name != "__metadata__"
    ]


def check_fp16_range(name: str, values: np.ndarray) -> float | None:
    """Return the offending magnitude if this tensor will not survive fp16.

    bf16 reaches ~3.4e38; fp16 stops at 65504. A weight past that becomes an
    infinity on conversion and poisons everything downstream, silently.
    """
    peak = float(np.abs(values).max()) if values.size else 0.0
    return peak if peak > FP16_MAX else None
