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
        yield name, _decode(name, meta, mapped, base)


def read_tensor(path: Path, name: str, rows: np.ndarray | None = None) -> np.ndarray:
    """One tensor as float32, or only the given rows of its first axis.

    `rows` exists for the embedding table. A prompt needs a few hundred of its
    151,936 rows, and decoding the whole table would cost a gigabyte of host
    memory at float32 for the 1.5B model. The rows are picked while still in
    the checkpoint's own dtype, so only they are ever decoded.
    """
    check_complete(path)
    header, base = read_header(path)
    if name not in header:
        raise WeightError(f"{path.name} has no tensor named {name}")
    mapped = np.memmap(path, dtype=np.uint8, mode="r")
    return _decode(name, header[name], mapped, base, rows)


def _decode(name: str, meta: dict, mapped: np.ndarray, base: int,
            rows: np.ndarray | None = None) -> np.ndarray:
    dtype = meta["dtype"]
    if dtype not in _DECODERS:
        raise WeightError(
            f"{name}: dtype {dtype} is not supported. This project reads "
            f"{', '.join(sorted(_DECODERS))}."
        )
    start, end = meta["data_offsets"]
    raw = mapped[base + start : base + end]
    shape = list(meta["shape"])
    if rows is not None:
        raw = np.ascontiguousarray(raw.reshape(shape[0], -1)[rows]).reshape(-1)
        shape[0] = len(rows)
    return _DECODERS[dtype](raw).reshape(shape)


def describe(path: Path) -> list[TensorInfo]:
    header, _ = read_header(path)
    return [
        TensorInfo(name, meta["dtype"], tuple(meta["shape"]), int(np.prod(meta["shape"])))
        for name, meta in header.items()
        if name != "__metadata__"
    ]


def check_shapes(path: Path, expected: dict[str, tuple[int, ...]], source: str) -> None:
    """Raise WeightError, naming every difference, unless the checkpoint holds
    exactly the tensors `expected` names, each of its shape. `source` says
    where `expected` came from. Reads only the header."""
    found = {info.name: info.shape for info in describe(path)}
    wrong = [f"    {n}: missing" for n in expected if n not in found]
    wrong += [f"    {n}: not implied by {source}" for n in found if n not in expected]
    wrong += [f"    {n}: shape {found[n]}, {source} implies {expected[n]}"
              for n in expected if n in found and found[n] != expected[n]]
    if wrong:
        raise WeightError(f"{path} does not hold what {source} describes; nothing was "
                          f"allocated:\n" + "\n".join(wrong))


def check_fp16_range(name: str, values: np.ndarray) -> str | None:
    """Return why this tensor will not survive conversion to fp16, or None.

    Two ways to fail, and they need separate handling because one of them is
    invisible to the obvious test.

    **Out of range.** bf16 carries float32's exponent and reaches past 3e38;
    fp16 stops at 65504. A weight beyond that becomes an infinity.

    **Not a number.** Every comparison with NaN is false, so a bare
    `peak > FP16_MAX` lets a NaN straight through — the exact silent corruption
    this function exists to stop. It is checked first, on its own.
    """
    if values.size == 0:
        return None
    if np.isnan(values).any():
        count = int(np.isnan(values).sum())
        return f"{count} NaN value(s)"
    peak = float(np.abs(values).max())  # infinity compares greater, so it lands here
    if peak > FP16_MAX:
        return f"magnitude {peak:.3e} exceeds fp16's maximum of {FP16_MAX:.0f}"
    return None
