"""Shared fixtures.

The real-checkpoint helper lives here because two test modules need it and the
skip logic is subtler than it looks: a file that exists is not the same as a
file that is complete.
"""

import gc
from pathlib import Path

import pytest

from microinfer import weights
from microinfer.models import UPSTREAM

MODELS = Path(__file__).resolve().parent.parent / "models"


def require_model(name: str) -> Path:
    """The model directory, or skip with the command that would fetch it.

    Skips on a truncated file as well as an absent one. A half-downloaded
    checkpoint is not a failure of the code under test, and reporting it as one
    buries the real result under noise. The message distinguishes the two cases
    so a genuinely corrupt file is still visible.
    """
    directory = MODELS / name
    checkpoint = directory / "model.safetensors"

    if not checkpoint.is_file():
        pytest.skip(
            f"{name} is not downloaded. Fetch it with:\n"
            f"  curl -L --fail --retry 5 --retry-all-errors -C - --create-dirs \\\n"
            f"    -o models/{name}/model.safetensors \\\n"
            f"    https://huggingface.co/{UPSTREAM[name]}/resolve/main/model.safetensors\n"
            f"  Then re-run: a truncated file is reported as truncated, not as absent.\n"
            f"  --fail matters: without it curl exits 0 on a connection the server\n"
            f"  closed early, leaving a short file that looks like a successful download."
        )

    try:
        weights.check_complete(checkpoint)
    except weights.WeightError as exc:
        pytest.skip(f"{name}: {exc}")

    return directory


def stable_free_bytes() -> int:
    """Device free memory, with this process's own pending frees settled first.

    Reading `cudaMemGetInfo` without this is unreliable in a suite, and the
    failure is subtle: a tensor dropped by an *earlier* test may be collected
    while a later one is measuring, which frees device memory between the two
    readings and shrinks the delta. Measured here, that turned a 32.00 MiB
    allocation into an apparent 31.19 MiB and failed an assertion that was
    correct.

    This matters well beyond these tests. ADR-0007's central contract — that
    reclaimed pages genuinely return to the driver — is measured exactly this
    way, and an allocator test built on an unsettled reading would either hide a
    real leak or cry wolf about one. #10 should use this rather than calling
    `device_memory_info` directly.

    What it cannot settle is memory held by *other processes*, which on this
    machine is the phenomenon under study rather than noise.
    """
    from microinfer import _microinfer

    gc.collect()
    return _microinfer.device_memory_info()["free"]
