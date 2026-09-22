"""Shared fixtures.

The real-checkpoint helper lives here because two test modules need it and the
skip logic is subtler than it looks: a file that exists is not the same as a
file that is complete.
"""

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
