"""Shared fixtures.

The real-checkpoint helper lives here because two test modules need it and the
skip logic is subtler than it looks: a file that exists is not the same as a
file that is complete.
"""

from pathlib import Path

import pytest

from microinfer import weights

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
        upstream = name.replace("qwen2.5", "Qwen2.5").replace("b-instruct", "B-Instruct")
        pytest.skip(
            f"{name} is not downloaded. Fetch it with:\n"
            f"  curl -L --create-dirs -o models/{name}/model.safetensors \\\n"
            f"    https://huggingface.co/Qwen/{upstream}/resolve/main/model.safetensors"
        )

    try:
        weights.check_complete(checkpoint)
    except weights.WeightError as exc:
        pytest.skip(f"{name}: {exc}")

    return directory
