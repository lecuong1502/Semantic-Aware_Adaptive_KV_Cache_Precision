"""Reading the reference the engine is judged against.

The generator lives in `tools/gen_golden.py` and imports torch. Nothing here
does, and nothing here may: this module runs inside the engine's process
(ADR-0002).

What a golden file holds, and why it holds only that, is explained in the
generator. The short version: an argmax at every position, full logits at a
sample of positions, and hidden states only where a diagnostic would need them.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path

import numpy as np


class GoldenError(RuntimeError):
    pass


@dataclass(frozen=True)
class Golden:
    """One prompt's reference output."""

    prompt_id: str
    path: Path

    @cached_property
    def _data(self) -> dict[str, np.ndarray]:
        with np.load(self.path) as handle:
            return {key: handle[key] for key in handle.files}

    @property
    def token_ids(self) -> np.ndarray:
        """The tokenised prompt. Stored so the engine needs no tokeniser of its
        own to be compared — one fewer thing that has to match before the
        numbers can be trusted."""
        return self._data["token_ids"]

    @property
    def argmax(self) -> np.ndarray:
        """The reference's chosen token at every position. This is what top-1
        agreement is measured against (ADR-0006)."""
        return self._data["argmax"]

    @property
    def logit_positions(self) -> np.ndarray:
        return self._data["logit_positions"]

    @property
    def logits(self) -> np.ndarray:
        """Full distributions, at `logit_positions` only. KL needs the whole
        distribution, and the whole distribution at every position would be
        gigabytes."""
        return self._data["logits"]

    @property
    def hidden_states(self) -> np.ndarray | None:
        """Shape (layers + 1, seq, hidden); the first entry is the embedding
        output. Absent for most prompts — it is a diagnostic, wanted only once
        the gate is already red."""
        return self._data.get("hidden_states")

    def __len__(self) -> int:
        return len(self.token_ids)


class GoldenSet:
    """Every reference generated for one model."""

    def __init__(self, directory: str | Path):
        self.directory = Path(directory)
        manifest = self.directory / "manifest.json"
        if not manifest.is_file():
            raise GoldenError(
                f"No golden tensors at {self.directory}. Generate them with:\n"
                f"  .venv-golden/bin/python tools/gen_golden.py --model {self.directory.name}\n"
                f"They are deliberately not committed; see the generator."
            )
        self.manifest = json.loads(manifest.read_text())

    @property
    def model(self) -> str:
        return self.manifest["model"]

    @property
    def dtype(self) -> str:
        """Always float32. The checkpoint is bfloat16, whose ulp is eight times
        coarser than the fp16 the engine stores — a reference in the
        checkpoint's own dtype would carry more error than the thing it
        judges."""
        return self.manifest["dtype"]

    def matches_config(self, config_bytes: bytes) -> bool:
        """Whether these references were generated from this exact config.json.

        A stale reference is worse than a missing one: it fails in a way that
        looks like a kernel bug.
        """
        import hashlib

        return hashlib.sha256(config_bytes).hexdigest()[:16] == self.manifest["config_sha256_16"]

    def ids(self) -> list[str]:
        return [p["id"] for p in self.manifest["prompts"]]

    def __len__(self) -> int:
        return len(self.manifest["prompts"])

    def __getitem__(self, prompt_id: str) -> Golden:
        path = self.directory / f"{prompt_id}.npz"
        if not path.is_file():
            raise GoldenError(f"{prompt_id} is in the manifest but {path} is missing")
        return Golden(prompt_id, path)

    def __iter__(self):
        return (self[i] for i in self.ids())
