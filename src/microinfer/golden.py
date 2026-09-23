"""Reading the reference the engine is judged against.

The generator lives in `tools/gen_golden.py` and imports torch. Nothing here
does, and nothing here may: this module runs inside the engine's process
(ADR-0002).

What a golden file holds, and why it holds only that, is explained in the
generator. The short version: an argmax at every position, full logits at a
sample of positions, and hidden states only where a diagnostic would need them.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path

import numpy as np


class GoldenError(RuntimeError):
    pass


def config_fingerprint(config_bytes: bytes) -> str:
    """Identify the configuration a reference was generated from.

    Defined on the engine's side of the ADR-0002 wall and imported by the
    generator. That wall forbids sharing torch, not stdlib; two independent
    copies of this would drift, and a drifted fingerprint silently stops
    catching stale references — the failure it exists to catch.
    """
    return hashlib.sha256(config_bytes).hexdigest()[:16]


def load_prompts(path: str | Path) -> list[dict]:
    """Read the committed prompt set. Shared with the generator for the same
    reason as the fingerprint."""
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


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

    @property
    def generated(self) -> np.ndarray | None:
        """HuggingFace's greedy continuation, new tokens only, for the smoke
        test's prompts; None for the rest. Ends at an end-of-sequence token if
        one came before the length limit."""
        return self._data.get("generated")

    @property
    def dense_logits(self) -> np.ndarray | None:
        """Logits at every position, for the few prompts kept densely to
        validate the gate's KL sample (ADR-0006). In a file of their own, so
        that reading any other field never loads them."""
        path = self.path.with_suffix(".dense.npz")
        if not path.is_file():
            return None
        with np.load(path) as handle:
            return handle["logits"]

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

        # Enforced, not merely documented. A reference in the checkpoint's own
        # bfloat16 would be eight times coarser than the fp16 the engine stores
        # and would fail ADR-0006 on its own account (ADR-0003's amendment).
        if self.manifest.get("dtype") != "float32":
            raise GoldenError(
                f"{self.directory} was generated at {self.manifest.get('dtype')!r}, "
                f"not float32. A reference must be more accurate than the thing "
                f"it measures; regenerate it."
            )

    @property
    def model(self) -> str:
        return self.manifest["model"]

    @property
    def dtype(self) -> str:
        """Always float32 — the constructor refuses anything else."""
        return self.manifest["dtype"]

    def matches_config(self, config_bytes: bytes) -> bool:
        """Whether these references were generated from this exact config.json.

        A stale reference is worse than a missing one: it fails in a way that
        looks like a kernel bug.
        """
        return config_fingerprint(config_bytes) == self.manifest["config_sha256_16"]

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
