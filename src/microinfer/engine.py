"""The engine's Seam A: load a model, know what it costs.

Milestone 0 stops well short of generating anything. What exists here is the
part every later ticket stands on — a verified configuration, weights on the
device in the format the kernels read, and an honest account of the memory.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from . import _microinfer, weights
from .config import ConfigMismatch, ModelConfig
from .footprint import Footprint
from .models import VERIFIED


def expected_weight_bytes(cfg: ModelConfig) -> int:
    """Parameter bytes implied by config.json alone, at fp16.

    Derived rather than measured, so that a mismatch against the real
    checkpoint is a finding rather than a tautology. Qwen2 puts a bias on the
    query, key and value projections and none on the output projection.
    """
    h, i = cfg.hidden_size, cfg.intermediate_size
    q_dim = cfg.num_attention_heads * cfg.head_dim
    kv_dim = cfg.num_key_value_heads * cfg.head_dim

    embedding = cfg.vocab_size * h
    per_layer = (
        h * q_dim + q_dim          # q_proj + bias
        + h * kv_dim + kv_dim      # k_proj + bias
        + h * kv_dim + kv_dim      # v_proj + bias
        + q_dim * h                # o_proj, no bias
        + 3 * h * i                # gate, up, down
        + 2 * h                    # the two RMSNorm weights
    )
    tail = h  # final norm
    head = 0 if cfg.tie_word_embeddings else cfg.vocab_size * h

    return 2 * (embedding + cfg.num_hidden_layers * per_layer + tail + head)


def kv_cache_bytes(cfg: ModelConfig, context_length: int, bytes_per_element: int = 2) -> int:
    """The cache at a given context length and precision.

    `bytes_per_element` is the whole point of the project: 2 at FP16, 1 at INT8,
    and fractional below that. Metadata is not counted here — ADR-0005 keeps
    that with the quantiser, where its size is known.
    """
    per_token = 2 * cfg.num_hidden_layers * cfg.num_key_value_heads * cfg.head_dim
    return per_token * context_length * bytes_per_element


class Engine:
    """Seam A. Everything a test or a caller touches goes through here."""

    def __init__(self, model_dir: str | Path, *, verify: bool = True):
        self.model_dir = Path(model_dir)
        self.config = ModelConfig.from_model_dir(self.model_dir)

        if verify:
            expected = VERIFIED.get(self.config.name)
            if expected is None:
                raise ConfigMismatch(
                    f"{self.config.name!r} has no verified constants recorded. "
                    f"Known: {', '.join(sorted(VERIFIED))}. Add a card and an "
                    f"entry in models.py before running against it."
                )
            self.config.verify(expected)

        self._tensors: dict[str, _microinfer.DeviceTensor] = {}
        self._workspace_bytes = 0

    # -- loading ------------------------------------------------------------

    def load_weights(self) -> None:
        """Read the checkpoint and put it on the device as fp16.

        Every tensor is range-checked on the way. bf16 carries float32's
        exponent and reaches far past fp16's 65504; a weight over that line
        would become an infinity and poison everything downstream in silence.
        """
        path = self.model_dir / "model.safetensors"
        overflows: dict[str, float] = {}

        for name, values in weights.iter_tensors(path):
            peak = weights.check_fp16_range(name, values)
            if peak is not None:
                overflows[name] = peak
                continue
            flat = np.ascontiguousarray(values, dtype=np.float32).reshape(-1)
            self._tensors[name] = _microinfer.upload_fp16(flat)

        if overflows:
            raise weights.WeightError(
                f"{len(overflows)} tensor(s) exceed fp16's range of "
                f"{weights.FP16_MAX} and would become infinities: {overflows}. "
                f"The checkpoint is bf16, which reaches further than fp16."
            )

    @property
    def tensors(self) -> dict[str, _microinfer.DeviceTensor]:
        return dict(self._tensors)

    # -- accounting ---------------------------------------------------------

    def footprint(self, context_length: int = 0, bytes_per_element: int = 2) -> Footprint:
        info = _microinfer.device_memory_info()
        return Footprint(
            weights=sum(t.nbytes for t in self._tensors.values()),
            kv_cache=kv_cache_bytes(self.config, context_length, bytes_per_element),
            workspace=self._workspace_bytes,
            device_free=info["free"],
            device_total=info["total"],
        )

    def expected_weight_bytes(self) -> int:
        return expected_weight_bytes(self.config)
