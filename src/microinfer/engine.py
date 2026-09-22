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


def kv_cache_bytes(cfg: ModelConfig, context_length: int,
                   bytes_per_element: float = 2.0) -> int:
    """Raw cache bytes at a given context length and element size.

    A *projection*, not a measurement: nothing here has been allocated. It is
    what the cache would occupy, which is the quantity ADR-0003's model choice
    was argued from.

    `bytes_per_element` is fractional below one byte — 0.5 at INT4, 0.25 at
    INT2 — hence the float. **Scale metadata is excluded.** ADR-0005 keeps that
    with the quantiser, where its size is known, and reminds anyone quoting a
    compression ratio to count it: INT4 is 4.63 effective bits, not 4.
    """
    per_token = cfg.kv_bytes_per_token // 2  # kv_bytes_per_token is at fp16
    return int(per_token * context_length * bytes_per_element)


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
        self._kv_cache_bytes = 0  # nothing allocates a cache yet; #14 will

    # -- loading ------------------------------------------------------------

    def load_weights(self) -> None:
        """Read the checkpoint and put it on the device as fp16.

        Every tensor is range-checked on the way. bf16 carries float32's
        exponent and reaches far past fp16's 65504; a weight over that line
        would become an infinity and poison everything downstream in silence.
        """
        path = self.model_dir / "model.safetensors"
        rejected: dict[str, str] = {}

        # Built locally and committed only on success. Uploading into
        # self._tensors as we go would leave a half-loaded model observable
        # through `tensors` and `footprint()` after the raise, reported as
        # though it were whole.
        loaded: dict[str, _microinfer.DeviceTensor] = {}
        for name, values in weights.iter_tensors(path):
            reason = weights.check_fp16_range(name, values)
            if reason is not None:
                rejected[name] = reason
                continue
            flat = np.ascontiguousarray(values, dtype=np.float32).reshape(-1)
            loaded[name] = _microinfer.upload_fp16(flat)

        if rejected:
            detail = "\n".join(f"    {n}: {why}" for n, why in rejected.items())
            raise weights.WeightError(
                f"{len(rejected)} tensor(s) cannot be stored as fp16 and were "
                f"not uploaded:\n{detail}\n  The checkpoint is bf16, which "
                f"carries float32's exponent and reaches far past fp16's "
                f"{weights.FP16_MAX:.0f}. Converting anyway would put an "
                f"infinity or a NaN into the weights, silently."
            )

        self._tensors = loaded

    @property
    def tensors(self) -> dict[str, _microinfer.DeviceTensor]:
        return dict(self._tensors)

    # -- accounting ---------------------------------------------------------

    def footprint(self) -> Footprint:
        """What the engine currently holds. Measurement only.

        Takes no context length on purpose. An earlier version accepted one and
        folded the *projected* cache into the same object, which made
        `unaccounted` subtract bytes nobody had allocated — it shrank as the
        projection grew and went negative past a large enough context. A number
        whose job is to separate causes cannot itself mix them. Projections go
        through `project_kv_cache`.
        """
        info = _microinfer.device_memory_info()
        return Footprint(
            weights=sum(t.nbytes for t in self._tensors.values()),
            kv_cache=self._kv_cache_bytes,
            workspace=self._workspace_bytes,
            device_free=info["free"],
            device_total=info["total"],
        )

    def project_kv_cache(self, context_length: int, bytes_per_element: float = 2.0) -> int:
        """What a cache of that length would cost. Nothing is allocated."""
        return kv_cache_bytes(self.config, context_length, bytes_per_element)

    def expected_weight_bytes(self) -> int:
        return expected_weight_bytes(self.config)
