"""The engine's Seam A: load a model, run it, and know what it costs.

`forward` and `generate` are the Milestone 0 checkpoint (#12): a from-scratch
FP16 inference path over a simple contiguous KV cache, one sequence at a time.
There is no batch dimension anywhere, by design; the thesis is about a single
user's cache under contention, and batching machinery would be code with no
experiment behind it.
"""

from __future__ import annotations

import json
from contextlib import contextmanager
from functools import cached_property
from pathlib import Path

import numpy as np

from . import _microinfer, model, weights
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

    It is also the raw bytes, not what the paged cache takes from the driver.
    That is whole granules for whole pages (ADR-0007, #14), plus the RoPE
    table that completes the keys (ADR-0009); `Engine.footprint` reports it.
    """
    per_token = cfg.kv_bytes_per_token // 2  # kv_bytes_per_token is at fp16
    return int(per_token * context_length * bytes_per_element)


class Engine:
    """Seam A. Everything a test or a caller touches goes through here."""

    #: The KV cache layouts: "paged", the engine's own (#14), and "contiguous",
    #: the Milestone 0 cache it replaced, kept as the reference the paged one
    #: is proven bit-identical to.
    KV_CACHES = ("paged", "contiguous")

    #: Tokens per prefill chunk (#15). A long prompt is prefilled this many
    #: positions at a time, so the workspace is sized to a chunk, not to the
    #: prompt. With it, Qwen2.5-1.5B prefills its whole 32K-token window on a
    #: 6 GiB card (benchmark log, prefill-throughput at 32768 tokens).
    DEFAULT_PREFILL_CHUNK = 512

    #: The precision tiers the paged cache can be held at, statically: every
    #: page at one tier for the life of a cache (#18, ADR-0008).
    KV_TIERS = ("FP16", "INT8", "INT4", "INT2")

    def __init__(self, model_dir: str | Path, *, verify: bool = True, kv_cache: str = "paged",
                 kv_tier: str = "FP16", prefill_chunk: int | None = DEFAULT_PREFILL_CHUNK):
        if kv_cache not in self.KV_CACHES:
            raise ValueError(f"kv_cache is one of {self.KV_CACHES}, got {kv_cache!r}")
        if kv_tier not in self.KV_TIERS:
            raise ValueError(f"kv_tier is one of {self.KV_TIERS}, got {kv_tier!r}")
        if kv_cache == "contiguous" and kv_tier != "FP16":
            raise ValueError("the contiguous cache holds FP16 only; quantised tiers are on pages")
        if prefill_chunk is not None and prefill_chunk < 1:
            raise ValueError(f"prefill_chunk must be positive or None, got {prefill_chunk}")
        self.kv_cache = kv_cache
        #: The tier every page of a cache is held at (#18). Static: nothing
        #: changes a page's tier after it is allocated.
        self.kv_tier = kv_tier
        #: None prefills a prompt in one step, with a workspace for all of it.
        self.prefill_chunk = prefill_chunk
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
        self._model: model.Model | None = None
        self._workspace_bytes = 0
        self._cache = None
        self._peak: Footprint | None = None

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
        self._model = model.Model(self.config, model.Weights.from_tensors(self.config, loaded))

    @property
    def tensors(self) -> dict[str, _microinfer.DeviceTensor]:
        return dict(self._tensors)

    # -- text ---------------------------------------------------------------

    @cached_property
    def tokenizer(self):
        """The checkpoint's own tokenizer.json, through HuggingFace's
        `tokenizers`: a Rust library with no torch and no CUDA, so it can sit
        in the engine's process (ADR-0002)."""
        from tokenizers import Tokenizer

        return Tokenizer.from_file(str(self.model_dir / "tokenizer.json"))

    def encode(self, text: str) -> np.ndarray:
        return np.asarray(self.tokenizer.encode(text).ids, dtype=np.int32)

    def decode(self, token_ids) -> str:
        return self.tokenizer.decode([int(t) for t in token_ids])

    @cached_property
    def eos_token_ids(self) -> frozenset[int]:
        """Where generation stops: generation_config.json's eos_token_id, which
        for the instruct models is both <|im_end|> and <|endoftext|>."""
        eos = json.loads((self.model_dir / "generation_config.json").read_text())["eos_token_id"]
        return frozenset(eos if isinstance(eos, list) else [eos])

    # -- inference ----------------------------------------------------------

    def forward(self, token_ids, *, capture_hidden_states: bool = False):
        """Logits for every position of one sequence, as fp32 (seq, vocab).

        With `capture_hidden_states=True`, returns `(logits, hidden_states)`,
        where hidden_states is (layers + 1, seq, hidden) in HuggingFace's
        `output_hidden_states` order: the per-layer diagnostic ADR-0006 runs
        when the gate goes red.
        """
        ids = self._check_ids(token_ids)
        self._check_window(len(ids))
        cache = self._new_cache(capacity=len(ids))
        logits, states = [], []
        chunks = self._prefill(cache, ids, hidden_states=capture_hidden_states)
        for chunk, ws, captured, _ in chunks:
            logits.append(self._model.logits(ws, len(chunk)))
            if capture_hidden_states:
                states.append(np.stack(captured))
        logits = np.concatenate(logits)
        if capture_hidden_states:
            # Each chunk captured every state for its own positions.
            return logits, np.concatenate(states, axis=1)
        return logits

    def generate(self, prompt, max_new_tokens: int = 64, *, stop_at_eos: bool = True) -> np.ndarray:
        """Greedy continuation of one prompt: the new token ids only.

        `prompt` is text, which is tokenised, or token ids. The prompt is
        prefilled in chunks of `prefill_chunk` positions, then each new token
        is decoded against the cache. Stops after `max_new_tokens`, or at an end-of-sequence token,
        which is included, as HuggingFace's generate includes it.
        """
        ids = self.encode(prompt) if isinstance(prompt, str) else self._check_ids(prompt)
        if max_new_tokens < 0:
            raise ValueError(f"max_new_tokens must not be negative, got {max_new_tokens}")
        if max_new_tokens == 0:
            return np.empty(0, dtype=np.int32)

        # The last new token is never fed back, so the cache needs one row less
        # than prompt plus output.
        self._check_window(len(ids) + max_new_tokens - 1)
        cache = self._new_cache(capacity=len(ids) + max_new_tokens - 1)
        out: list[int] = []
        for chunk, ws, _, last in self._prefill(cache, ids):
            if last:  # the prompt's last row decides the first new token
                out.append(self._model.greedy_last(ws, len(chunk)))

        step = model.Workspace(self.config, rows=1)
        with self._holding(cache, step):
            while len(out) < max_new_tokens and not (stop_at_eos and out[-1] in self.eos_token_ids):
                self._model.run(step, cache, np.array(out[-1:], np.int32))
                out.append(self._model.greedy_last(step, 1))
        return np.asarray(out, dtype=np.int32)

    def cached_kv(self, token_ids) -> tuple[np.ndarray, np.ndarray]:
        """What the cache holds after prefilling one sequence: keys and values,
        each fp32 (layers, seq, kv_heads, head_dim). The keys are as stored,
        rotated and without their projection's bias (ADR-0009), which is what
        a precision tier rounds (#16).

        Prefilled through the contiguous cache, whose buffers read back as
        they are; the paged cache holds the same bits (#14)."""
        ids = self._check_ids(token_ids)
        self._check_window(len(ids))
        cache = model.ContiguousCache(self.config, capacity=len(ids))
        for _ in self._prefill(cache, ids):
            pass
        shape = (len(ids), self.config.num_key_value_heads, self.config.head_dim)
        return tuple(np.stack([t.to_numpy().reshape(shape) for t in buffers])
                     for buffers in (cache.keys, cache.values))

    def _check_ids(self, token_ids) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("no weights on the device; call load_weights() first")
        ids = np.asarray(token_ids)
        if ids.ndim != 1:
            raise ValueError(
                f"token_ids must be one sequence, 1-D; got shape {ids.shape}. The engine "
                f"runs a single sequence at a time and has no batch dimension.")
        if ids.size == 0:
            raise ValueError("token_ids is empty")
        if not np.issubdtype(ids.dtype, np.integer):
            raise ValueError(f"token_ids must be integers, got {ids.dtype}")
        if ids.min() < 0 or ids.max() >= self.config.vocab_size:
            raise ValueError(f"token ids must lie in [0, {self.config.vocab_size}); "
                             f"got [{ids.min()}, {ids.max()}]")
        return ids.astype(np.int32)

    def _check_window(self, positions: int) -> None:
        """Refuse, before any work, a sequence the model was not built for: past
        max_position_embeddings its RoPE angles were never trained, and the
        paged cache reserves exactly that many positions."""
        window = self.config.max_position_embeddings
        if positions > window:
            raise ValueError(f"{positions} positions exceed the model's context window of "
                             f"{window}: shorten the prompt or ask for fewer new tokens")

    def _prefill(self, cache, ids: np.ndarray, *, hidden_states: bool = False):
        """Run `ids` into `cache` a chunk at a time. Yields, per chunk: the
        chunk's ids; the workspace holding its final-normed states, which the
        next chunk overwrites; the hidden states it captured, if asked for,
        else None; and whether it is the prompt's last chunk.

        One workspace serves every chunk, sized to the chunk rather than to the
        prompt (#15). The cache already continues a sequence from its length:
        each chunk's positions, its RoPE angles and its causal mask follow from
        where the one before it stopped."""
        n = len(ids)
        size = n if self.prefill_chunk is None else min(self.prefill_chunk, n)
        ws = model.Workspace(self.config, rows=size)
        with self._holding(cache, ws):
            for start in range(0, n, size):
                chunk = ids[start:start + size]
                captured = [] if hidden_states else None
                self._model.run(ws, cache, chunk, captured)
                yield chunk, ws, captured, start + size >= n

    def _new_cache(self, capacity: int):
        """A cache for one sequence. The contiguous one is sized for `capacity`
        up front; the paged one ignores it and takes pages as positions arrive,
        so a generation that stops early never held room for the rest."""
        if self.kv_cache == "contiguous":
            return model.ContiguousCache(self.config, capacity)
        return model.PagedCache(self.config, getattr(_microinfer.Tier, self.kv_tier))

    @contextmanager
    def _holding(self, cache, ws: model.Workspace):
        """Account for a cache and a workspace while they are alive, and note
        the peak: the footprint at the moment the engine held the most.

        Read after the work, not before it. Memory taken lazily while the work
        runs, such as cuBLAS's workspace on first use and the device copies of
        token ids and positions, then shows in the device reading. It shows as
        `unaccounted`, since the engine did not allocate it, but it is not
        missed."""
        self._cache = cache
        self._workspace_bytes = ws.nbytes
        try:
            yield
            now = self.footprint()
            if self._peak is None or now.engine_total > self._peak.engine_total or (
                    now.engine_total == self._peak.engine_total
                    and now.device_free < self._peak.device_free):
                self._peak = now
        finally:
            self._cache = None
            self._workspace_bytes = 0

    def peak_footprint(self) -> Footprint | None:
        """The footprint when the engine held the most device memory, or None
        if nothing has run: the reading taken after the largest cache and
        workspace had done their work."""
        return self._peak

    def reset_peak(self) -> None:
        self._peak = None

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
            # Read now, not when the cache was made: a paged cache grows.
            kv_cache=self._cache.nbytes if self._cache is not None else 0,
            workspace=self._workspace_bytes,
            device_free=info["free"],
            device_total=info["total"],
        )

    def project_kv_cache(self, context_length: int, bytes_per_element: float = 2.0) -> int:
        """What a cache of that length would cost. Nothing is allocated."""
        return kv_cache_bytes(self.config, context_length, bytes_per_element)

    def expected_weight_bytes(self) -> int:
        return expected_weight_bytes(self.config)
