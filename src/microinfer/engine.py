"""The engine's Seam A: load a model, run it, and know what it costs.

`forward` and `generate` are the Milestone 0 checkpoint (#12): a from-scratch
FP16 inference path over a simple contiguous KV cache, one sequence at a time.
There is no batch dimension anywhere, by design; the thesis is about a single
user's cache under contention, and batching machinery would be code with no
experiment behind it.
"""

from __future__ import annotations

import json
import threading
from contextlib import contextmanager
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import Callable

import numpy as np

from . import _microinfer, model, weights
from .config import ConfigMismatch, ModelConfig
from .footprint import Footprint
from .models import VERIFIED


def expected_weight_shapes(cfg: ModelConfig) -> dict[str, tuple[int, ...]]:
    """Every tensor the checkpoint must hold, by its name there, with its
    shape: implied by config.json alone.

    Derived rather than read, so that a mismatch against the real checkpoint
    is a finding rather than a tautology. Qwen2 puts a bias on the query, key
    and value projections and none on the output projection; a model that
    ties its embeddings stores no LM head.
    """
    h, i, vocab = cfg.hidden_size, cfg.intermediate_size, cfg.vocab_size
    q_dim = cfg.num_attention_heads * cfg.head_dim
    kv_dim = cfg.num_key_value_heads * cfg.head_dim

    shapes = {"model.embed_tokens.weight": (vocab, h)}
    for layer in range(cfg.num_hidden_layers):
        p = f"model.layers.{layer}."
        shapes.update({
            p + "input_layernorm.weight": (h,),
            p + "self_attn.q_proj.weight": (q_dim, h),
            p + "self_attn.q_proj.bias": (q_dim,),
            p + "self_attn.k_proj.weight": (kv_dim, h),
            p + "self_attn.k_proj.bias": (kv_dim,),
            p + "self_attn.v_proj.weight": (kv_dim, h),
            p + "self_attn.v_proj.bias": (kv_dim,),
            p + "self_attn.o_proj.weight": (h, q_dim),
            p + "post_attention_layernorm.weight": (h,),
            p + "mlp.gate_proj.weight": (i, h),
            p + "mlp.up_proj.weight": (i, h),
            p + "mlp.down_proj.weight": (h, i),
        })
    shapes["model.norm.weight"] = (h,)
    if not cfg.tie_word_embeddings:
        shapes["lm_head.weight"] = (vocab, h)
    return shapes


FP16_BYTES = np.dtype(np.float16).itemsize


def expected_weight_bytes(cfg: ModelConfig) -> int:
    """Parameter bytes implied by config.json alone, at fp16."""
    return FP16_BYTES * sum(int(np.prod(shape)) for shape in expected_weight_shapes(cfg).values())


@dataclass(frozen=True)
class WeightLayout:
    """Where each weight goes in the engine's one weight arena (#23), and how
    big the arena is."""

    offsets: dict[str, int]
    nbytes: int


def weight_layout(cfg: ModelConfig) -> WeightLayout:
    """Each weight's byte offset in the arena, in expected_weight_shapes'
    order, each a multiple of the alignment every weight is given
    (weight_alignment, kernels.h); and the arena's size, rounded up to it too.
    From config.json alone, so the arena is sized before anything is read or
    uploaded."""
    align = _microinfer.weight_alignment

    def aligned(n: int) -> int:
        return -(-n // align) * align

    offsets, end = {}, 0
    for name, shape in expected_weight_shapes(cfg).items():
        offsets[name] = aligned(end)
        end = offsets[name] + FP16_BYTES * int(np.prod(shape))
    return WeightLayout(offsets, aligned(end))


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


def paged_cache_ranges(cfg: ModelConfig, context_length: int, tier: str = "FP16") -> list[int]:
    """The bytes the paged cache's pages need for `context_length` positions
    at a static tier (#18), one figure per address range they sit in:
    computed from the layout, before the driver rounds anything.

    At FP16 there is one range, with a page for every started span of P
    positions in every layer. At a quantised tier there are two: a page for
    every *full* span at the tier's size, scale metadata included, and two
    FP16 open pages per layer for the rest (ADR-0005, ADR-0011)."""
    Tier = _microinfer.Tier
    page_tokens = _microinfer.device.page_tokens
    layers = cfg.num_hidden_layers
    page_bytes = model.tier_page_bytes(cfg)
    fp16 = page_bytes[int(Tier.FP16)]
    if tier == "FP16":
        return [layers * -(-context_length // page_tokens) * fp16]
    page = page_bytes[int(getattr(Tier, tier))]
    open_pages = layers * len(_microinfer.device.open_pages) * fp16 if context_length > 0 else 0
    return [layers * (context_length // page_tokens) * page, open_pages]


def paged_cache_bytes(cfg: ModelConfig, context_length: int, tier: str = "FP16") -> int:
    """What the paged cache's pages take from the driver for `context_length`
    positions at a static tier: paged_cache_ranges, each rounded up to whole
    granules, since the driver backs each range in granules (ADR-0007). The
    RoPE table beside the pages is not included; it is the same at every
    tier."""
    granule = _microinfer.granule_bytes()
    return sum(-(-n // granule) * granule for n in paged_cache_ranges(cfg, context_length, tier))


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

    #: Which halves of a page a quantised tier quantises. "both" is the tier;
    #: "keys" and "values" are a diagnostic that splits its cost (kv_pages.h).
    KV_HALVES = ("both", "keys", "values")

    def __init__(self, model_dir: str | Path, *, verify: bool = True, kv_cache: str = "paged",
                 kv_tier: str = "FP16", kv_halves: str = "both",
                 prefill_chunk: int | None = DEFAULT_PREFILL_CHUNK):
        if kv_cache not in self.KV_CACHES:
            raise ValueError(f"kv_cache is one of {self.KV_CACHES}, got {kv_cache!r}")
        if prefill_chunk is not None and prefill_chunk < 1:
            raise ValueError(f"prefill_chunk must be positive or None, got {prefill_chunk}")
        self.kv_cache = kv_cache
        self.kv_tier = kv_tier
        self.kv_halves = kv_halves
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
        self._arena: _microinfer.DeviceArena | None = None
        self._model: model.Model | None = None
        self._workspace_bytes = 0
        self._cache = None
        self._peak: Footprint | None = None

    # -- loading ------------------------------------------------------------

    def load_weights(self) -> None:
        """Read the checkpoint and put it on the device as fp16, in one arena.

        The checkpoint's tensors are checked against the ones config.json
        implies, by name and shape, before anything is allocated. Then one
        arena is allocated, sized from config.json (weight_layout), and each
        tensor is uploaded into it at its offset (#23): one allocation, where
        one per tensor cost several per cent of the weights' size (ADR-0007,
        note from #23).

        Every tensor is range-checked on the way. bf16 carries float32's
        exponent and reaches far past fp16's 65504; a weight over that line
        would become an infinity and poison everything downstream in silence.
        """
        path = self.model_dir / "model.safetensors"
        weights.check_shapes(path, expected_weight_shapes(self.config), "config.json")

        layout = weight_layout(self.config)
        arena = _microinfer.DeviceArena(layout.nbytes)
        rejected: dict[str, str] = {}

        # Built locally and committed only on success. Uploading into
        # self._tensors as we go would leave a half-loaded model observable
        # through `tensors` and `footprint()` after the raise, reported as
        # though it were whole. On a raise, the arena goes with the last
        # tensor in it.
        loaded: dict[str, _microinfer.DeviceTensor] = {}
        for name, values in weights.iter_tensors(path):
            reason = weights.check_fp16_range(name, values)
            if reason is not None:
                rejected[name] = reason
                continue
            flat = np.ascontiguousarray(values, dtype=np.float32).reshape(-1)
            loaded[name] = arena.put(layout.offsets[name], flat)

        if rejected:
            detail = "\n".join(f"    {n}: {why}" for n, why in rejected.items())
            raise weights.WeightError(
                f"{len(rejected)} tensor(s) cannot be stored as fp16 and were "
                f"not uploaded:\n{detail}\n  The checkpoint is bf16, which "
                f"carries float32's exponent and reaches far past fp16's "
                f"{weights.FP16_MAX:.0f}. Converting anyway would put an "
                f"infinity or a NaN into the weights, silently."
            )

        self._arena = arena
        self._tensors = loaded
        self._model = model.Model(self.config, model.Weights.from_tensors(self.config, loaded))

    @property
    def weight_arena(self) -> _microinfer.DeviceArena | None:
        """The one allocation the weights live in, once loaded (#23)."""
        return self._arena

    @property
    def kv_tier(self) -> str:
        """The tier every page of a cache is held at (#18). Static within a
        cache: nothing changes a page's tier after it is allocated. Setting it
        applies to the next cache, and is checked as the constructor checks
        it."""
        return self._kv_tier

    @kv_tier.setter
    def kv_tier(self, tier: str) -> None:
        if tier not in self.KV_TIERS:
            raise ValueError(f"kv_tier is one of {self.KV_TIERS}, got {tier!r}")
        if self.kv_cache == "contiguous" and tier != "FP16":
            raise ValueError("the contiguous cache holds FP16 only; quantised tiers are on pages")
        self._kv_tier = tier

    @property
    def kv_halves(self) -> str:
        """Which halves of a page the tier quantises: "both", or, as a
        diagnostic, "keys" or "values" alone (kv_pages.h)."""
        return self._kv_halves

    @kv_halves.setter
    def kv_halves(self, halves: str) -> None:
        if halves not in self.KV_HALVES:
            raise ValueError(f"kv_halves is one of {self.KV_HALVES}, got {halves!r}")
        self._kv_halves = halves

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

    def hold(self, prompt, *, context: int | None = None, stop: threading.Event,
             report: Callable[[str, int, int | None], None] | None = None) -> None:
        """Keep a generation session in progress until `stop` is set: RQ1's
        engine under contention (#51).

        The prompt is prefilled, then tokens are decoded greedily to position
        `context`, the model's window by default. There the session goes back
        to the end of the prompt and decodes the same span again, from the
        last token it produced, over the pages it already holds: from the end
        of the first pass the cache holds the full context, and holds no
        more however long the session runs. Each pass reads only the prompt
        and itself, as a fresh generation from that token would.

        Going back rewrites positions in place, which only FP16 pages allow: a
        quantised page is sealed once its positions are written (ADR-0011).

        `report(state, position, token)` is called after each prefill chunk,
        as ("prefilling", positions so far, None), and after each decoded
        token, as ("decoding", positions in the cache, the token). The cache
        is released when the session stops, and `stop` is checked between
        steps, so it stops within one.
        """
        if self.kv_tier != "FP16" or self.kv_halves != "both":
            raise ValueError("holding rewrites positions in place, which only FP16 pages allow")
        ids = self.encode(prompt) if isinstance(prompt, str) else self._check_ids(prompt)
        context = self.config.max_position_embeddings if context is None else context
        if not len(ids) < context:
            raise ValueError(f"the context, {context}, must exceed the prompt's "
                             f"{len(ids)} positions, to leave a span to decode")
        self._check_window(context)
        report = report or (lambda *_: None)

        cache = self._new_cache(capacity=context)
        token = None
        for chunk, ws, _, last in self._prefill(cache, ids):
            report("prefilling", cache.length, None)
            if last:
                token = self._model.greedy_last(ws, len(chunk))

        step = model.Workspace(self.config, rows=1)
        with self._holding(cache, step):
            while not stop.is_set():
                if cache.length == context:
                    cache.length = len(ids)  # back to the end of the prompt
                self._model.run(step, cache, np.array([token], np.int32))
                token = self._model.greedy_last(step, 1)
                report("decoding", cache.length, token)

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
        tier = getattr(_microinfer.Tier, self.kv_tier)
        if self.kv_halves == "both":
            return model.PagedCache(self.config, tier)
        if tier == _microinfer.Tier.FP16:
            raise ValueError("kv_halves splits a quantised tier; FP16 has nothing to split")
        return model.PagedCache(self.config, tier,
                                getattr(_microinfer.device.Halves, self.kv_halves.title()))

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
