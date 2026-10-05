"""The engine's Seam A: load a model, run it, and know what it costs.

`forward` and `generate` are the Milestone 0 checkpoint (#12): a from-scratch
FP16 inference path over a simple contiguous KV cache, one sequence at a time.
There is no batch dimension anywhere, by design; the thesis is about a single
user's cache under contention, and batching machinery would be code with no
experiment behind it.
"""

from __future__ import annotations

import collections
import json
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import Callable

import numpy as np

from . import _microinfer, controller, model, nvml, weights
from .config import ConfigMismatch, ModelConfig
from .footprint import Footprint
from .models import VERIFIED
from .monitor import DEFAULT as DEFAULT_THRESHOLDS
from .monitor import GREEN, RED, YELLOW, Monitor, PressureEvent, Thresholds
from .score_sources import SOURCES, score_array


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


def tier_map_of(names: list[list[str]] | None) -> list[list]:
    """A tier map of tier names as KVPages takes it, of Tiers; None is
    the empty map."""
    return [[getattr(_microinfer.Tier, name) for name in row] for row in names or []]


def paged_cache_ranges(cfg: ModelConfig, context_length: int, tier: str = "FP16",
                       tier_map: list[list[str]] | None = None) -> list[int]:
    """The bytes the paged cache's pages need for `context_length` positions,
    at a static tier (#18) or built from a tier map of tier names (#92), one
    figure per tier's address range, indexed by Tier: computed from the
    layout, before the driver rounds anything.

    A cache that does not seal, at FP16 with no other tier in its map, has a
    page for every started span of P positions in every layer, all at FP16.
    One that seals has a page for every *full* span at its birth tier's size,
    scale metadata included, and two FP16 open pages per layer for the rest
    (ADR-0005, ADR-0011), in the FP16 range beside any page born there."""
    device = _microinfer.device
    page_tokens = device.page_tokens
    layers = cfg.num_hidden_layers
    page_bytes = model.tier_page_bytes(cfg)
    fp16 = int(_microinfer.Tier.FP16)
    cache_tier = getattr(_microinfer.Tier, tier)
    rows = tier_map_of(tier_map) or [[] for _ in range(layers)]
    ranges = [0] * len(page_bytes)
    if not device.KVPages.seals_for(cache_tier, rows):
        ranges[fp16] = layers * -(-context_length // page_tokens) * page_bytes[fp16]
        return ranges
    for row in rows:
        for i in range(context_length // page_tokens):
            born = int(row[i] if i < len(row) else cache_tier)
            ranges[born] += page_bytes[born]
    if context_length > 0:
        ranges[fp16] += layers * len(device.open_pages) * page_bytes[fp16]
    return ranges


def paged_cache_bytes(cfg: ModelConfig, context_length: int, tier: str = "FP16",
                      tier_map: list[list[str]] | None = None) -> int:
    """What the paged cache's pages take from the driver for `context_length`
    positions, at a static tier or built from a tier map: paged_cache_ranges,
    each rounded up to whole granules, since the driver backs each range in
    granules (ADR-0007). The RoPE table beside the pages is not included; it
    is the same at every tier."""
    granule = _microinfer.granule_bytes()
    return sum(-(-n // granule) * granule
               for n in paged_cache_ranges(cfg, context_length, tier, tier_map))


#: What Engine.hold reports after each prefill chunk and each decoded token.
PREFILLING, DECODING = "prefilling", "decoding"
#: How a generation ended (#107): every token asked for, or up to an end of
#: sequence; or stopped gracefully, memory exhausted.
COMPLETE, EXHAUSTED = "complete", "exhausted"


@dataclass(frozen=True)
class Ending:
    """How a generation ended: its state, COMPLETE or EXHAUSTED, the
    positions its cache held, the tokens it kept, and, if exhausted, why."""

    state: str
    positions: int
    tokens: np.ndarray
    reason: str = ""


class _Exhausted(Exception):
    """A step's allocation failed and an emergency plan did not answer it."""


@dataclass(frozen=True)
class PressureRecord:
    """A pressure event as the engine drained it: how many positions its
    cache held then, and when, on the monitor's clock (time.monotonic_ns)."""

    event: PressureEvent
    positions_held: int
    drained_ns: int

    @property
    def waited_ns(self) -> int:
        """How long the event waited for the step boundary that drained it."""
        return self.drained_ns - self.event.t_mono_ns


def pressure_chunk(seconds_per_position: float, *, seconds: float, tile: int,
                   cap: int | None) -> int:
    """The positions of a prefill chunk run under pressure (#135): as many
    as `seconds` holds at `seconds_per_position`, the time a position of
    the chunk before took, in whole `tile`s, the attention kernel's query
    tile (ADR-0011); one tile at least, though it may take longer, and no
    more than `cap`, the engine's prefill_chunk, if it has one."""
    if seconds_per_position <= 0:
        raise ValueError(f"a time per position is positive; got {seconds_per_position}")
    tiles = int(seconds / seconds_per_position) // tile
    positions = max(1, tiles) * tile
    return positions if cap is None else min(positions, cap)


#: The least room a YELLOW step keeps in its budget for its next downgrade.
MOVE_MARGIN_SECONDS = 0.002
#: The empty granules an adaptive cache keeps mapped at each quantised tier,
#: so that an emergency plan's first downgrades have room (#107).
EMERGENCY_SPARES = 1
#: How often, at GREEN with no plan in hand, the engine looks for upgrades to
#: make (#106): not every step, most of which would find none.
UPGRADE_RETRY_SECONDS = 0.25


@dataclass
class AppliedBatch:
    """The part of a plan one step applied: its downgrades [start, end),
    the step's wall time for them, the plan's making included in the step
    that made it, and the memory the cache held before and after, by its
    allocator and, when the engine measures plans, by the driver's account
    of this process."""

    start: int
    end: int
    seconds: float
    cache_bytes_before: int
    cache_bytes_after: int
    own_bytes_before: int | None = None
    own_bytes_after: int | None = None
    #: When the step finished, on time.monotonic().
    at_seconds: float = 0.0


@dataclass
class PlanRecord:
    """A plan the engine made, downgrades or upgrades, with the pressure
    event that caused it, and what became of it: the batches applied, the
    moves skipped (their page had moved since), and how it ended,
    "applied", "cancelled at GREEN", "replaced" by a later plan,
    "cancelled by pressure" (upgrades), or the error that stopped it. An
    emergency plan answers an allocation that failed, not an event: its
    `pressure` is the last event, if any, and `emergency` the error."""

    pressure: PressureRecord | None
    plan: controller.Plan | controller.UpgradePlan
    planning_seconds: float
    batches: list[AppliedBatch]
    skipped: int = 0
    ended: str = ""
    #: The headroom the plan was made for: the event's, or, for a plan made
    #: because pressure persisted, the monitor's latest reading.
    headroom_bytes: int = 0
    #: Made not on an event but because the level stayed YELLOW or RED while
    #: the cache sealed more pages.
    persisting: bool = False
    #: The OutOfMemory an emergency plan answers (#107); None for any other.
    emergency: str | None = None

    @property
    def applied(self) -> int:
        return sum(b.end - b.start for b in self.batches)

    @property
    def upgrades(self) -> bool:
        return isinstance(self.plan, controller.UpgradePlan)


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

    #: The precision tiers a page of the paged cache can be held at (ADR-0008).
    #: A page keeps its tier for the life of a cache: kv_tier (#18), or the
    #: one kv_tier_map names for it (#92).
    KV_TIERS = ("FP16", "INT8", "INT4", "INT2")

    #: The time a prefill chunk aims at under pressure (#135). A chunk sized
    #: from the one before can overrun it, a position costing more as the
    #: context grows, so it is half #135's target of 1 s from an event to
    #: its plan.
    DEFAULT_PRESSURE_CHUNK_SECONDS = 0.5

    #: With kv_scoring, how many decode steps apart the scores take a step's
    #: attention mass (#100): every step cost Qwen2.5-0.5B's decode more
    #: than noise at 512 positions, and #88 makes every R-th its fallback.
    DEFAULT_SCORE_EVERY = 4

    #: Which halves of a page a quantised tier quantises. "both" is the tier;
    #: "keys" and "values" are a diagnostic that splits its cost (kv_pages.h).
    KV_HALVES = ("both", "keys", "values")

    def __init__(self, model_dir: str | Path, *, verify: bool = True, kv_cache: str = "paged",
                 kv_tier: str = "FP16", kv_halves: str = "both",
                 kv_tier_map: list[list[str]] | None = None, kv_scoring: bool = False,
                 kv_score_every: int | None = None, kv_adaptive: bool = False,
                 kv_score_source: str = "semantic", kv_score_seed: int = 0,
                 kv_plan_budget_seconds: float = controller.DEFAULT_BUDGET_SECONDS,
                 kv_upgrade_cooldown_seconds: float = controller.DEFAULT_UPGRADE_COOLDOWN_SECONDS,
                 prefill_chunk: int | None = DEFAULT_PREFILL_CHUNK,
                 pressure_chunk_seconds: float = DEFAULT_PRESSURE_CHUNK_SECONDS):
        if kv_cache not in self.KV_CACHES:
            raise ValueError(f"kv_cache is one of {self.KV_CACHES}, got {kv_cache!r}")
        if prefill_chunk is not None and prefill_chunk < 1:
            raise ValueError(f"prefill_chunk must be positive or None, got {prefill_chunk}")
        if pressure_chunk_seconds <= 0:
            raise ValueError(f"pressure_chunk_seconds must be positive, got "
                             f"{pressure_chunk_seconds}")
        self.kv_cache = kv_cache
        self.kv_tier = kv_tier
        self.kv_halves = kv_halves
        #: None prefills a prompt in one step, with a workspace for all of it.
        self.prefill_chunk = prefill_chunk
        #: The time a prefill chunk aims at while an adaptive engine's
        #: monitor reads YELLOW or RED (#135): each such chunk is sized from
        #: the chunk before it, so that the next event is drained, and
        #: planned for, within about this long.
        self.pressure_chunk_seconds = pressure_chunk_seconds
        self.model_dir = Path(model_dir)
        self.config = ModelConfig.from_model_dir(self.model_dir)
        self.kv_tier_map = kv_tier_map
        self.kv_scoring = kv_scoring
        self.kv_score_every = (self.DEFAULT_SCORE_EVERY if kv_score_every is None
                               else kv_score_every)
        if kv_score_source not in SOURCES:
            raise ValueError(f"kv_score_source is one of {SOURCES}, got {kv_score_source!r}")
        if kv_adaptive and (kv_cache == "contiguous" or kv_halves != "both"):
            raise ValueError("an adaptive cache is paged, and no diagnostic kv_halves")
        #: Whether pressure events make the engine downgrade pages (#105):
        #: on YELLOW or RED, a plan from the scores, applied between steps.
        self.kv_adaptive = kv_adaptive
        #: Where a plan's scores come from (#104), and the random one's seed.
        self.kv_score_source, self.kv_score_seed = kv_score_source, kv_score_seed
        #: The time a YELLOW plan, or upgrades at GREEN, may take of one step,
        #: planning included.
        self.kv_plan_budget_seconds = kv_plan_budget_seconds
        #: How long after the last downgrade an upgrade waits (#103, #106).
        self.kv_upgrade_cooldown_seconds = kv_upgrade_cooldown_seconds
        #: Read what this process holds by the driver's account around every
        #: batch a plan applies; off, as each reading costs a millisecond.
        self.measure_plans = False
        #: Every plan made, with its event and what became of it.
        self.plans: list[PlanRecord] = []
        #: When the last downgrade was applied, on time.monotonic().
        self.last_downgrade_seconds: float | None = None
        #: When pressure last ended, the monitor's GREEN after a YELLOW or a
        #: RED, on time.monotonic(): the upgrade cooldown runs from this or
        #: the last downgrade, the later (#106).
        self.pressure_ended_seconds: float | None = None
        self._pending: PlanRecord | None = None
        self._cursor = 0
        self._last_pressure: PressureRecord | None = None
        self._planned_pages = 0
        self._upgrades_looked_at: float | None = None
        #: How long the last downgrades took, the longest of them the guess
        #: for the next: an outlier passes out of it, as none would of a max.
        self._recent_moves: collections.deque[float] = collections.deque(maxlen=64)
        self._thresholds: Thresholds | None = None
        #: How the last generation ended (#107); None before the first, and
        #: while one runs.
        self.ending: Ending | None = None

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
        self._monitor: Monitor | None = None
        #: Every pressure event drained, in order (start_monitor).
        self.pressure_events: list[PressureRecord] = []
        #: The error a monitor stopped on, if one did.
        self.monitor_error: BaseException | None = None
        self._positions_held = 0  # by the last cache drained beside

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
        """The tier every page of a cache is held at (#18), or, with a
        kv_tier_map, every page beyond the map's rows. Static within a cache:
        nothing changes a page's tier after it is allocated. Setting it
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
        # Before the constructor sets a map, there is none to check.
        if halves != "both" and getattr(self, "_kv_tier_map", None) is not None:
            raise ValueError("a diagnostic kv_halves rounds every page at kv_tier, and "
                             "takes no tier map")
        self._kv_halves = halves

    @property
    def kv_scoring(self) -> bool:
        """Whether a cache keeps the importance score of its pages (#99),
        folding each decode step's attention mass into it on the device.
        Setting it applies to the next cache."""
        return self._kv_scoring

    @kv_scoring.setter
    def kv_scoring(self, scoring: bool) -> None:
        if scoring and self.kv_cache == "contiguous":
            raise ValueError("the contiguous cache has no pages to score")
        self._kv_scoring = bool(scoring)

    @property
    def kv_score_every(self) -> int:
        """With kv_scoring, how many decode steps apart the scores take a
        step's attention mass: every step at 1 (#99), every R-th at R, the
        fallback #88 allows where every step costs more than noise (#100).
        Setting it applies to the next cache."""
        return self._kv_score_every

    @kv_score_every.setter
    def kv_score_every(self, every: int) -> None:
        if not isinstance(every, int) or every < 1:
            raise ValueError(f"kv_score_every is a count of decode steps, at least 1; "
                             f"got {every!r}")
        self._kv_score_every = every

    @property
    def kv_tier_map(self) -> list[list[str]] | None:
        """The tier map a cache is built from (#92), or None: row l names, by
        tier name, the tier each page of layer l is born at, page i at
        [l][i], and kv_tier beyond a row (ADR-0011, amended by #90). Setting
        it applies to the next cache, and is checked as the constructor
        checks it."""
        return self._kv_tier_map

    @kv_tier_map.setter
    def kv_tier_map(self, tier_map: list[list[str]] | None) -> None:
        if tier_map is not None:
            layers = self.config.num_hidden_layers
            if len(tier_map) != layers:
                raise ValueError(f"a tier map needs a row for every layer: {len(tier_map)} "
                                 f"rows for {layers} layers")
            names = {name for row in tier_map for name in row}
            if not names <= set(self.KV_TIERS):
                raise ValueError(f"kv_tier_map names tiers of {self.KV_TIERS}, got "
                                 f"{sorted(names - set(self.KV_TIERS))}")
            if self.kv_cache == "contiguous":
                raise ValueError("the contiguous cache holds FP16 only; a tier map is on pages")
            if self.kv_halves != "both":
                raise ValueError("a diagnostic kv_halves rounds every page at kv_tier, and "
                                 "takes no tier map")
            tier_map = [list(row) for row in tier_map]
        self._kv_tier_map = tier_map

    def _seals(self) -> bool:
        """Whether the next cache seals its pages (ADR-0011): KVPages's rule,
        for kv_tier and kv_tier_map. A diagnostic kv_halves is at a quantised
        kv_tier, so it seals too."""
        return _microinfer.device.KVPages.seals_for(
            getattr(_microinfer.Tier, self.kv_tier), tier_map_of(self.kv_tier_map),
            self.kv_adaptive)

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

    def generate(self, prompt, max_new_tokens: int = 64, *, stop_at_eos: bool = True,
                 report: Callable[[str, int, int | None], None] | None = None) -> np.ndarray:
        """Greedy continuation of one prompt: the new token ids only.

        `prompt` is text, which is tokenised, or token ids. The prompt is
        prefilled in chunks of `prefill_chunk` positions, then each new token
        is decoded against the cache. Stops after `max_new_tokens`, or at an end-of-sequence token,
        which is included, as HuggingFace's generate includes it.

        `report`, if given, is called as hold calls it (#108): after each
        prefill chunk, as (PREFILLING, positions so far, None), and after
        each decoded token, as (DECODING, the position after it, the token).
        The first new token is the prefill's, and is not reported.
        """
        report = report or (lambda *_: None)
        ids = self.encode(prompt) if isinstance(prompt, str) else self._check_ids(prompt)
        if max_new_tokens < 0:
            raise ValueError(f"max_new_tokens must not be negative, got {max_new_tokens}")
        if max_new_tokens == 0:
            return np.empty(0, dtype=np.int32)

        # The last new token is never fed back, so the cache needs one row less
        # than prompt plus output.
        self._check_window(len(ids) + max_new_tokens - 1)
        self.ending = None
        cache = self._new_cache(capacity=len(ids) + max_new_tokens - 1)
        out: list[int] = []
        def prefilled() -> bool:
            report(PREFILLING, cache.length, None)
            return True

        try:
            out.append(self._first_token(cache, ids, go_on=prefilled))
            step = model.Workspace(self.config, rows=1)
            with self._holding(cache, step):
                while len(out) < max_new_tokens and not (stop_at_eos
                                                         and out[-1] in self.eos_token_ids):
                    self._run(step, cache, np.array(out[-1:], np.int32))
                    out.append(self._model.greedy_last(step, 1))
                    self._drain_pressure(cache)
                    report(DECODING, cache.length, out[-1])
        except (_Exhausted, _microinfer.OutOfMemory) as exc:
            # Out of memory anywhere else, a workspace or a step's logits, is
            # answered as one the emergency plan could not answer.
            if not (isinstance(exc, _Exhausted) or self._adapts(cache)):
                raise
            return self._end(Ending(EXHAUSTED, cache.length, np.asarray(out, np.int32),
                                    str(exc)))
        return self._end(Ending(COMPLETE, cache.length, np.asarray(out, np.int32)))

    def _end(self, ending: Ending) -> np.ndarray:
        self.ending = ending
        return ending.tokens

    def _adapts(self, cache) -> bool:
        return self.kv_adaptive and isinstance(cache, model.PagedCache)

    def _run(self, ws: model.Workspace, cache, ids: np.ndarray, captured=None) -> None:
        """One step, a decoded token or a prefill chunk. In an adaptive
        engine, a step whose allocation fails is answered by an emergency
        plan, and run once more (#107): a RED plan for no headroom, applied
        at once. A step that fails reserves nothing (KVPages::reserve), so
        it can be run again. If the plan has nothing to downgrade, frees
        nothing, or the step fails again, the generation is exhausted. An
        engine that does not adapt raises OutOfMemory, as it did."""
        for attempt in range(2):
            try:
                self._model.run(ws, cache, ids, captured)
                return
            except _microinfer.OutOfMemory as exc:
                if not self._adapts(cache):
                    raise
                if attempt == 1:
                    raise _Exhausted(f"{exc}, again after an emergency plan") from exc
                self._emergency(cache, exc)

    def _emergency(self, cache: model.PagedCache, exc: Exception) -> None:
        """Plan and apply every downgrade for no headroom; raises _Exhausted
        if that freed no cache memory, with why."""
        started = time.perf_counter()
        if self._pending is not None:
            self._pending.ended, self._pending = "replaced", None
        self._recent_moves.clear()
        before = cache.nbytes
        entry = self._make_plan(cache, self._last_pressure, 0, started, emergency=str(exc))
        if entry is None:
            raise _Exhausted(f"{exc}; nothing to downgrade") from exc
        self._pending, self._cursor = entry, 0
        self._apply(cache, started, True, self._own_bytes())
        if self._pending is not None:  # only a RED plan is applied at once
            raise RuntimeError(f"an emergency plan was left part done: {entry.ended}")
        if cache.nbytes >= before:
            raise _Exhausted(f"{exc}; the emergency plan freed nothing ({entry.ended})") from exc

    def hold(self, prompt, *, context: int | None = None, stop: threading.Event,
             report: Callable[[str, int, int | None], None] | None = None) -> None:
        """Hold a generation in progress until `stop` is set: RQ1's engine
        under contention (#51).

        The prompt is prefilled, then tokens are decoded greedily to position
        `context`, the model's window by default. There the hold goes back to
        the end of the prompt and decodes the same span again, a new *pass*,
        from the last token it produced, over the pages it already holds:
        from the end of the first pass the cache holds the full context, and
        holds no more however long the hold lasts. Each pass reads only the
        prompt and itself, as a fresh generation from that token would.

        Going back rewrites positions in place, which only FP16 pages allow: a
        quantised page is sealed once its positions are written (ADR-0011).
        It also means that once the first pass is done a hold allocates
        nothing but each step's token ids: only while the cache still grows
        can contention make it run out of memory (OutOfMemory, #52).

        `report(state, position, token)` is called after each prefill chunk,
        as (PREFILLING, positions so far, None), and after each decoded token,
        as (DECODING, the position after it, the token). `stop` is checked
        after every chunk and every step, so a hold stops within one, and
        releases its cache as it returns.
        """
        if self._seals():
            raise ValueError("holding rewrites positions in place, which only FP16 pages allow")
        ids = self.encode(prompt) if isinstance(prompt, str) else self._check_ids(prompt)
        context = self.config.max_position_embeddings if context is None else context
        if not len(ids) < context:
            raise ValueError(f"the context, {context}, must exceed the prompt's "
                             f"{len(ids)} positions, to leave a span to decode")
        self._check_window(context)
        report = report or (lambda *_: None)

        cache = self._new_cache(capacity=context)

        def prefilled() -> bool:
            report(PREFILLING, cache.length, None)
            return not stop.is_set()

        token = self._first_token(cache, ids, go_on=prefilled)
        if token is None:
            return  # stopped while prefilling
        step = model.Workspace(self.config, rows=1)
        with self._holding(cache, step):
            while not stop.is_set():
                if cache.length == context:
                    cache.length = len(ids)  # back to the end of the prompt
                self._model.run(step, cache, np.array([token], np.int32))
                token = self._model.greedy_last(step, 1)
                self._drain_pressure(cache)
                report(DECODING, cache.length, token)

    def _first_token(self, cache, ids: np.ndarray, *,
                     go_on: Callable[[], bool] | None = None) -> int | None:
        """Prefill `ids` into `cache`: the greedy choice the prompt's last row
        makes, the first new token. `go_on`, if given, is asked after every
        chunk; if it says no, prefilling stops there and this returns None."""
        token = None
        for chunk, ws, _, last in self._prefill(cache, ids):
            self._drain_pressure(cache)
            if go_on is not None and not go_on():
                return None
            if last:
                token = self._model.greedy_last(ws, len(chunk))
        # Run out, not left: the prefill notes its peak footprint as it ends.
        return token

    # -- the pressure monitor (#62) -----------------------------------------

    def start_monitor(self, monitor: Monitor | None = None) -> Monitor:
        """Start `monitor`, a VRAM pressure monitor on NVML by default, and
        from now on drain its events between steps into pressure_events,
        which it starts empty, as it does monitor_error. A step is a decoded
        token, and a prefill chunk too: a 32K-token prefill takes minutes,
        and its chunks are where it can be interrupted. The engine records
        the events; with kv_adaptive it also plans and downgrades on YELLOW
        and RED (#105), and records each plan in `plans`, which it starts
        empty. Returns the monitor."""
        if self._monitor is not None:
            raise RuntimeError("a pressure monitor is already running; stop it first")
        self.pressure_events, self.monitor_error, self._positions_held = [], None, 0
        self.plans, self._pending, self.last_downgrade_seconds = [], None, None
        self._last_pressure, self._planned_pages = None, 0
        self._upgrades_looked_at, self.pressure_ended_seconds = None, None
        if self.kv_adaptive:
            self._plan_inputs  # read from the log now, not in a step a plan delays
        self._monitor = (monitor or Monitor()).start()
        self._thresholds = self._monitor.thresholds
        return self._monitor

    def stop_monitor(self) -> None:
        """Stop the monitor and record whatever it left undrained, beside the
        positions the last cache drained beside held."""
        if self._monitor is None:
            return
        self._monitor.stop()
        self._drain_pressure(self._cache)
        self._monitor = None

    def _drain_pressure(self, cache) -> None:
        """Record the monitor's events, with the positions `cache` holds. A
        monitor that failed is recorded in monitor_error and let go: the
        generation it watched goes on."""
        if self._monitor is None:
            return
        try:
            events = self._monitor.drain()
        except RuntimeError as exc:
            self.monitor_error, self._monitor = exc, None
            return
        now = time.monotonic_ns()
        if cache is not None:
            self._positions_held = cache.length
        drained = [PressureRecord(e, self._positions_held, now) for e in events]
        self.pressure_events += drained
        if self.kv_adaptive and isinstance(cache, model.PagedCache):
            self._react(cache, drained)

    # -- plans (#105) --------------------------------------------------------

    def _react(self, cache: model.PagedCache, drained: list[PressureRecord]) -> None:
        """Between two steps: make a plan for the latest YELLOW or RED among
        the events just drained, replacing any not yet done; cancel one at
        GREEN; and apply what this step may of the plan in hand. At RED a
        plan is applied at once. At YELLOW a step takes no more than the
        budget, planning included: downgrades are applied while the time
        spent, and twice the longest of the last 64 downgrades, fit it,
        every one timed, so that a step is held to the budget, not to an
        estimate of it; never less than MOVE_MARGIN_SECONDS, a downgrade
        that pins another allocation of shadows taking half a millisecond
        (#97), a garbage collection more. A step that makes no plan applies
        one downgrade at least."""
        own_before = self._own_bytes()
        started = time.perf_counter()
        planned = False
        # The latest level is the one to answer: one plan a step at most.
        latest = drained[-1] if drained else None
        if latest is not None:
            self._last_pressure = latest
        for record in drained:
            if record.event.level == GREEN and record.event.previous in (YELLOW, RED):
                self.pressure_ended_seconds = record.event.t_mono_ns / 1e9
        if (self._pending is not None and self._pending.upgrades
                and any(r.event.level in (YELLOW, RED) for r in drained)):
            # Pressure, though it may have passed within the step: the
            # upgrades not yet made wait for the cooldown again.
            self._pending.ended, self._pending = "cancelled by pressure", None
        if (latest is not None and latest.event.level == GREEN and self._pending is not None
                and not self._pending.upgrades):
            self._pending.ended, self._pending = "cancelled at GREEN", None
        elif latest is not None and latest.event.level in (YELLOW, RED):
            if self._pending is not None:
                self._pending.ended = "replaced"
            self._recent_moves.clear()  # a new plan's moves are timed afresh
            self._pending = self._make_plan(cache, latest, latest.event.headroom_bytes, started)
            self._cursor, planned = 0, True
        elif self._persisting(cache):
            # The level held, no plan is in hand, and the cache has sealed
            # pages since the last plan: plan again, for the headroom now.
            self._recent_moves.clear()
            self._pending = self._make_plan(cache, self._last_pressure,
                                            self._monitor.headroom_bytes, started,
                                            persisting=True)
            self._cursor, planned = 0, self._pending is not None
        elif self._upgrades_due():
            # GREEN, no plan in hand: restore pages, as the policy allows.
            self._recent_moves.clear()
            self._pending = self._make_upgrade_plan(cache, started)
            self._cursor, planned = 0, self._pending is not None
        if self._pending is not None:
            self._apply(cache, started, planned, own_before)

    def _upgrades_due(self) -> bool:
        if (self._pending is not None or self._monitor is None
                or self._monitor.level != GREEN or self._monitor.headroom_bytes is None
                or self.last_downgrade_seconds is None):
            return False
        now = time.monotonic()
        if (self._upgrades_looked_at is not None
                and now - self._upgrades_looked_at < UPGRADE_RETRY_SECONDS):
            return False
        self._upgrades_looked_at = now
        return True

    def _make_upgrade_plan(self, cache: model.PagedCache, started: float) -> PlanRecord | None:
        """Upgrades for the headroom the monitor reads now (#106): from each
        page's shadow, in the policy's order (#103), recorded with the
        pressure event that brought GREEN; None, unrecorded, if the policy
        holds them or there are none to make. The cooldown runs from the
        last downgrade or the end of the last pressure, the later: under a
        train of pulses whose later ones find nothing left to downgrade,
        upgrades between them would be undone by the next."""
        pages = cache.pages
        tiers = pages.page_tiers()
        layers, count = tiers.shape
        semantic = cache.scores.download() if self.kv_score_source == "semantic" else None
        scores = score_array(self.kv_score_source, layers, count, semantic=semantic,
                             seed=self.kv_score_seed)
        page_bytes, errors, seconds = self._plan_inputs
        cooldown_from = max(t for t in (self.last_downgrade_seconds, self.pressure_ended_seconds)
                            if t is not None)
        plan = controller.upgrade_plan_arrays(
            self._monitor.headroom_bytes, self._thresholds, scores, tiers, pages.shadowed(),
            page_bytes, errors, move_seconds=seconds,
            last_downgrade_seconds=cooldown_from,
            cooldown_seconds=self.kv_upgrade_cooldown_seconds,
            budget_seconds=self.kv_plan_budget_seconds)
        if len(plan) == 0:
            return None
        entry = PlanRecord(self._last_pressure, plan, time.perf_counter() - started, [],
                           headroom_bytes=self._monitor.headroom_bytes)
        self.plans.append(entry)
        self._recent_moves.append(float(plan.seconds.max()))  # until one is timed
        return entry

    def _persisting(self, cache: model.PagedCache) -> bool:
        return (self._pending is None and self._under_pressure(cache)
                and self._last_pressure is not None
                and self._monitor.headroom_bytes is not None
                and cache.pages.pages_per_layer > self._planned_pages)

    def _make_plan(self, cache: model.PagedCache, record: PressureRecord | None, headroom: int,
                   started: float, persisting: bool = False,
                   emergency: str | None = None) -> PlanRecord | None:
        """A plan for this headroom, recorded with the event it answers, or
        the error an emergency plan answers; None if it has nothing to do,
        recorded as applied, unless it was made because pressure persisted,
        when nothing is recorded."""
        self._planned_pages = cache.pages.pages_per_layer
        pages = cache.pages
        tiers = pages.page_tiers()
        layers, count = tiers.shape
        semantic = cache.scores.download() if self.kv_score_source == "semantic" else None
        scores = score_array(self.kv_score_source, layers, count, semantic=semantic,
                             seed=self.kv_score_seed)
        page_bytes, errors, seconds = self._plan_inputs
        # The monitor's thresholds; ADR-0013's for an emergency plan made
        # with no monitor running.
        plan = controller.plan_arrays(
            headroom, self._thresholds or DEFAULT_THRESHOLDS, scores, tiers, page_bytes, errors,
            move_seconds=seconds, positions=cache.length, page_tokens=pages.page_tokens,
            budget_seconds=self.kv_plan_budget_seconds)
        entry = PlanRecord(record, plan, time.perf_counter() - started, [],
                           headroom_bytes=headroom, persisting=persisting, emergency=emergency)
        if len(plan) == 0 and persisting:
            return None
        self.plans.append(entry)
        if len(plan) == 0:
            entry.ended = "nothing to downgrade" if emergency else "applied"
            return None
        self._recent_moves.append(float(plan.seconds.max()))  # until one is timed
        return entry

    @cached_property
    def _plan_inputs(self):
        """Each tier's page bytes, its logged roundtrip error, and each
        downgrade's logged latency: this model's, or, not measured, any
        model's as an estimate."""
        page_bytes = dict(zip(controller.TIERS, model.tier_page_bytes(self.config)))
        errors = controller.logged_tier_errors(self.config.name)
        try:
            seconds = controller.logged_move_seconds(self.config.name)
        except ValueError:
            seconds = controller.logged_move_seconds(None)
        return page_bytes, errors, seconds

    def _apply(self, cache: model.PagedCache, started: float, planned: bool,
               own_before: int | None) -> None:
        entry = self._pending
        plan, pages = entry.plan, cache.pages
        tier_of = [getattr(_microinfer.Tier, t) for t in controller.TIERS]
        cache_before, start, i = cache.nbytes, self._cursor, self._cursor
        budget = self.kv_plan_budget_seconds
        move = pages.upgrade if entry.upgrades else pages.downgrade
        try:
            while i < len(plan):
                if plan.level != RED:
                    spent = time.perf_counter() - started
                    next_move = max(2 * max(self._recent_moves), MOVE_MARGIN_SECONDS)
                    if spent + next_move > budget and (i > start or planned):
                        break
                layer, page = int(plan.layers[i]), int(plan.pages[i])
                if pages.page_tier(layer, page) != tier_of[plan.current[i]]:
                    entry.skipped += 1
                else:
                    moving = time.perf_counter()
                    move(layer, page, tier_of[plan.target[i]])
                    self._recent_moves.append(time.perf_counter() - moving)
                    if not entry.upgrades:
                        self.last_downgrade_seconds = time.monotonic()
                i += 1
        except _microinfer.OutOfMemory as exc:  # the emergency is #107's to answer
            entry.ended, self._pending = f"stopped: {exc}", None
        seconds = time.perf_counter() - started
        if i > start:
            entry.batches.append(AppliedBatch(start, i, seconds, cache_before, cache.nbytes,
                                              own_before, self._own_bytes(), time.monotonic()))
        self._cursor = i
        if self._pending is not None and i == len(plan):
            entry.ended, self._pending = "applied", None

    def _own_bytes(self) -> int | None:
        if not self.measure_plans:
            return None
        try:
            return nvml.settled_own_used_bytes()
        except nvml.NvmlUnavailable:
            return None

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
        where the one before it stopped.

        While an adaptive engine's monitor reads YELLOW or RED, a chunk is
        sized by pressure_chunk from the time a position of the chunk before
        took (#135), so that the events the monitor raises
        meanwhile are drained between short steps. At GREEN a chunk is
        prefill_chunk, as without a monitor."""
        n = len(ids)
        size = n if self.prefill_chunk is None else min(self.prefill_chunk, n)
        ws = model.Workspace(self.config, rows=size)
        seconds_per_position = None
        with self._holding(cache, ws):
            start = 0
            while start < n:
                step = size
                if seconds_per_position is not None and self._under_pressure(cache):
                    step = pressure_chunk(seconds_per_position,
                                          seconds=self.pressure_chunk_seconds,
                                          tile=_microinfer.attention_tiles["query"], cap=size)
                chunk = ids[start:start + step]
                captured = [] if hidden_states else None
                began = time.perf_counter()
                self._run(ws, cache, chunk, captured)
                seconds_per_position = (time.perf_counter() - began) / len(chunk)
                start += len(chunk)
                yield chunk, ws, captured, start >= n

    def _under_pressure(self, cache) -> bool:
        """Whether this engine adapts and its monitor's level is YELLOW or
        RED: a monitor that stopped, or none, is no pressure to act on."""
        return (self._adapts(cache) and self._monitor is not None
                and self._monitor.level in (YELLOW, RED))

    def _new_cache(self, capacity: int):
        """A cache for one sequence. The contiguous one is sized for `capacity`
        up front; the paged one ignores it and takes pages as positions arrive,
        so a generation that stops early never held room for the rest."""
        if self.kv_cache == "contiguous":
            return model.ContiguousCache(self.config, capacity)
        Tier = _microinfer.Tier
        tier = getattr(Tier, self.kv_tier)
        if self.kv_halves == "both":
            # An adaptive cache seals, so that any page can be downgraded, and
            # scores its pages when a plan reads the scorer's scores.
            scoring = self.kv_scoring or (self.kv_adaptive and self.kv_score_source == "semantic")
            cache = model.PagedCache(self.config, tier, tier_map=tier_map_of(self.kv_tier_map),
                                     always_seal=self.kv_adaptive, scoring=scoring,
                                     score_every=self.kv_score_every)
            if self.kv_adaptive:
                # A spare granule at each quantised tier (#107): a downgrade
                # maps its new page before it frees the old, so with none, an
                # emergency plan, made when the driver has nothing to give,
                # could not start. Once it has freed enough FP16 pages to
                # empty a granule, the driver has room again.
                for t in controller.TIERS[1:]:
                    cache.allocator.keep_spare_granules(getattr(Tier, t), EMERGENCY_SPARES)
            return cache
        if tier == Tier.FP16:
            raise ValueError("kv_halves splits a quantised tier; FP16 has nothing to split")
        return model.PagedCache(self.config, tier,
                                getattr(_microinfer.device.Halves, self.kv_halves.title()),
                                scoring=self.kv_scoring, score_every=self.kv_score_every)

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
