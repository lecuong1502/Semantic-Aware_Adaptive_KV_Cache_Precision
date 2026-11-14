#!/usr/bin/env python3
"""What one launch of attention over a long cache costs, at every tier (#204).

    .venv/bin/python tools/attention_speed.py --issue 204
    .venv/bin/python tools/attention_speed.py --issue 204 --lengths 32768 --tiers INT4

A decode step at 32K positions is dominated by attention: every layer reads
the whole cache. A full decode measurement there needs a 32K-token prefill
first, which takes minutes, so this measures the kernel alone, through the
cache the engine reads (Seam B): one layer's pages, `--lengths` positions of
random keys and values at the model's shapes, keys stored without their bias
and completed by it, as Qwen2.5's are (ADR-0009).

Two launches are timed at each length and tier:
- **decode:** one query over every position, as a decode step attends, with
  the per-page attention mass and without it (#98);
- **prefill tile:** the queries of one tile of a prefill chunk, the smallest
  step a prefill under pressure takes (ADR-0016, amendment of #135), over the
  same positions. Its cost is most of a step's fixed part at long contexts.

`mixed` is a cache born from a map that cycles the four tiers page by page,
each layer's row offset by one, so that every tier sits next to every other.

Each figure is the median over `--repeat` batches of `--batch` launches,
after a warm-up, in microseconds a launch, and as milliseconds a token over
every layer of the model, which is what decode pays for attention. A launch
includes resolving the page table to the device, as the engine's does.
Appends one "attention-speed" entry per tier. Refuses a dirty tree.
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from microinfer import _microinfer, benchlog  # noqa: E402
from microinfer.config import ModelConfig  # noqa: E402
from microinfer.model import tier_page_bytes  # noqa: E402

device = _microinfer.device
Tier = _microinfer.Tier
TIERS = ("FP16", "INT8", "INT4", "INT2", "mixed")
#: Positions stored at a time while the cache is filled.
FILL = 4096
SEED = 204


def tier_map(tier: str, layers: int, pages: int):
    """The map a cache at `tier` is born from: none for one tier, and for
    `mixed`, the four tiers cycled page by page, each layer's row offset."""
    if tier != "mixed":
        return None
    order = [Tier.FP16, Tier.INT8, Tier.INT4, Tier.INT2]
    return [[order[(i + layer) % 4] for i in range(pages)] for layer in range(layers)]


def fill(cfg: ModelConfig, tier: str, length: int, last: int, rng):
    """A one-layer cache of `length` positions at `tier`, stored as decode
    and prefill store them, the last `last` positions in a run of their own;
    its allocator, and those last rows, which the queries brought."""
    kv_heads, hd, page_tokens = cfg.num_key_value_heads, cfg.head_dim, device.page_tokens
    pages = -(-length // page_tokens)
    mapping = tier_map(tier, 1, pages)
    born = Tier.FP16 if tier in ("FP16", "mixed") else Tier.__members__[tier]
    capacity = [0] * 4
    for t in (Tier.__members__.values() if mapping is not None else (born,)):
        capacity[int(t)] = pages + 2
    capacity[int(Tier.FP16)] += len(device.open_pages)
    allocator = _microinfer.PagedKVCache(tier_page_bytes(cfg, page_tokens), capacity)
    cache = device.KVPages(allocator, 1, page_tokens, kv_heads, hd, born, device.Halves.Both,
                           mapping or [])
    width = kv_heads * hd

    def rows(n):
        return _microinfer.upload_fp16(rng.standard_normal(n * width).astype(np.float32))

    start = 0
    for n in [FILL] * ((length - last) // FILL) + [(length - last) % FILL, last]:
        if n == 0:
            continue
        cache.reserve(start + n)
        k, v = rows(n), rows(n)
        cache.store(0, k, v, start, n)
        start += n
    return allocator, cache, k, v


def timed(launch, sync, repeat: int, batch: int) -> float:
    """Median microseconds a launch, over `repeat` batches of `batch`."""
    for _ in range(3):
        launch()
    sync()
    per = []
    for _ in range(repeat):
        start = time.perf_counter()
        for _ in range(batch):
            launch()
        sync()
        per.append((time.perf_counter() - start) / batch * 1e6)
    return statistics.median(per)


def measure(cfg: ModelConfig, tier: str, length: int, tile: int, repeat: int, batch: int) -> dict:
    rng = np.random.default_rng(SEED)
    heads, kv_heads, hd = cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim
    allocator, cache, k_last, v_last = fill(cfg, tier, length, tile, rng)
    rope = device.RopeTable(hd, cfg.rope_theta)
    rope.cover(length)
    bias = _microinfer.upload_fp16(
        (rng.standard_normal(kv_heads * hd) * 30).astype(np.float32))
    width = kv_heads * hd
    # The last position's own rows, for a query of one: the last of the run.
    host_k, host_v = k_last.to_numpy(), v_last.to_numpy()
    one_k = _microinfer.upload_fp16(host_k[-width:].astype(np.float32))
    one_v = _microinfer.upload_fp16(host_v[-width:].astype(np.float32))
    q1 = _microinfer.upload_fp16(rng.standard_normal(heads * hd).astype(np.float32))
    qt = _microinfer.upload_fp16(rng.standard_normal(tile * heads * hd).astype(np.float32))
    out1, outt = device.empty(heads * hd), device.empty(tile * heads * hd)
    pages = -(-length // device.page_tokens)
    mass = device.empty_f32(heads * pages)

    def decode(with_mass):
        return lambda: cache.attention(0, q1, bias, rope, out1, 1, length, heads, kv_heads,
                                       hd, one_k, one_v, mass if with_mass else None)

    def prefill_tile():
        cache.attention(0, qt, bias, rope, outt, tile, length, heads, kv_heads, hd,
                        k_last, v_last)

    def sync():
        out1.to_numpy()

    result = {}
    for name, launch in (("decode", decode(False)), ("decode_with_mass", decode(True)),
                         ("prefill_tile", prefill_tile)):
        us = timed(launch, sync, repeat, batch)
        result[name] = {"us_per_launch": us,
                        "ms_per_token_all_layers": us * cfg.num_hidden_layers / 1e3}
    del allocator
    return result


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--issue", type=int, required=True,
                        help="the ticket this measurement is made for")
    parser.add_argument("--model", default="qwen2.5-1.5b-instruct")
    parser.add_argument("--lengths", nargs="+", type=int, default=[16384, 32768])
    parser.add_argument("--tiers", nargs="+", default=list(TIERS), choices=TIERS)
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--batch", type=int, default=10)
    parser.add_argument("--log", type=Path, default=benchlog.DEFAULT_LOG)
    args = parser.parse_args(argv)

    if benchlog.environment(args.log)["git_dirty"]:
        parser.error("tracked files have uncommitted changes; commit first, so that "
                     "each entry names the code that produced it")

    cfg = ModelConfig.from_card(args.model)
    tile = _microinfer.attention_tiles["query"]
    for tier in args.tiers:
        results = {}
        for length in args.lengths:
            results[str(length)] = measure(cfg, tier, length, tile, args.repeat, args.batch)
            r = results[str(length)]
            print(f"{tier:>5} {length:>6}: decode {r['decode']['us_per_launch']:9.1f} us "
                  f"({r['decode']['ms_per_token_all_layers']:7.1f} ms/token), with mass "
                  f"{r['decode_with_mass']['us_per_launch']:9.1f} us, prefill tile "
                  f"{r['prefill_tile']['us_per_launch']:9.1f} us "
                  f"({r['prefill_tile']['ms_per_token_all_layers']:7.1f} ms/step)")
        tiers = ({t: 0.25 for t in ("FP16", "INT8", "INT4", "INT2")} if tier == "mixed"
                 else {tier: 1.0})
        benchlog.append(
            "attention-speed", model=args.model,
            context_length={"min": min(args.lengths), "max": max(args.lengths)},
            precision_tiers=tiers,
            config={"issue": args.issue, "layout": "causal paged (ADR-0011)",
                    "tier_map": "four tiers cycled by page" if tier == "mixed" else None,
                    "lengths": args.lengths, "page_tokens": device.page_tokens,
                    "prefill_tile_queries": tile,
                    "key_bias": True, "repeat": args.repeat, "batch": args.batch,
                    "statistic": "median", "seed": SEED, "layers": cfg.num_hidden_layers},
            results=results, log=args.log)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
