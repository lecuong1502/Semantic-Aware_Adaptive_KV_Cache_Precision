#!/usr/bin/env python3
"""How long a plan's moves take, page by page and in batches (#205).

    .venv/bin/python tools/plan_apply_speed.py --issue 205
    .venv/bin/python tools/plan_apply_speed.py --issue 205 --paths batch

The survival experiment's first plan (#108, `1fdbeebd...`) downgraded 44,016
pages of Qwen2.5-1.5B's cache at 16K positions in 1.7 s, and restoring a 32K
cache after pressure took about 540 s, paced by steps (#155). This measures
the two through the cache the engine holds (Seam B), at the model's shapes,
with no model loaded:

- **the plan:** a cache of every layer's pages at --plan-positions, sealed
  at FP16, and a plan like that one: every page outside the recency floor
  down from FP16 a tier at a time, to INT8, INT4 and INT2, in that order, a
  layer's pages in turn, cut at --plan-moves;
- **the restore:** a cache of --restore-positions, every page downgraded to
  INT4, and every page back up to FP16, from its shadow.

Each is applied page by page, as Engine._apply did before #205, and in
batches of --batch moves, a batch naming each page once at the tier it ends
at, as Engine._apply now makes them. Every path starts from the same cache,
built afresh. Also timed: batches of a few sizes, one at a time, from which
a batch's fixed part and its part a move come, as the engine's pacing uses
them. Appends one "plan-apply-speed" entry. Refuses a dirty tree.
"""

from __future__ import annotations

import argparse
import gc
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
DOWN = (Tier.INT8, Tier.INT4, Tier.INT2)
SEED = 205
#: Positions stored at a time while a cache is filled.
FILL = 4096


def sealed_cache(cfg: ModelConfig, positions: int, rng):
    """Every layer's pages for `positions`, sealed at FP16 in a cache that
    can move them; its allocator, and the pages a layer holds."""
    layers, kv_heads, hd = cfg.num_hidden_layers, cfg.num_key_value_heads, cfg.head_dim
    page_tokens = device.page_tokens
    pages = positions // page_tokens
    capacity = [layers * (pages + len(device.open_pages) + 2)] * 4
    allocator = _microinfer.PagedKVCache(tier_page_bytes(cfg, page_tokens), capacity)
    cache = device.KVPages(allocator, layers, page_tokens, kv_heads, hd, Tier.FP16,
                           device.Halves.Both, [], True)
    width = kv_heads * hd
    for start in range(0, positions, FILL):
        n = min(FILL, positions - start)
        cache.reserve(start + n)
        for layer in range(layers):
            k = _microinfer.upload_fp16(rng.standard_normal(n * width).astype(np.float32))
            v = _microinfer.upload_fp16(rng.standard_normal(n * width).astype(np.float32))
            cache.store(layer, k, v, start, n)
    return allocator, cache, pages


def plan_like_the_first(layers: int, pages: int, floor: int, moves: int):
    """(layer, page, tier) moves a tier at a time, as the first plan made
    them: every page outside the recency floor to INT8, then to INT4, then
    to INT2, layer by layer, cut at `moves`."""
    plan = [(layer, page, tier) for tier in DOWN for layer in range(layers)
            for page in range(pages - floor)]
    return plan[:moves]


def batches(plan, size: int):
    """The plan in batches of `size` plan moves, each naming a page once at
    the tier it ends at, in the plan's order."""
    for first in range(0, len(plan), size):
        ends: dict[tuple[int, int], Tier] = {}
        for layer, page, tier in plan[first:first + size]:
            ends[(layer, page)] = tier
        yield [(layer, page, tier) for (layer, page), tier in ends.items()]


def per_page(cache, plan) -> float:
    start = time.perf_counter()
    for layer, page, tier in plan:
        current = cache.page_tier(layer, page)
        (cache.downgrade if int(tier) > int(current) else cache.upgrade)(layer, page, tier)
    return time.perf_counter() - start


def batched(cache, plan, size: int) -> float:
    start = time.perf_counter()
    for batch in batches(plan, size):
        cache.move_pages(batch)
    return time.perf_counter() - start


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--issue", type=int, required=True,
                        help="the ticket this measurement is made for")
    parser.add_argument("--model", default="qwen2.5-1.5b-instruct")
    parser.add_argument("--plan-positions", type=int, default=16896,
                        help="528 pages: with the floor, 44,016 moves")
    parser.add_argument("--plan-moves", type=int, default=44016)
    parser.add_argument("--recency-floor", type=int, default=4,
                        help="pages a layer keeps at FP16, its last")
    parser.add_argument("--restore-positions", type=int, default=32768)
    parser.add_argument("--batch", type=int, default=1 << 30,
                        help="plan moves a batch; by default the whole plan")
    parser.add_argument("--paths", nargs="+", default=["per-page", "batch"],
                        choices=["per-page", "batch"])
    parser.add_argument("--sizes", nargs="+", type=int, default=[1, 16, 256, 4096])
    parser.add_argument("--log", type=Path, default=benchlog.DEFAULT_LOG)
    args = parser.parse_args(argv)

    if benchlog.environment(args.log)["git_dirty"]:
        parser.error("tracked files have uncommitted changes; commit first, so that "
                     "each entry names the code that produced it")
    cfg = ModelConfig.from_card(args.model)
    rng = np.random.default_rng(SEED)
    results: dict = {"plan": {}, "restore": {}}

    for path in args.paths:
        allocator, cache, pages = sealed_cache(cfg, args.plan_positions, rng)
        plan = plan_like_the_first(cfg.num_hidden_layers, pages, args.recency_floor,
                                   args.plan_moves)
        seconds = (per_page(cache, plan) if path == "per-page"
                   else batched(cache, plan, args.batch))
        results["plan"][path] = {"seconds": seconds, "moves": len(plan)}
        print(f"plan, {path}: {len(plan)} moves in {seconds:.3f} s")
        del cache, allocator
        gc.collect()

    for path in args.paths:
        allocator, cache, pages = sealed_cache(cfg, args.restore_positions, rng)
        everything = [(layer, page) for layer in range(cfg.num_hidden_layers)
                      for page in range(pages)]
        cache.move_pages([(layer, page, Tier.INT4) for layer, page in everything])
        restore = [(layer, page, Tier.FP16) for layer, page in everything]
        seconds = (per_page(cache, restore) if path == "per-page"
                   else batched(cache, restore, args.batch))
        results["restore"][path] = {"seconds": seconds, "moves": len(restore)}
        print(f"restore, {path}: {len(restore)} moves in {seconds:.3f} s")
        del cache, allocator
        gc.collect()

    if "batch" in args.paths:
        allocator, cache, pages = sealed_cache(cfg, args.plan_positions, rng)
        free = [(layer, page) for layer in range(cfg.num_hidden_layers)
                for page in range(pages)]
        timed = {}
        for size in args.sizes:
            runs = []
            for _ in range(3):
                take, free = free[:size], free[size:]
                cache.move_pages([(layer, page, Tier.INT4) for layer, page in take])
                runs.append(cache.last_batch["seconds"])
            timed[str(size)] = statistics.median(runs)
            print(f"a batch of {size}: {timed[str(size)] * 1e3:.2f} ms")
        sizes = np.array(args.sizes, float)
        per_move, fixed = np.polyfit(sizes, [timed[str(int(s))] for s in sizes], 1)
        results["batch_seconds_by_size"] = timed
        results["batch_fixed_seconds"] = float(fixed)
        results["batch_seconds_a_move"] = float(per_move)
        print(f"a batch: {fixed * 1e3:.2f} ms and {per_move * 1e6:.2f} us a move")
        del cache, allocator

    benchlog.append(
        "plan-apply-speed", model=args.model,
        context_length={"min": args.plan_positions, "max": args.restore_positions},
        precision_tiers=None,
        config={"issue": args.issue, "plan": "every page outside the recency floor down "
                "from FP16 a tier at a time, layer by layer, as the first plan of "
                "survival entry 1fdbeebd", "plan_positions": args.plan_positions,
                "plan_moves": args.plan_moves, "recency_floor_pages": args.recency_floor,
                "restore": "every page from INT4 back to FP16",
                "restore_positions": args.restore_positions, "batch_plan_moves": args.batch,
                "page_tokens": device.page_tokens, "layout": "Seam B, no model loaded"},
        results=results, log=args.log)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
