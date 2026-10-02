#!/usr/bin/env python3
"""Requantisation latency, by tier pair and per MiB, measured and logged (#97).

    .venv/bin/python tools/requantisation_latency.py --issue 97 \\
        [--model qwen2.5-1.5b-instruct] [--positions 4096] [--samples 256]

A cache at FP16 that seals, with `positions` positions of random rows stored
in every layer and the model never run, has pages downgraded and upgraded
between every pair of tiers, each move timed by KVPages.last_move: the wall
time of the whole move, once the device has finished what it was given
before. Every step of a move is synchronous, so that is its whole cost. Of
it, the shadow's copy is reported apart (#94): to the host on a first
downgrade, back to the device on a move from the shadow.

Three groups of `samples` pages, one for each quantised tier T, go FP16 to T
(a first downgrade, which copies the shadow), T back to FP16, and FP16 to T
again (the shadow held). The INT8 group then goes round the quantised tiers,
INT8 to INT4 to INT2, up to INT8, down to INT2, up to INT4 and up to INT8,
so that every pair of tiers is timed `samples` times. Three pages of their
own make every one of these moves first, untimed, so that no figure carries
a kernel's first launch.

Each pair, and each first downgrade apart, is one entry in the benchmark
log: median, P90 and max of the seconds a move took, of the seconds per MiB
reclaimed or restored, and of the shadow's copy and its share of the move.
A MiB here is of the page's own change in bytes, not of the driver's
granules, which a move maps or unmaps only now and then. Those moves, and
the first downgrades that pin another allocation of shadows, are the
slowest, and each entry counts them, so that a max can be read against
them. The controller's per-step time budget is set from these, by its own
ticket (#88). Refuses a dirty tree.
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from microinfer import ModelConfig, _microinfer, benchlog, model  # noqa: E402
from microinfer.contention import spread  # noqa: E402
from microinfer.footprint import MIB  # noqa: E402

Tier = _microinfer.Tier
QUANTISED = (Tier.INT8, Tier.INT4, Tier.INT2)
# The INT8 group's round of the quantised tiers: every pair between them.
ROUND = (("downgrade", Tier.INT4), ("downgrade", Tier.INT2), ("upgrade", Tier.INT8),
         ("downgrade", Tier.INT2), ("upgrade", Tier.INT4), ("upgrade", Tier.INT8))


def measure(cfg: ModelConfig, positions: int, samples: int,
            rng: np.random.Generator) -> list[dict]:
    """Every timed move, as last_move reports it, and whether it was a
    first downgrade, mapped or unmapped a granule, or pinned another
    allocation of shadows."""
    layers = cfg.num_hidden_layers
    pages = positions // _microinfer.device.page_tokens
    kv_width = cfg.num_key_value_heads * cfg.head_dim
    cache = model.PagedCache(cfg, Tier.FP16, always_seal=True)
    cache.reserve(positions)
    rows = _microinfer.upload_fp16(
        rng.standard_normal(positions * kv_width).astype(np.float32))
    for layer in range(layers):
        cache.store(layer, rows, rows, 0, positions)

    keys = [(layer, i) for i in range(pages) for layer in range(layers)]
    needed = 3 * samples + 3
    if len(keys) < needed:
        raise ValueError(f"{len(keys)} pages hold too few for {samples} samples a group; "
                         f"{needed} are needed")
    warm, groups = keys[:3], [keys[3 + g * samples:3 + (g + 1) * samples] for g in range(3)]
    records = []

    def mapped():
        return [cache.allocator.mapped_bytes(t) for t in Tier.__members__.values()]

    def move(key, way, target, timed=True):
        first = way == "downgrade" and not cache.pages.has_shadow(*key)
        granules, pinned = mapped(), cache.pages.shadow_bytes
        getattr(cache.pages, way)(*key, target)
        if timed:
            records.append({**cache.pages.last_move, "first_downgrade": first,
                            "mapped_granules": mapped() != granules,
                            "pinned_shadows": cache.pages.shadow_bytes != pinned})

    def schedule(groups):
        for group, tier in zip(groups, QUANTISED):
            for key in group:
                yield key, "downgrade", tier
                yield key, "upgrade", Tier.FP16
                yield key, "downgrade", tier
        for key in groups[0]:
            for way, target in ROUND:
                yield key, way, target

    for step in schedule([[key] for key in warm]):
        move(*step, timed=False)
    for step in schedule(groups):
        move(*step)
    return records


def groups(records: list[dict]) -> dict[tuple[str, str, bool], list[dict]]:
    """The moves of each pair of tiers, by name, first downgrades apart:
    one entry of the log each."""
    grouped = defaultdict(list)
    for r in records:
        grouped[(r["from_tier"].name, r["to_tier"].name, r["first_downgrade"])].append(r)
    return dict(grouped)


def summarise(records: list[dict]) -> dict:
    """One group of moves of one pair: their seconds, per MiB of the page's
    change in bytes, and the shadow's copy and its share of each move; and
    how many mapped or unmapped a granule, or pinned shadows."""
    moved = abs(records[0]["from_bytes"] - records[0]["to_bytes"])
    seconds = [r["seconds"] for r in records]
    return {"moves": len(records), "bytes_per_move": moved,
            "seconds": spread(seconds),
            "seconds_per_mib": spread([s / (moved / MIB) for s in seconds]),
            "shadow_seconds": spread([r["shadow_seconds"] for r in records]),
            "shadow_share": spread([r["shadow_seconds"] / r["seconds"] for r in records]),
            "moves_mapping_granules": sum(r["mapped_granules"] for r in records),
            "moves_pinning_shadows": sum(r["pinned_shadows"] for r in records)}


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--issue", type=int, required=True,
                        help="the ticket this measurement is made for")
    parser.add_argument("--model", default="qwen2.5-1.5b-instruct")
    parser.add_argument("--positions", type=int, default=4096)
    parser.add_argument("--samples", type=int, default=256)
    parser.add_argument("--log", type=Path, default=benchlog.DEFAULT_LOG)
    args = parser.parse_args(argv)

    if benchlog.environment(args.log)["git_dirty"]:
        parser.error("tracked files have uncommitted changes; commit first, so that "
                     "each entry names the code that produced it")

    cfg = ModelConfig.from_card(args.model)
    records = measure(cfg, args.positions, args.samples, np.random.default_rng(args.issue))
    for (source, target, first), group in sorted(groups(records).items()):
        results = summarise(group)
        print(f"{source}->{target}{' (first)' if first else ''}: "
              f"median {results['seconds']['median'] * 1e3:.3f} ms, "
              f"P90 {results['seconds']['p90'] * 1e3:.3f} ms, "
              f"max {results['seconds']['max'] * 1e3:.3f} ms, "
              f"{results['seconds_per_mib']['median'] * 1e3:.2f} ms/MiB, "
              f"shadow {results['shadow_share']['median']:.0%} of it")
        benchlog.append(
            "requantisation-latency", model=args.model, context_length=args.positions,
            precision_tiers=None,
            config={"from_tier": source, "to_tier": target, "first_downgrade": first,
                    "samples": args.samples, "page_tokens": _microinfer.device.page_tokens,
                    "timed": "KVPages.last_move: wall time of the whole move, once the "
                             "device has finished earlier work; shadow_seconds the "
                             "shadow's copy within it, either way",
                    "per_mib": "of the page's change in bytes, not the driver's granules",
                    "issue": args.issue},
            results=results, log=args.log)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
