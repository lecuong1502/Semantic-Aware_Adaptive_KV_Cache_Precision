#!/usr/bin/env python3
"""The KV cache's footprint at each static tier, measured and computed (#18).

    .venv/bin/python tools/tier_footprint.py --issue 18 [--model qwen2.5-1.5b-instruct]

For the model's whole context window at each tier, a paged cache is made and
its pages reserved, without running the model. What this process holds by the
driver's own account (nvml.own_used_bytes), which no other process moves, is
set against paged_cache_bytes, which computes it from the layout. The RoPE
table is covered first, so the pages alone are measured.

Each entry also records what a compression ratio must be quoted from
(ADR-0005): the bytes the pages themselves need, scale metadata and open
pages counted but not the granules' rounding, the effective bits per cached
element they come to, and the ratio to FP16's. Refuses a dirty tree.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from microinfer import Engine, ModelConfig, _microinfer, benchlog, model, nvml  # noqa: E402
from microinfer import paged_cache_bytes  # noqa: E402

MIB = 2**20


def payload_bytes(cfg: ModelConfig, tokens: int, tier: str) -> int:
    """The pages' own bytes for `tokens` positions: what the layout needs,
    before the driver rounds each range up to granules."""
    page_tokens = _microinfer.device.page_tokens
    layers = cfg.num_hidden_layers
    page_bytes = model.tier_page_bytes(cfg)
    fp16 = page_bytes[int(_microinfer.Tier.FP16)]
    if tier == "FP16":
        return layers * -(-tokens // page_tokens) * fp16
    page = page_bytes[int(getattr(_microinfer.Tier, tier))]
    return (layers * (tokens // page_tokens) * page
            + layers * len(_microinfer.device.open_pages) * fp16)


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--issue", type=int, required=True,
                        help="the ticket this measurement is made for")
    parser.add_argument("--model", default="qwen2.5-1.5b-instruct")
    parser.add_argument("--log", type=Path, default=benchlog.DEFAULT_LOG)
    args = parser.parse_args(argv)

    if benchlog.environment(args.log)["git_dirty"]:
        parser.error("tracked files have uncommitted changes; commit first, so that "
                     "each entry names the code that produced it")

    cfg = ModelConfig.from_card(args.model)
    tokens = cfg.max_position_embeddings
    elements = 2 * tokens * cfg.num_hidden_layers * cfg.num_key_value_heads * cfg.head_dim
    fp16_payload = payload_bytes(cfg, tokens, "FP16")
    for tier in Engine.KV_TIERS:
        cache = model.PagedCache(cfg, getattr(_microinfer.Tier, tier))
        cache.rope.cover(tokens)
        before = nvml.own_used_bytes()
        cache.pages.reserve(tokens)
        measured = nvml.own_used_bytes() - before
        del cache
        computed = paged_cache_bytes(cfg, tokens, tier)
        payload = payload_bytes(cfg, tokens, tier)
        results = {"measured_bytes": measured, "computed_bytes": computed,
                   "measured_over_computed": measured / computed,
                   "payload_bytes": payload, "effective_bits": 8 * payload / elements,
                   "compression_vs_fp16": fp16_payload / payload}
        print(f"{tier}: measured {measured / MIB:.1f} MiB, computed {computed / MIB:.1f} MiB, "
              f"{results['effective_bits']:.3f} effective bits, "
              f"{results['compression_vs_fp16']:.2f}x FP16")
        benchlog.append(
            "kv-footprint", model=args.model, context_length=tokens,
            precision_tiers={tier: 1.0},
            config={"measured": "nvml.own_used_bytes across reserving every page, "
                                "RoPE table covered first",
                    "computed": "paged_cache_bytes", "page_tokens": _microinfer.device.page_tokens,
                    "open_pages_per_layer": len(_microinfer.device.open_pages)
                    if tier != "FP16" else 0,
                    "issue": args.issue},
            results=results, log=args.log)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
