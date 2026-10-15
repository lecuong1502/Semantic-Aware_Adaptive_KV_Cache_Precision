"""Measure each layer's sensitivity to INT2, for ADR-0018's layer weights (#145).

Each layer alone is held at INT2, every other at FP16, through a tier map, on
WikiText-2's test split, in windows read by teacher forcing as
tools/perplexity.py reads them. A layer's sensitivity is the KL divergence
of the next-token distribution from FP16's, KL(FP16 || one layer at INT2),
the mean over every position but each window's last. One entry goes to the
benchmark log, a divergence for each layer, in layer order; normalised,
they are the weights controller.logged_layer_weights gives the planner.
Refuses a dirty tree, as the checkpoint recorder does.
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "tools"))

from perplexity import ATTENTION, ROWS, SHA256, URL, held_out_text  # noqa: E402

from microinfer import Engine, benchlog, gate, model  # noqa: E402


def one_layer_at(layer: int, *, layers: int, pages: int, tier: str = "INT2") -> list[list[str]]:
    """The tier map that holds `layer` at `tier` over `pages` pages, and
    every other layer at the engine's kv_tier, FP16."""
    return [[tier] * pages if row == layer else [] for row in range(layers)]


def summed_kl(reference: np.ndarray, logits: np.ndarray) -> float:
    """KL(reference || ours) summed over rows, `ROWS` rows at a time: a
    2048-row window's logits in float64 at once would be 2.5 GB a copy."""
    return sum(float(gate.kl_divergence(reference[r:r + ROWS], logits[r:r + ROWS]).sum())
               for r in range(0, len(reference), ROWS))


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--issue", type=int, required=True,
                        help="the ticket this measurement is made for")
    parser.add_argument("--model", default="qwen2.5-1.5b-instruct")
    parser.add_argument("--window", type=int, default=2048)
    parser.add_argument("--windows", type=int, default=8,
                        help="measure on the first N windows")
    parser.add_argument("--log", type=Path, default=benchlog.DEFAULT_LOG)
    args = parser.parse_args(argv)

    if benchlog.environment(args.log)["git_dirty"]:
        parser.error("tracked files have uncommitted changes; commit first, so that "
                     "each entry names the code that produced it")

    engine = Engine(REPO / "models" / args.model)
    engine.load_weights()
    layers = engine.config.num_hidden_layers
    pages = math.ceil(args.window / model.device.page_tokens)
    ids = engine.encode(held_out_text())
    windows = min(len(ids) // args.window, args.windows)
    print(f"{layers} layers; {windows} windows of {args.window}")

    kl, scored, start = np.zeros(layers), 0, time.perf_counter()
    for w in range(windows):
        window = ids[w * args.window:(w + 1) * args.window]
        engine.kv_tier_map = None
        reference = engine.forward(window)[:-1]
        for layer in range(layers):
            engine.kv_tier_map = one_layer_at(layer, layers=layers, pages=pages)
            kl[layer] += summed_kl(reference, engine.forward(window)[:-1])
        scored += len(window) - 1
        print(f"window {w + 1}/{windows}: {time.perf_counter() - start:.0f} s")
    engine.kv_tier_map = None
    kl /= scored
    for layer, divergence in enumerate(kl):
        print(f"layer {layer:2d}: KL {divergence:.6f}")

    benchlog.append(
        "layer-sensitivity", model=args.model, context_length=args.window,
        precision_tiers={"FP16": (layers - 1) / layers, "INT2": 1 / layers},
        config={"dataset": "WikiText-2 raw, test split", "source": URL, "sha256": SHA256,
                "window": args.window, "stride": args.window,
                "measured": "KL(FP16 || one layer alone at INT2) of the next-token "
                            "distribution, the mean over every position but each "
                            "window's last",
                "kv_cache": engine.kv_cache, "prefill_chunk": engine.prefill_chunk,
                "attention": ATTENTION, "issue": args.issue},
        results={"kl": kl.tolist(), "scored_tokens": scored, "windows": windows,
                 "seconds": time.perf_counter() - start},
        log=args.log)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
