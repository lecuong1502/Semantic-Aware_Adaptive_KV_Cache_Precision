"""Measure each layer's sensitivity to INT2, for ADR-0018's layer weights (#145).

Each layer alone is held at INT2, every other at FP16, through a tier map, on
WikiText-2's test split, in windows read by teacher forcing as
tools/perplexity.py reads them. A layer's sensitivity is the KL divergence
of the next-token distribution from FP16's, KL(FP16 || one layer at INT2),
the mean over every position but each window's last. One entry goes to the
benchmark log, a divergence for each layer, in layer order; normalised,
they are the weights controller.logged_layer_weights gives the planner.
Refuses a dirty tree, as the checkpoint recorder does.

A query reads its own page at FP16 (ADR-0011), so a window's first page
sees no INT2 at all, and every position up to 31 neighbours at FP16. The
bias is alike for every layer, and normalising leaves the weights' ratios.
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "tools"))

from perplexity import ATTENTION, ROWS, SHA256, URL, held_out_text  # noqa: E402

from microinfer import Engine, benchlog, model  # noqa: E402


#: Threads the divergences are taken on: about 5 GB of temporaries at once,
#: on top of the reference's p.
WORKERS = 4


def one_layer_at(layer: int, *, layers: int, pages: int, tier: str = "INT2") -> list[list[str]]:
    """The tier map that holds `layer` at `tier` over `pages` pages, and
    every other layer at the engine's kv_tier, FP16."""
    return [[tier] * pages if row == layer else [] for row in range(layers)]


class Reference:
    """FP16's next-token distributions over one window, to measure others'
    KL divergence from, `ROWS` rows at a time.

    KL(p || q) of a row is sum p log p - sum p z + log sum exp z, z being the
    row's logits under q: what depends on the reference alone, p and the
    first sum, is taken once a window rather than once a layer, which more
    than halves the work. In float64, as gate.kl_divergence is; the window's
    p take 2.5 GB at 2048 positions. Chunks are taken on threads, as NumPy
    lets go of the GIL in its loops, each holding about four float64 copies
    of its rows, 0.3 GB each on Qwen's vocabulary: WORKERS bounds that."""

    def __init__(self, logits: np.ndarray, pool: ThreadPoolExecutor):
        self.pool = pool
        chunks = list(pool.map(self._reference, self._chunks(logits)))
        self.p = [p for p, _ in chunks]
        self.negentropy = sum(negentropy for _, negentropy in chunks)

    @staticmethod
    def _chunks(logits: np.ndarray) -> list[np.ndarray]:
        return [logits[r:r + ROWS] for r in range(0, len(logits), ROWS)]

    @staticmethod
    def _reference(rows: np.ndarray) -> tuple[np.ndarray, float]:
        z = rows.astype(np.float64)
        z -= z.max(-1, keepdims=True)
        log_p = z - np.log(np.exp(z).sum(-1, keepdims=True))
        p = np.exp(log_p)
        return p, float((p * log_p).sum())

    def summed_kl(self, logits: np.ndarray) -> float:
        """KL(reference || ours), summed over the window's rows."""
        def cross(p: np.ndarray, rows: np.ndarray) -> float:
            z = rows.astype(np.float64)
            top = z.max(-1)
            lse = top + np.log(np.exp(z - top[:, None]).sum(-1))
            return float(lse.sum() - np.einsum("ij,ij->", p, z))

        return self.negentropy + sum(self.pool.map(cross, self.p, self._chunks(logits)))


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
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        for w in range(windows):
            window = ids[w * args.window:(w + 1) * args.window]
            engine.kv_tier_map = None
            reference = Reference(engine.forward(window)[:-1], pool)
            for layer in range(layers):
                engine.kv_tier_map = one_layer_at(layer, layers=layers, pages=pages)
                kl[layer] += reference.summed_kl(engine.forward(window)[:-1])
            del reference
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
