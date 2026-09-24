#!/usr/bin/env python3
"""Perplexity on held-out text at each static KV tier (#18).

    .venv/bin/python tools/perplexity.py --issue 18 [--model qwen2.5-0.5b-instruct] \\
        [--tiers FP16 INT8 INT4 INT2] [--window 2048] [--windows N]

The held-out set is WikiText-2's test split, raw, the set KVQuant reports
perplexity on (Hooper et al., arXiv:2401.18079, Table 1). The archive is
fetched once into data/, which git ignores, and checked against the hash it
had when this tool was written, so every run scores the same text.

The text is tokenised whole and cut into consecutive, non-overlapping windows
of `--window` tokens; a partial last window is dropped. Each window is run
through Engine.forward, and every position but the first is scored against
the token that follows it: teacher forcing over all input tokens, as KVQuant
measures it. Perplexity is exp of the mean negative log-likelihood over every
scored token of every window.

The cache is held at one tier for the whole run (Engine.kv_tier), so the
forward pass attends over quantised pages for every full span of P positions
and over the FP16 open page for the rest (ADR-0005). That is the same
condition KVQuant's perplexity measures: keys and values quantised as they
are attended to, with no full-precision prompt.

One engine serves every tier; one entry per tier goes to the benchmark log,
each carrying the FP16 figure if FP16 was run too. Refuses a dirty tree, as
the checkpoint recorder does.
"""

from __future__ import annotations

import argparse
import hashlib
import math
import sys
import time
import urllib.request
import zipfile
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from microinfer import Engine, benchlog  # noqa: E402

DATA = REPO / "data" / "wikitext-2-raw-v1.zip"
URL = "https://wikitext.smerity.com/wikitext-2-raw-v1.zip"
SHA256 = "ef7edb566e3e2b2d31b29c1fdb0c89a4cc683597484c3dc2517919c615435a11"
SPLIT = "wikitext-2-raw/wiki.test.raw"

#: Rows of logits turned into log-likelihoods at a time: a 2048-row window's
#: logits are 1.2 GB in fp32, and float64 work on all of them at once would
#: double it.
ROWS = 256


def held_out_text() -> str:
    if not DATA.exists():
        DATA.parent.mkdir(exist_ok=True)
        print(f"fetching {URL}")
        urllib.request.urlretrieve(URL, DATA)
    digest = hashlib.sha256(DATA.read_bytes()).hexdigest()
    if digest != SHA256:
        raise SystemExit(f"{DATA} has sha256 {digest}, not {SHA256}; delete it to fetch again")
    with zipfile.ZipFile(DATA) as z:
        return z.read(SPLIT).decode("utf-8")


def window_nll(logits: np.ndarray, targets: np.ndarray) -> float:
    """Summed negative log-likelihood of `targets[i]` under `logits[i]`,
    through a max-shifted log-sum-exp in float64."""
    total = 0.0
    for first in range(0, len(targets), ROWS):
        rows = logits[first:first + ROWS].astype(np.float64)
        top = rows.max(axis=1, keepdims=True)
        lse = top[:, 0] + np.log(np.exp(rows - top).sum(axis=1))
        total += float((lse - rows[np.arange(len(rows)), targets[first:first + ROWS]]).sum())
    return total


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--issue", type=int, required=True,
                        help="the ticket this measurement is made for")
    parser.add_argument("--model", default="qwen2.5-0.5b-instruct")
    parser.add_argument("--tiers", nargs="+", default=list(Engine.KV_TIERS),
                        choices=Engine.KV_TIERS)
    parser.add_argument("--window", type=int, default=2048)
    parser.add_argument("--windows", type=int, default=None,
                        help="score only the first N windows (default: all)")
    parser.add_argument("--log", type=Path, default=benchlog.DEFAULT_LOG)
    args = parser.parse_args(argv)

    if benchlog.environment(args.log)["git_dirty"]:
        parser.error("tracked files have uncommitted changes; commit first, so that "
                     "each entry names the code that produced it")

    engine = Engine(REPO / "models" / args.model)
    engine.load_weights()
    ids = engine.encode(held_out_text())
    windows = len(ids) // args.window
    if args.windows is not None:
        windows = min(windows, args.windows)
    print(f"{len(ids)} tokens; {windows} windows of {args.window}")

    results = {}
    for tier in args.tiers:
        engine.kv_tier = tier
        nll, scored, start = 0.0, 0, time.perf_counter()
        for w in range(windows):
            window = ids[w * args.window:(w + 1) * args.window]
            nll += window_nll(engine.forward(window)[:-1], window[1:])
            scored += len(window) - 1
        mean = nll / scored
        results[tier] = {"perplexity": math.exp(mean), "nll_mean": mean,
                         "scored_tokens": scored, "windows": windows,
                         "seconds": time.perf_counter() - start}
        if "FP16" in results and tier != "FP16":
            fp16 = results["FP16"]["perplexity"]
            results[tier].update(fp16_perplexity=fp16,
                                 relative_increase=results[tier]["perplexity"] / fp16 - 1)
        print(f"{tier}: perplexity {results[tier]['perplexity']:.4f} "
              f"({results[tier]['seconds']:.0f} s)")
        benchlog.append(
            "perplexity", model=args.model, context_length=args.window,
            precision_tiers={tier: 1.0},
            config={"dataset": "WikiText-2 raw, test split", "source": URL, "sha256": SHA256,
                    "window": args.window, "stride": args.window,
                    "scoring": "teacher forcing, every position but each window's first",
                    "kv_cache": engine.kv_cache, "prefill_chunk": engine.prefill_chunk,
                    "issue": args.issue},
            results=results[tier], log=args.log)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
