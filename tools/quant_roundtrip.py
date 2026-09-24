#!/usr/bin/env python3
"""What a quantised tier loses of the keys and values it holds, per tier (#16, #17).

    .venv/bin/python tools/quant_roundtrip.py --issue 17 \
        [--model qwen2.5-1.5b-instruct] [--tiers INT8 INT4 INT2]

For every prompt of the model's golden set, prefills it and takes every full
page of every layer as the cache holds it: keys rotated and without their bias
(ADR-0009), and values. Each page goes through quantise_page and
dequantise_page, and what comes back is compared in float64 with what went
in, which is exact there, being fp16. Keys and values are reported apart,
since they are quantised along different axes (ADR-0005):

- relative RMS error, ||x' - x|| / ||x||, and the same as a signal-to-noise
  ratio in dB;
- the largest absolute error;
- the error in steps of the group's own scale: the mean, and the largest.
  Half a step is what rounding alone costs; more, where the fp16 output
  cannot resolve a step;
- how many elements lie beyond the bound microinfer.quantisation derives,
  which tests/test_quant.py asserts on two prompts and this counts on all.

Each entry also carries the tier's layout: its metadata bytes and the
effective bits per element with them counted, which is the figure a
compression ratio must quote, and the issue the measurement was made for,
which the caller names: the tool serves every tier's ticket. Appends one
entry per tier. Refuses a dirty tree, as the checkpoint recorder does.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from microinfer import Engine, _microinfer, benchlog  # noqa: E402
from microinfer.golden import GoldenSet  # noqa: E402
from microinfer.quantisation import P, read_page, round_trip_bound  # noqa: E402


def in_steps(err: np.ndarray, scale: np.ndarray) -> np.ndarray:
    """The error in units of each element's own group scale. A group of zero
    range has scale 0 and comes back exactly, so it is left out."""
    scale = np.broadcast_to(scale, err.shape)
    live = scale > 0
    return err[live] / scale[live]


def summary(x: list[np.ndarray], err: list[np.ndarray], steps: list[np.ndarray],
            beyond: list[int]) -> dict:
    x, err, steps = np.concatenate(x), np.concatenate(err), np.concatenate(steps)
    rel = float(np.linalg.norm(err) / np.linalg.norm(x))
    return {"relative_rms": rel, "snr_db": float(-20 * np.log10(rel)),
            "max_abs": float(err.max()), "mean_steps": float(steps.mean()),
            "max_steps": float(steps.max()), "beyond_bound": int(sum(beyond)),
            "elements": int(x.size)}


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--issue", type=int, required=True,
                        help="the ticket this measurement is made for")
    parser.add_argument("--model", default="qwen2.5-1.5b-instruct")
    parser.add_argument("--tiers", nargs="+", default=["INT8", "INT4", "INT2"])
    parser.add_argument("--log", type=Path, default=benchlog.DEFAULT_LOG)
    args = parser.parse_args(argv)

    if benchlog.environment(args.log)["git_dirty"]:
        parser.error("tracked files have uncommitted changes; commit first, so that "
                     "each entry names the code that produced it")

    engine = Engine(REPO / "models" / args.model)
    engine.load_weights()
    cfg = engine.config
    heads, head_dim = cfg.num_key_value_heads, cfg.head_dim
    golden = GoldenSet(REPO / "tests" / "golden" / args.model)
    cached = [engine.cached_kv(item.token_ids) for item in golden if len(item) >= P]

    for name in args.tiers:
        tier = getattr(_microinfer.Tier, name)
        layout = _microinfer.quantised_page_layout(tier, heads, head_dim)
        seen = {"keys": ([], [], [], []), "values": ([], [], [], [])}
        pages = 0
        for keys, values in cached:
            for layer in range(cfg.num_hidden_layers):
                for start in range(0, keys.shape[1] // P * P, P):
                    k, v = keys[layer, start:start + P], values[layer, start:start + P]
                    page = _microinfer.quantise_page(k, v, tier)
                    got_k, got_v = _microinfer.dequantise_page(page, tier, heads, head_dim)
                    err = (np.abs(got_k.astype(np.float64) - k),
                           np.abs(got_v.astype(np.float64) - v))
                    parts = read_page(page, tier, heads, head_dim)
                    scale = (parts["key_scales"][None], parts["value_scales"][..., None])
                    got = (got_k, got_v)
                    for i, (part, x) in enumerate((("keys", k), ("values", v))):
                        seen[part][0].append(x.astype(np.float64).ravel())
                        seen[part][1].append(err[i].ravel())
                        seen[part][2].append(in_steps(err[i], scale[i]))
                        bound = round_trip_bound(x, got[i], scale[i])
                        seen[part][3].append(int((err[i] > bound).sum()))
                    pages += 1
        results = {part: summary(*lists) for part, lists in seen.items()}
        results.update(pages=pages, effective_bits=layout["effective_bits"],
                       metadata_bytes=layout["metadata_bytes"], page_bytes=layout["page_bytes"])
        print(f"{name}: {pages} pages, {layout['effective_bits']:.3f} effective bits; "
              f"keys {results['keys']}; values {results['values']}")
        lengths = [len(item) for item in golden if len(item) >= P]
        benchlog.append(
            "kv-quantisation-roundtrip", model=args.model, precision_tiers={name: 1.0},
            context_length={"min": min(lengths), "max": max(lengths), "prompts": len(lengths)},
            config={"reference": "float64, against the fp16 values the cache holds",
                    "data": "every full page of every layer of the golden prompts",
                    "keys": "per (head, channel) across the page, without bias (ADR-0009)",
                    "values": "per (head, token)", "page_tokens": P, "issue": args.issue},
            results=results, log=args.log)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
