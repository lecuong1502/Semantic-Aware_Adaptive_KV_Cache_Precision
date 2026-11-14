#!/usr/bin/env python3
"""How far one build's decode logits are from another's, by ADR-0006's gate (#204).

    .venv/bin/python tools/decode_agreement.py --save before.npz          # on the old build
    .venv/bin/python tools/decode_agreement.py --against before.npz --issue 204   # on the new

A change to the attention kernel that changes its floating-point summation
order, as splitting a query's keys across blocks does, changes logits in
their last bits. Its correctness gate is ADR-0006's top-1 agreement and KL,
taken here between the two builds rather than against HuggingFace: the old
build is the reference, and the new one must agree with it.

The golden prompts are too short for that. A decode step splits its keys only
at long contexts, so this reads a held-out text (WikiText-2's test split, as
tools/perplexity.py does), prefills `--context` positions in the engine's
chunks, and then teacher-forces `--forced` positions one by one, as decode
steps: every one of those positions' logits comes from a decode launch over a
long cache. Each model runs at every tier and over a mixed map, which cycles
the four tiers page by page, each layer's row offset by one.

`--save` keeps, for each (model, tier), the argmax at every forced position
and the whole logits at `--samples` of them, evenly spaced and always the
last (ADR-0006's amendment on the KL sample). `--against` compares: top-1
agreement over every forced position, and mean KL(reference || ours) over the
sampled ones, against the gate's 99% and 1e-3. Appends one
"decode-agreement" entry. Refuses a dirty tree when logging.
"""

from __future__ import annotations

import argparse
import sys
import time
import zipfile
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from microinfer import Engine, benchlog  # noqa: E402
from microinfer.gate import KL_MAX, TOP1_MIN, kl_divergence  # noqa: E402

DATA = REPO / "data" / "wikitext-2-raw-v1.zip"
SPLIT = "wikitext-2-raw/wiki.test.raw"
MODELS = ("qwen2.5-0.5b-instruct", "qwen2.5-1.5b-instruct")
CONFIGS = ("FP16", "INT8", "INT4", "INT2", "mixed")


def mixed_map(layers: int, pages: int) -> list[list[str]]:
    order = ["FP16", "INT8", "INT4", "INT2"]
    return [[order[(i + layer) % 4] for i in range(pages)] for layer in range(layers)]


def forced_logits(engine: Engine, config: str, ids: np.ndarray, context: int) -> np.ndarray:
    """The logits after each position from `context` on, each a decode step's."""
    pages = -(-len(ids) // 32) + 1
    if config == "mixed":
        engine.kv_tier, engine.kv_tier_map = "FP16", mixed_map(engine.config.num_hidden_layers,
                                                               pages)
    else:
        engine.kv_tier, engine.kv_tier_map = config, None
    with engine.conversation() as conversation:
        conversation.force(ids[:context])
        return conversation.force(ids[context:], one_by_one=True, logits=True)


def sampled(n: int, samples: int) -> np.ndarray:
    return np.unique(np.linspace(0, n - 1, samples).round().astype(int))


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    what = parser.add_mutually_exclusive_group(required=True)
    what.add_argument("--save", type=Path, help="keep this build's logits as the reference")
    what.add_argument("--against", type=Path, help="compare this build with a reference")
    parser.add_argument("--issue", type=int, help="the ticket a comparison is logged for")
    parser.add_argument("--models", nargs="+", default=list(MODELS), choices=MODELS)
    parser.add_argument("--configs", nargs="+", default=list(CONFIGS), choices=CONFIGS)
    parser.add_argument("--context", type=int, default=8192)
    parser.add_argument("--forced", type=int, default=128)
    parser.add_argument("--samples", type=int, default=32)
    parser.add_argument("--log", type=Path, default=benchlog.DEFAULT_LOG)
    args = parser.parse_args(argv)

    if args.against is not None:
        if args.issue is None:
            parser.error("--against logs an entry, and needs --issue")
        if benchlog.environment(args.log)["git_dirty"]:
            parser.error("tracked files have uncommitted changes; commit first, so that "
                         "each entry names the code that produced it")
        reference = dict(np.load(args.against))
        args.context, args.forced = int(reference["context"]), int(reference["forced"])
        args.samples = int(reference["samples"])
    with zipfile.ZipFile(DATA) as z:
        text = z.read(SPLIT).decode("utf-8")

    keep: dict[str, np.ndarray] = {"context": np.array(args.context),
                                   "forced": np.array(args.forced),
                                   "samples": np.array(args.samples)}
    results: dict[str, dict] = {}
    for name in args.models:
        engine = Engine(REPO / "models" / name)
        engine.load_weights()
        ids = engine.encode(text)[: args.context + args.forced]
        at = sampled(args.forced, args.samples)
        results[name] = {}
        for config in args.configs:
            start = time.perf_counter()
            logits = forced_logits(engine, config, ids, args.context)
            seconds = time.perf_counter() - start
            key = f"{name}/{config}"
            if args.save is not None:
                keep[f"{key}/argmax"] = logits.argmax(-1)
                keep[f"{key}/logits"] = logits[at]
                print(f"{key}: kept {len(logits)} positions ({seconds:.0f} s)")
                continue
            agree = float((logits.argmax(-1) == reference[f"{key}/argmax"]).mean())
            kl = kl_divergence(reference[f"{key}/logits"], logits[at])
            diff = np.abs(reference[f"{key}/logits"] - logits[at])
            results[name][config] = {
                "top1": agree, "kl_mean": float(kl.mean()), "kl_max": float(kl.max()),
                "kl_samples": len(at), "max_abs_logit": float(diff.max()),
                "passed": agree >= TOP1_MIN and float(kl.mean()) < KL_MAX,
                "seconds": seconds}
            print(f"{key}: top-1 {agree:.4f}, KL mean {kl.mean():.2e} max {kl.max():.2e}, "
                  f"|logit| max {diff.max():.3g} ({seconds:.0f} s)")
        del engine

    if args.save is not None:
        np.savez(args.save, **keep)
        return 0
    passed = all(r["passed"] for per in results.values() for r in per.values())
    benchlog.append(
        "decode-agreement", model=list(args.models),
        context_length={"min": args.context + 1, "max": args.context + args.forced},
        precision_tiers=None,
        config={"issue": args.issue, "reference": "the build before the change, decode logits",
                "text": SPLIT, "context": args.context, "forced": args.forced,
                "kl_samples": args.samples, "configs": list(args.configs),
                "mixed": "four tiers cycled by page, each layer's row offset by one",
                "top1_min": TOP1_MIN, "kl_max": KL_MAX},
        results={"passed": passed, "per_model": results}, log=args.log)
    print("passed" if passed else "FAILED")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
