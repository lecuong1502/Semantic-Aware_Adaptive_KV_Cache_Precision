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
sampled ones, against the gate's 99% and 1e-3, three ways:

- **every forced position;**
- **before a forced page seals:** the forced positions whose cache, but for
  their own open page at FP16, is the prefill's, which both builds compute
  alike. Their difference is the kernel's alone. Once a page of forced
  positions seals at a quantised tier, a key that moved in its last bit can
  round to another code, and the difference grows by what the tier makes of
  such noise, not by anything the kernel did;
- **the control:** this build against itself, prefilled in chunks of 256
  rather than 512, which moves the same logits by cuBLAS's rounding alone
  (ADR-0006, note from #15). It is how far a tier carries noise of that size.

Appends one "decode-agreement" entry. Refuses a dirty tree when logging.
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

from microinfer import Engine, _microinfer, benchlog  # noqa: E402
from microinfer.gate import KL_MAX, TOP1_MIN, kl_divergence  # noqa: E402

DATA = REPO / "data" / "wikitext-2-raw-v1.zip"
SPLIT = "wikitext-2-raw/wiki.test.raw"
MODELS = ("qwen2.5-0.5b-instruct", "qwen2.5-1.5b-instruct")
CONFIGS = ("FP16", "INT8", "INT4", "INT2", "mixed")
#: The build's page size (ADR-0004).
PAGE_TOKENS = _microinfer.device.page_tokens
#: The control's prefill chunk: another GEMM shape, the same arithmetic.
CONTROL_CHUNK = 256


def mixed_map(layers: int, pages: int) -> list[list[str]]:
    order = ["FP16", "INT8", "INT4", "INT2"]
    return [[order[(i + layer) % 4] for i in range(pages)] for layer in range(layers)]


def gate(reference_argmax, reference_logits, argmax, logits) -> dict:
    """ADR-0006's two terms between two runs' logits: top-1 over every
    position given, KL over the sampled ones."""
    agree = float((argmax == reference_argmax).mean())
    kl = kl_divergence(reference_logits, logits)
    return {"top1": agree, "positions": len(argmax), "kl_mean": float(kl.mean()),
            "kl_max": float(kl.max()), "kl_samples": len(kl),
            "max_abs_logit": float(np.abs(reference_logits - logits).max()),
            "passed": agree >= TOP1_MIN and float(kl.mean()) < KL_MAX}


def forced_logits(engine: Engine, config: str, ids: np.ndarray, context: int) -> np.ndarray:
    """The logits after each position from `context` on, each a decode step's."""
    pages = -(-len(ids) // PAGE_TOKENS) + 1
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
        chunk = engine.prefill_chunk
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
            ref_argmax, ref_logits = reference[f"{key}/argmax"], reference[f"{key}/logits"]
            argmax = logits.argmax(-1)
            # Forced positions before the first page of them seals.
            unsealed = PAGE_TOKENS - args.context % PAGE_TOKENS
            early = at < unsealed
            engine.prefill_chunk = CONTROL_CHUNK
            control = forced_logits(engine, config, ids, args.context)
            engine.prefill_chunk = chunk
            results[name][config] = {
                "every_position": gate(ref_argmax, ref_logits, argmax, logits[at]),
                "before_a_forced_page_seals": gate(ref_argmax[:unsealed],
                                                   ref_logits[early], argmax[:unsealed],
                                                   logits[at[early]]),
                "control_same_build_chunks_of_256": gate(argmax, logits[at],
                                                         control.argmax(-1), control[at]),
                "seconds": seconds}
            for what, r in results[name][config].items():
                if what != "seconds":
                    print(f"{key} {what}: top-1 {r['top1']:.4f}, KL mean {r['kl_mean']:.2e} "
                          f"max {r['kl_max']:.2e}{'' if r['passed'] else '  (fails)'}")
        del engine

    if args.save is not None:
        np.savez(args.save, **keep)
        return 0
    passed = {what: all(r[what]["passed"] for per in results.values() for r in per.values())
              for what in ("every_position", "before_a_forced_page_seals",
                           "control_same_build_chunks_of_256")}
    benchlog.append(
        "decode-agreement", model=list(args.models),
        context_length={"min": args.context + 1, "max": args.context + args.forced},
        precision_tiers=None,
        config={"issue": args.issue, "reference": "the build before the change, decode logits",
                "text": SPLIT, "context": args.context, "forced": args.forced,
                "kl_samples": args.samples, "configs": list(args.configs),
                "mixed": "four tiers cycled by page, each layer's row offset by one",
                "top1_min": TOP1_MIN, "kl_max": KL_MAX, "control_chunk": CONTROL_CHUNK},
        results={"passed": passed, "per_model": results}, log=args.log)
    print(passed)
    return 0 if passed["before_a_forced_page_seals"] else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
