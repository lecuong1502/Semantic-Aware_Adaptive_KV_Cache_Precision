#!/usr/bin/env python3
"""How far prefill on the tensor cores is from the kernel before it, by ADR-0006's gate (#213).

    .venv/bin/python tools/prefill_agreement.py --issue 213
        [--models ...] [--configs ...] [--context 8192] [--chunk 512] [--pages 16]

The kernel that prefills on the tensor cores rounds in other places than the
one before it, which a decode step still runs, so prefill's logits differ in
their last bits. Its correctness gate is ADR-0006's top-1 agreement and KL,
between the two kernels, which one build holds both of
(device.prefill_on_tensor_cores): the one before is the reference. Each model
runs at every tier and over a mixed map, which cycles the four tiers page by
page, each layer's row offset by one, over WikiText-2's test split.

- **One page over one cache.** A context is prefilled by the tensor cores, and
  the page after it by each kernel, over the same cache to the bit. No page
  seals before its last position, so what differs is the kernel's alone.
  --pages of them, at contexts spaced evenly to --context.
- **One chunk over one cache.** The same with the chunk of --chunk after
  --context: also what a tier makes of the difference, since a key that moved
  in its last bit can round to another code as its page seals.
- **A whole prefill.** Every position prefilled by each kernel, and the
  chunk's logits compared: what a tier makes of the difference over a cache.
- **The control** is the kernel before against itself, prefilled in chunks of
  256 rather than 512, which moves the same logits by cuBLAS's rounding alone
  (ADR-0006, note from #15): how far a tier carries noise of that size.

Then 32 positions are forced one by one after each whole prefill, as decode
steps, and the importance scores they leave are compared: a step's per-page
attention mass is the decode kernel's, which is unchanged, over the cache
each prefill left. Their largest difference, as a share of the largest score,
and the rank correlation are logged, with the control's.

Appends one "prefill-agreement" entry. Refuses a dirty tree.
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
sys.path.insert(0, str(REPO / "tools"))

from decode_agreement import (  # noqa: E402
    CONFIGS, CONTROL_CHUNK, DATA, MODELS, PAGE_TOKENS, SPLIT, gate, mixed_map, sampled)

from microinfer import Engine, _microinfer, benchlog  # noqa: E402
from microinfer.gate import KL_MAX, TOP1_MIN  # noqa: E402
from microinfer.page_damage import spearman  # noqa: E402

#: Decode steps forced after a prefill, for the importance scores they leave.
STEPS = 32
on_tensor_cores = _microinfer.device.prefill_on_tensor_cores


def run(engine: Engine, ids: np.ndarray, context: int, chunk: int, *, context_on: bool,
        chunk_on: bool, scores: bool) -> tuple[np.ndarray, np.ndarray | None]:
    """The logits after each of the `chunk` positions after `context`, the
    context prefilled on the tensor cores or not, and the chunk; and the
    scores STEPS decode steps after them leave, if asked for."""
    asked = on_tensor_cores(context_on)
    try:
        with engine.conversation() as conversation:
            conversation.force(ids[:context])
            on_tensor_cores(chunk_on)
            logits = conversation.force(ids[context:context + chunk], logits=True)
            if not scores:
                return logits, None
            conversation.force(ids[context + chunk:context + chunk + STEPS], one_by_one=True)
            return logits, conversation.scores()
    finally:
        on_tensor_cores(asked)


def page_contexts(context: int, pages: int) -> list[int]:
    """Where the pages compared begin: evenly spaced page boundaries, the
    last at `context`."""
    return [max(1, round(context * i / pages / PAGE_TOKENS)) * PAGE_TOKENS
            for i in range(1, pages + 1)]


def scores_apart(reference: np.ndarray, scores: np.ndarray) -> dict:
    """How far two runs' importance scores are: their largest difference as a
    share of the reference's largest score, and their rank correlation."""
    held = np.isfinite(reference) & np.isfinite(scores)
    a, b = reference[held], scores[held]
    return {"max_difference": float(np.abs(a - b).max()), "largest": float(a.max()),
            "max_relative_to_largest": float(np.abs(a - b).max() / a.max()),
            "spearman": spearman(a.ravel(), b.ravel())}


def compare(engine: Engine, ids: np.ndarray, args: argparse.Namespace) -> dict:
    chunk, context = args.chunk, args.context
    at = sampled(chunk, args.samples)

    def gated(reference, logits):
        return gate(reference.argmax(-1), reference[at], logits.argmax(-1), logits[at])

    new, new_scores = run(engine, ids, context, chunk, context_on=True, chunk_on=True,
                          scores=True)
    over_same, _ = run(engine, ids, context, chunk, context_on=True, chunk_on=False,
                       scores=False)
    before, before_scores = run(engine, ids, context, chunk, context_on=False, chunk_on=False,
                                scores=True)
    engine_chunk, engine.prefill_chunk = engine.prefill_chunk, CONTROL_CHUNK
    try:
        control, control_scores = run(engine, ids, context, chunk, context_on=False,
                                      chunk_on=False, scores=True)
    finally:
        engine.prefill_chunk = engine_chunk
    one_page = {on: np.concatenate([
        run(engine, ids, at_page, PAGE_TOKENS, context_on=True, chunk_on=on, scores=False)[0]
        for at_page in page_contexts(context, args.pages)]) for on in (False, True)}
    few = sampled(len(one_page[True]), args.samples)
    return {
        "one_page": gate(one_page[False].argmax(-1), one_page[False][few],
                         one_page[True].argmax(-1), one_page[True][few]),
        "one_chunk": gated(over_same, new),
        "whole_prefill": gated(before, new),
        "control_before_in_chunks_of_256": gated(before, control),
        "scores": scores_apart(before_scores, new_scores),
        "scores_control": scores_apart(before_scores, control_scores)}


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--issue", type=int, required=True)
    parser.add_argument("--models", nargs="+", default=list(MODELS))
    parser.add_argument("--configs", nargs="+", default=list(CONFIGS), choices=CONFIGS)
    parser.add_argument("--context", type=int, default=8192)
    parser.add_argument("--chunk", type=int, default=512)
    parser.add_argument("--pages", type=int, default=16)
    parser.add_argument("--samples", type=int, default=32)
    parser.add_argument("--log", type=Path, default=benchlog.DEFAULT_LOG)
    args = parser.parse_args(argv)
    if benchlog.environment(args.log)["git_dirty"]:
        parser.error("tracked files have uncommitted changes; commit first, so that "
                     "each entry names the code that produced it")
    with zipfile.ZipFile(DATA) as z:
        text = z.read(SPLIT).decode("utf-8")
    results: dict[str, dict] = {}
    for name in args.models:
        engine = Engine(REPO / "models" / name, kv_scoring=True)
        engine.load_weights()
        if args.chunk > engine.prefill_chunk:
            parser.error(f"a chunk is at most the engine's, {engine.prefill_chunk}")
        ids = engine.encode(text)[: args.context + args.chunk + STEPS]
        pages = -(-len(ids) // PAGE_TOKENS) + 1
        results[name] = {}
        for config in args.configs:
            mixed = config == "mixed"
            engine.kv_tier = "FP16" if mixed else config
            engine.kv_tier_map = (mixed_map(engine.config.num_hidden_layers, pages) if mixed
                                  else None)
            start = time.perf_counter()
            r = results[name][config] = compare(engine, ids, args)
            print(f"{name}/{config} ({time.perf_counter() - start:.0f} s)")
            for what, v in r.items():
                print(f"  {what}: " + (
                    f"top-1 {v['top1']:.4f}, KL mean {v['kl_mean']:.2e}"
                    f"{'' if v['passed'] else '  (fails)'}" if "top1" in v else
                    f"largest difference {v['max_relative_to_largest']:.2e} of the largest, "
                    f"Spearman {v['spearman']:.6f}"), flush=True)
        del engine
    gates = [k for k in next(iter(next(iter(results.values())).values())) if "scores" not in k]
    passed = {what: all(r[what]["passed"] for per in results.values() for r in per.values())
              for what in gates}
    benchlog.append(
        "prefill-agreement", model=list(args.models),
        context_length={"min": args.context + 1, "max": args.context + args.chunk},
        precision_tiers=None,
        config={"issue": args.issue,
                "reference": "the kernel before, device.prefill_on_tensor_cores(False)",
                "text": SPLIT, "context": args.context, "chunk": args.chunk,
                "pages": args.pages,
                "page_contexts": page_contexts(args.context, args.pages),
                "kl_samples": args.samples, "configs": list(args.configs),
                "decode_steps_for_scores": STEPS,
                "mixed": "four tiers cycled by page, each layer's row offset by one",
                "top1_min": TOP1_MIN, "kl_max": KL_MAX, "control_chunk": CONTROL_CHUNK},
        results={"passed": passed, "per_model": results}, log=args.log)
    print(passed)
    return 0 if passed["one_page"] else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
