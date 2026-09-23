#!/usr/bin/env python3
"""How far chunked prefill is from single-shot prefill, per chunk size (#15).

    .venv/bin/python tools/chunked_agreement.py [--chunks 1 37 128 512]

For every prompt of the golden set, runs Engine.forward once single-shot and
once per chunk size, and compares the logits at every position: the largest
absolute difference, the largest KL(single || chunked), and how many argmaxes
changed, split by whether single-shot's top two logits were closer than
--near-tie there (by default tests/test_chunked_prefill.py's LOGIT_BOUND).

They differ at all because cuBLAS picks its algorithm from the GEMM's shape,
so a row rounds differently in a chunk-sized GEMM than in a prompt-sized one
(ADR-0006, note from #15). This is the measurement that note and the test's
bounds cite. adversarial-00 is reported apart from the rest: it amplifies
any rounding difference about a thousand times (ADR-0010), so its numbers
describe the prompt, not the chunking.

Appends one entry to the benchmark log. Refuses a dirty tree, as the
checkpoint recorder does.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from microinfer import Engine, benchlog  # noqa: E402
from microinfer.gate import kl_divergence  # noqa: E402
from microinfer.golden import GoldenSet  # noqa: E402

ILL_CONDITIONED = "adversarial-00"  # ADR-0010


def compare(single: np.ndarray, chunked: np.ndarray, near_tie_margin: float) -> dict:
    top_two = np.sort(single, -1)[:, -2:]
    near_tie = top_two[:, 1] - top_two[:, 0] < near_tie_margin
    flipped = single.argmax(-1) != chunked.argmax(-1)
    return {"max_abs_logit": float(np.abs(single - chunked).max()),
            "max_kl": float(kl_divergence(single, chunked).max()),
            "argmax_flips": int(flipped.sum()),
            "argmax_flips_not_near_tie": int((flipped & ~near_tie).sum())}


def worst(per_prompt: dict[str, dict]) -> dict:
    return {"max_abs_logit": max(r["max_abs_logit"] for r in per_prompt.values()),
            "max_kl": max(r["max_kl"] for r in per_prompt.values()),
            "argmax_flips": sum(r["argmax_flips"] for r in per_prompt.values()),
            "argmax_flips_not_near_tie": sum(r["argmax_flips_not_near_tie"]
                                             for r in per_prompt.values()),
            "worst_prompt": max(per_prompt, key=lambda p: per_prompt[p]["max_abs_logit"])}


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", default="qwen2.5-0.5b-instruct")
    parser.add_argument("--chunks", type=int, nargs="+", default=[1, 37, 128, 512])
    parser.add_argument("--near-tie", type=float, default=0.2)
    parser.add_argument("--log", type=Path, default=benchlog.DEFAULT_LOG)
    args = parser.parse_args(argv)

    if benchlog.environment(args.log)["git_dirty"]:
        parser.error("tracked files have uncommitted changes; commit first, so that "
                     "each entry names the code that produced it")

    engine = Engine(REPO / "models" / args.model, prefill_chunk=None)
    engine.load_weights()
    golden = GoldenSet(REPO / "tests" / "golden" / args.model)

    by_chunk: dict[int, dict[str, dict]] = {c: {} for c in args.chunks}
    for item in golden:
        engine.prefill_chunk = None
        single = engine.forward(item.token_ids)
        for chunk in args.chunks:
            engine.prefill_chunk = chunk
            by_chunk[chunk][item.prompt_id] = compare(single, engine.forward(item.token_ids),
                                                        args.near_tie)

    results, excluded = {}, {}
    for chunk, per_prompt in by_chunk.items():
        excluded[str(chunk)] = per_prompt.pop(ILL_CONDITIONED)
        results[str(chunk)] = worst(per_prompt)
        print(f"chunk {chunk:4}: {results[str(chunk)]}; {ILL_CONDITIONED}: {excluded[str(chunk)]}")

    lengths = [len(item) for item in golden if item.prompt_id != ILL_CONDITIONED]
    benchlog.append(
        "chunked-prefill-agreement", model=args.model, precision_tiers={"FP16": 1.0},
        context_length={"min": min(lengths), "max": max(lengths), "prompts": len(lengths)},
        config={"chunks": args.chunks, "compared_against": "single-shot prefill",
                "positions": "every position", "near_tie_margin": args.near_tie,
                "excluded": {ILL_CONDITIONED: "ill-conditioned, ADR-0010"},
                "kv_cache": engine.kv_cache, "issue": 15},
        results={"by_chunk": results, ILL_CONDITIONED: excluded},
        log=args.log)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
