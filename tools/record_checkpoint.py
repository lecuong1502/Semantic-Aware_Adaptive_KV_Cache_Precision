#!/usr/bin/env python3
"""Record the Milestone 0 checkpoint in the benchmark log (#12, #13).

    .venv/bin/python tools/record_checkpoint.py [--model qwen2.5-0.5b-instruct]

Appends three entries: the correctness gate, the greedy smoke test, and the
peak device memory of a generation. These are the log's first entries, the
first working version it was asked to start from.

Refuses to run on a working tree with uncommitted changes to tracked files. An
entry names the commit that produced it, and a dirty tree would make that name
a lie.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from microinfer import Engine, benchlog  # noqa: E402
from microinfer.gate import KL_MAX, TOP1_MIN, run_gate  # noqa: E402
from microinfer.golden import GoldenSet  # noqa: E402

#: Every page is FP16 in the non-paged engine. Keys are cached without their
#: projection's bias (ADR-0009), which is part of what "FP16" means here.
TIERS = {"FP16": 1.0}

#: The generation whose peak is recorded: long enough that the cache and the
#: prefill workspace are the largest things the engine holds.
PEAK_PROMPT, PEAK_NEW = 600, 64


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", default="qwen2.5-0.5b-instruct")
    parser.add_argument("--log", type=Path, default=benchlog.DEFAULT_LOG)
    args = parser.parse_args(argv)

    if benchlog.environment()["git_dirty"]:
        parser.error("tracked files have uncommitted changes; commit first, so that "
                     "each entry names the code that produced it")

    engine = Engine(REPO / "models" / args.model)
    engine.load_weights()
    golden = GoldenSet(REPO / "tests" / "golden" / args.model)
    lengths = [len(item) for item in golden]
    reference = {k: golden.manifest[k] for k in ("dtype", "torch", "transformers", "logit_samples")}

    # The gate.
    report = run_gate(engine, golden)
    print(report.render())
    benchlog.append(
        "gate", model=args.model, precision_tiers=TIERS,
        context_length={"min": min(lengths), "max": max(lengths), "prompts": len(lengths)},
        config={"reference": reference, "top1_min": TOP1_MIN, "kl_max": KL_MAX},
        results={
            "passed": report.passed,
            "top1": report.top1,
            "positions": report.positions,
            "kl_mean": report.kl_mean,
            "kl_samples": report.kl_samples,
            "per_prompt": [{"prompt": p.prompt_id, "positions": p.positions,
                            "disagreeing": p.disagreeing, "kl_mean": float(p.kl.mean()),
                            "kl_max": float(p.kl.max())} for p in report.prompts],
        },
        log=args.log)

    # The smoke test: greedy tokens against HuggingFace's.
    smoke, continued = [], []
    for item in golden:
        if item.generated is None:
            continue
        ours = engine.generate(item.token_ids, max_new_tokens=len(item.generated))
        n = min(len(ours), len(item.generated))
        first = next((i for i in range(n) if ours[i] != item.generated[i]), n)
        same = len(ours) == len(item.generated) and first == n
        smoke.append({"prompt": item.prompt_id, "match": same,
                      "first_divergence": None if same else first})
        continued.append(len(item) + len(item.generated))
    matching = sum(s["match"] for s in smoke)
    print(f"\nsmoke: {matching}/{len(smoke)} match HuggingFace")
    benchlog.append(
        "smoke", model=args.model, precision_tiers=TIERS,
        context_length={"min": min(continued), "max": max(continued), "prompts": len(smoke),
                        "new_tokens": golden.manifest["smoke_tokens"]},
        config={"reference": reference, "greedy": golden.manifest["greedy"]},
        results={"matching": matching, "prompts": len(smoke), "per_prompt": smoke},
        log=args.log)

    # Peak device memory, after a warm run (tests/test_inference.py says why).
    ids = np.arange(100, 100 + PEAK_PROMPT, dtype=np.int32)
    engine.generate(ids, PEAK_NEW, stop_at_eos=False)
    engine.reset_peak()
    engine.generate(ids, PEAK_NEW, stop_at_eos=False)
    peak = engine.peak_footprint()
    print("\n" + peak.render())
    benchlog.append(
        "peak_memory", model=args.model, precision_tiers=TIERS,
        context_length=PEAK_PROMPT + PEAK_NEW,
        config={"prompt_tokens": PEAK_PROMPT, "new_tokens": PEAK_NEW, "warm": True},
        results={"weights": peak.weights, "kv_cache": peak.kv_cache,
                 "workspace": peak.workspace, "engine_total": peak.engine_total,
                 "unaccounted": peak.unaccounted, "device_free": peak.device_free,
                 "device_total": peak.device_total},
        log=args.log)

    print(f"\nappended 3 entries to {args.log.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
