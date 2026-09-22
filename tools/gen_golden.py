#!/usr/bin/env python3
"""Generate the reference the engine is judged against.

**This is the only file in the repository that may import torch** (ADR-0002).
It runs offline, in its own environment, and nothing in the engine's test suite
invokes it — tests compare against the files it leaves behind.

Two decisions are worth understanding before changing anything here.

**Generate in fp32, not in the checkpoint's bfloat16.** Qwen2.5 ships bf16,
whose relative ulp is 2^-8 — eight times coarser than the fp16 the engine
stores. A bf16 reference would carry more error than the implementation it is
meant to judge, and would exceed ADR-0006's own bound of 4 fp16 ulps before the
engine rounded once. A reference has to be more accurate than the thing it
measures; that is the same reasoning that puts the per-kernel references in
float64.

**Store selectively.** The vocabulary is 151,936 tokens, so full logits for
every position of every prompt would be gigabytes. What ADR-0006 actually needs
is narrower:

- top-1 agreement at *every* position — an argmax per position is enough, and
  costs four bytes;
- mean KL over a *sample* of positions — that needs full distributions, so full
  logits are kept for a handful of deterministically chosen positions;
- a per-layer diagnostic that names the first layer to diverge — needed only
  when the gate is already red, so hidden states are kept for a few short
  prompts rather than for all of them.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO = Path(__file__).resolve().parent.parent

#: Full logits are kept at this many positions per prompt, for the KL term.
LOGIT_SAMPLES = 8

#: Hidden states are kept only for prompts in these groups — enough to localise
#: a divergence without storing a gigabyte of activations.
HIDDEN_GROUPS = {"short"}


def sample_positions(length: int, count: int) -> np.ndarray:
    """Positions to keep full logits for: deterministic, and always including
    the last, which is the one that decides the next token."""
    if length <= count:
        return np.arange(length, dtype=np.int32)
    return np.unique(np.linspace(0, length - 1, count).round().astype(np.int32))


def config_fingerprint(model_dir: Path) -> str:
    """So a regenerated reference can be told apart from a stale one."""
    return hashlib.sha256((model_dir / "config.json").read_bytes()).hexdigest()[:16]


def generate(model_dir: Path, prompts_path: Path, out_dir: Path, seed: int) -> None:
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)

    print(f"loading {model_dir.name} in float32 (not the checkpoint's bfloat16)")
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForCausalLM.from_pretrained(model_dir, dtype=torch.float32)
    model.eval()

    prompts = [json.loads(line) for line in prompts_path.read_text().splitlines() if line.strip()]
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "model": model_dir.name,
        "config_sha256_16": config_fingerprint(model_dir),
        "dtype": "float32",
        "checkpoint_dtype": json.loads((model_dir / "config.json").read_text())["torch_dtype"],
        "seed": seed,
        "torch": torch.__version__,
        "transformers": __import__("transformers").__version__,
        "logit_samples": LOGIT_SAMPLES,
        "hidden_groups": sorted(HIDDEN_GROUPS),
        "prompts": [],
    }

    for prompt in prompts:
        ids = tokenizer(prompt["text"], return_tensors="pt").input_ids
        with torch.no_grad():
            out = model(ids, output_hidden_states=prompt["group"] in HIDDEN_GROUPS)

        logits = out.logits[0].to(torch.float32)          # (seq, vocab)
        length = logits.shape[0]
        positions = sample_positions(length, LOGIT_SAMPLES)

        arrays = {
            "token_ids": ids[0].numpy().astype(np.int32),
            "argmax": logits.argmax(-1).numpy().astype(np.int32),
            "logit_positions": positions,
            "logits": logits[positions].numpy().astype(np.float32),
        }
        if out.hidden_states is not None:
            # (layers + 1, seq, hidden) — the first entry is the embedding output.
            arrays["hidden_states"] = torch.stack(out.hidden_states)[:, 0].numpy().astype(np.float32)

        path = out_dir / f"{prompt['id']}.npz"
        np.savez(path, **arrays)

        manifest["prompts"].append({
            "id": prompt["id"],
            "group": prompt["group"],
            "tokens": int(length),
            "has_hidden_states": "hidden_states" in arrays,
            "bytes": path.stat().st_size,
        })
        print(f"  {prompt['id']:<16} {length:>5} tokens  {path.stat().st_size / 2**20:>7.1f} MiB")

    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    total = sum(p["bytes"] for p in manifest["prompts"])
    print(f"\n{len(prompts)} prompts, {total / 2**20:.1f} MiB total, manifest written")


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", default="qwen2.5-0.5b-instruct")
    parser.add_argument("--models-dir", type=Path, default=REPO / "models")
    parser.add_argument("--prompts", type=Path, default=REPO / "tools" / "prompts.jsonl")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    model_dir = args.models_dir / args.model
    if not (model_dir / "config.json").is_file():
        parser.error(f"{model_dir} has no config.json")

    generate(model_dir, args.prompts, args.out or REPO / "tests" / "golden" / args.model, args.seed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
