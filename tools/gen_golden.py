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

- **Top-1 agreement at every position.** An argmax per position satisfies it in
  four bytes, so this term of the gate is exact.
- **Mean KL.** This needs whole distributions, and a whole distribution is
  151,936 floats. ADR-0006 defines the term over a stated sample for that
  reason; this file supplies the sample, it does not redefine the gate.
- **A per-layer diagnostic** naming the first layer to diverge. Wanted only once
  the gate is already red. Kept for a subset of prompts, and within those only
  at the sampled positions: the question is which *layer* went wrong, which
  needs every layer at a few positions rather than every position.

**Top-k truncation was tried and rejected**, because it looks obviously right
and is not. Storing only the most probable tokens per position would have made
every position affordable. Measured on the real 0.5B references, the probability
mass outside the top 2048 tokens reaches **0.35** at the least certain
positions — the model is genuinely uncertain early in a prompt, and a truncated
distribution would corrupt KL far past the 1e-3 bound. Recorded here because the
idea is attractive enough that someone will propose it again.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO = Path(__file__).resolve().parent.parent

# The fingerprint and the prompt loader are shared with the engine's reader
# rather than reimplemented, because two copies of a fingerprint drift and a
# drifted fingerprint silently stops catching stale references.
#
# Loaded by path, not as `microinfer.golden`, and that detail is the ADR-0002
# wall showing itself. Importing the package runs its __init__, which loads the
# CUDA extension — and this environment deliberately has no CUDA extension and
# no GPU-capable torch. golden.py itself imports only the standard library and
# NumPy, so loading the module alone is safe.
def _load_shared():
    spec = importlib.util.spec_from_file_location(
        "_microinfer_golden_shared", REPO / "src" / "microinfer" / "golden.py"
    )
    module = importlib.util.module_from_spec(spec)
    # Registered before execution: @dataclass resolves its own module through
    # sys.modules, and a module loaded by path is not there unless put there.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


_shared = _load_shared()
config_fingerprint = _shared.config_fingerprint
load_prompts = _shared.load_prompts

#: Positions per prompt at which full logits are kept, for ADR-0006's KL term.
#:
#: Not derived from a measurement, because the measurement is not available yet:
#: KL is between this reference and the engine, and the engine does not produce
#: logits until #12. ADR-0006's amendment records that and makes validating this
#: number part of that ticket. Until then it is a storage budget — 16 positions
#: across 24 prompts is roughly 230 MiB per model.
LOGIT_SAMPLES = 16

#: Hidden states are kept only for prompts in these groups.
#:
#: `adversarial` is here deliberately. A first-diverging layer is most likely to
#: show on a long prompt, where RoPE at high positions and cache precision
#: actually bite — so keeping the diagnostic only for short prompts would leave
#: a red gate on a long one with nothing to localise it.
HIDDEN_GROUPS = {"short", "adversarial"}

#: ADR-0006's smoke test: greedy continuations of this many tokens, for this
#: many prompts spread evenly over the set so that every group is represented.
SMOKE_PROMPTS = 10
SMOKE_TOKENS = 64

#: Prompts whose logits are kept at *every* position, in a file of their own.
#: ADR-0006's second amendment makes #12 validate the 16-position KL sample
#: against a dense mean for at least one prompt. adversarial-00 is the prompt
#: on which the engine and the reference differ most, so it is where a sample
#: is most likely to mislead; long-03 is an ordinary long prompt beside it.
DENSE_PROMPTS = ("adversarial-00", "long-03")

#: Greedy means greedy. The instruct checkpoints' generation_config.json sets
#: do_sample, temperature, top_p, top_k and a repetition_penalty of 1.1, and
#: HuggingFace applies the penalty even when sampling is off. Each is overridden
#: explicitly, so the reference is argmax at every step and nothing else.
GREEDY = {"do_sample": False, "num_beams": 1, "repetition_penalty": 1.0,
          "temperature": None, "top_p": None, "top_k": None}


def sample_positions(length: int, count: int) -> np.ndarray:
    """Positions to keep full logits for: deterministic, evenly spaced, and
    always including the last, which is the one that decides the next token."""
    if length <= count:
        return np.arange(length, dtype=np.int32)
    positions = np.linspace(0, length - 1, count).round().astype(np.int32)
    # Spacing above one makes these strictly increasing; asserted rather than
    # de-duplicated, so a change to the spacing rule fails loudly.
    assert np.all(np.diff(positions) > 0), "sampled positions must be distinct"
    return positions


def smoke_ids(prompts: list[dict]) -> set[str]:
    """SMOKE_PROMPTS prompts, evenly spaced over the set, first and last included."""
    picks = np.linspace(0, len(prompts) - 1, min(SMOKE_PROMPTS, len(prompts))).round().astype(int)
    return {prompts[i]["id"] for i in picks}


def generate(model_dir: Path, prompts_path: Path, out_dir: Path, seed: int) -> None:
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)

    print(f"loading {model_dir.name} in float32 (not the checkpoint's bfloat16)")
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForCausalLM.from_pretrained(model_dir, dtype=torch.float32)
    model.eval()

    prompts = load_prompts(prompts_path)
    smoke = smoke_ids(prompts)
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "model": model_dir.name,
        "config_sha256_16": config_fingerprint((model_dir / "config.json").read_bytes()),
        "dtype": "float32",
        "checkpoint_dtype": json.loads((model_dir / "config.json").read_text())["torch_dtype"],
        # Recorded for reproducibility, though a forward pass under no_grad with
        # no sampling has no stochastic step for it to govern today.
        "seed": seed,
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "logit_samples": LOGIT_SAMPLES,
        "hidden_groups": sorted(HIDDEN_GROUPS),
        "smoke_tokens": SMOKE_TOKENS,
        "greedy": {k: v for k, v in GREEDY.items() if v is not None},
        "dense_prompts": list(DENSE_PROMPTS),
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
            # (layers + 1, len(positions), hidden); the first entry is the
            # embedding output.
            #
            # Kept at the sampled positions only, not at every one. The
            # diagnostic's job is to name the first layer that diverges, which
            # needs every *layer* at a few positions — not every position. At
            # every position the long prompts cost 691 MiB across both models,
            # against 51 MiB this way, and the question it answers is the same.
            stacked = torch.stack(out.hidden_states)[:, 0]
            arrays["hidden_states"] = stacked[:, positions].numpy().astype(np.float32)

        if prompt["id"] in smoke:
            with torch.no_grad():
                continued = model.generate(ids, attention_mask=torch.ones_like(ids),
                                           max_new_tokens=SMOKE_TOKENS, **GREEDY)
            # New tokens only, ending at an end-of-sequence token if one came
            # first; generate includes it, and so does the engine.
            arrays["generated"] = continued[0, ids.shape[1]:].numpy().astype(np.int32)

        path = out_dir / f"{prompt['id']}.npz"
        np.savez(path, **arrays)

        if prompt["id"] in DENSE_PROMPTS:
            np.savez(out_dir / f"{prompt['id']}.dense.npz", logits=logits.numpy().astype(np.float32))

        manifest["prompts"].append({
            "id": prompt["id"],
            "group": prompt["group"],
            "tokens": int(length),
            "has_hidden_states": "hidden_states" in arrays,
            "has_generated": "generated" in arrays,
            "has_dense_logits": prompt["id"] in DENSE_PROMPTS,
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
