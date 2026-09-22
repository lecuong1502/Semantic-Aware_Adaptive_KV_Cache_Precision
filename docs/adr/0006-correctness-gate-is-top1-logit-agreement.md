# Correctness is gated on top-1 logit agreement, not on matching generated text

Three layers of checking, with distinct jobs:

1. **Per-kernel unit tests** compare each hand-written kernel against a float64
   NumPy reference on random inputs: max relative error < 2e-3, mean < 2e-4.
   These need no model and no golden tensors, run in milliseconds, and are
   where bugs are actually caught.
2. **Per-layer comparison** against golden hidden states — cosine similarity
   > 0.999, reporting the first layer that diverges. This is diagnostic and
   does not block; its job is to answer *where* when layer 3 goes red.
3. **The merge gate** is logit agreement over a fixed set of 20 prompts: top-1
   agreement >= 99% of positions, and mean `KL(HF || ours)` < 1e-3.

Greedy generation matching HuggingFace for 64 tokens on at least 8 of 10
prompts is a smoke test. A red smoke test is investigated; it does not block a
merge.

Exact text match is rejected as a gate. The engine calls cuBLAS along a
different path with a different accumulation order than HuggingFace, so
bit-reproducibility is not available. A logit difference of 1e-4 at a
near-tie flips an argmax, and every subsequent token diverges from it — the
test then goes red across the whole suite while pointing at nothing. The
failure mode is weeks spent chasing a difference that is not a bug.

## Consequences

- The golden tensor generator must emit per-layer hidden states and final
  logits, not only generated text.
- "Top-1 agreement >= 99%" is the acceptance criterion every Milestone 0
  ticket inherits. Tickets state it explicitly rather than saying "matches the
  reference".
