# Correctness is gated on top-1 logit agreement, not on matching generated text

Three layers of checking, with distinct jobs:

1. **Per-kernel unit tests** compare each hand-written kernel against a float64
   NumPy reference on random inputs, within the tolerances in the amendment
   below. These need no model and no golden tensors, run in milliseconds, and
   are where bugs are actually caught.
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

---

## Amendment: per-kernel tolerances are stated in ulps of the output format

The original per-kernel tolerances were absolute numbers — max relative error
< 2e-3, mean < 2e-4 — and the mean bound was wrong. Building the first kernel
(#3) showed why.

The largest relative ulp of fp16 is `2^-11 = 4.883e-4`. Measured against that:

| | value | in ulps |
|---|---:|---:|
| Original max bound | 2e-3 | 4.10 |
| Original mean bound | 2e-4 | **0.41** |
| Measured mean, one fp16 rounding | 1.76e-4 | 0.36 |
| Measured mean, two fp16 roundings | 2.94e-4 | 0.60 |

The max bound was already calibrated in ulps without saying so: 4.10 is four
ulps. The mean bound was not. At 0.41 ulp it sat **14% above the mean error of a
single fp16 rounding**, which is the floor any correct implementation produces
when its output dtype is fp16. It therefore admitted a value rounded once and
rejected a value rounded twice — and a value is rounded twice whenever a caller
passes fp32 into a kernel whose domain is fp16, which is exactly what Seam B's
NumPy surface invites.

A bound that close to a format's noise floor measures the format, not the
implementation. Attention would have failed it for reasons unrelated to
correctness.

**The tolerances are therefore expressed in ulps of the kernel's output format:**

- **max relative error < 4 ulp**
- **mean relative error < 1 ulp**

For an fp16 output that is max < 1.953e-3 and mean < 4.883e-4. The max bound is
unchanged in substance; the mean bound moves from 0.41 to 1 ulp, which leaves
room for a value to pass through several roundings while still rejecting any
kernel whose average error exceeds the resolution of the format it writes into.

Stating the bound in ulps rather than in absolute numbers also means a future
kernel with an fp32 output is held to a proportionally tighter standard
automatically, rather than inheriting a tolerance calibrated for fp16.

### What this does not change

Nothing above touches the merge gate. That remains top-1 logit agreement >= 99%
with mean `KL(HF || ours)` < 1e-3, and generated text remains a smoke test.

---

## Amendment: the KL term is an estimate over a stated sample

This ADR set the gate as "top-1 agreement >= 99% of positions, and mean
`KL(HF || ours)` < 1e-3", and #6 had to build the reference that gate is
computed against. The two terms turn out to cost very different amounts, and
the ADR did not say so.

**Top-1 agreement is exact.** The reference stores its own argmax at every
position, four bytes each. Nothing is approximated.

**Mean KL cannot be, at any reasonable price.** KL needs whole distributions,
and a distribution over Qwen2.5's vocabulary is 151,936 floats — 608 KiB per
position, so roughly 3.4 GiB for one model's 24 prompts. The reference therefore
keeps full logits at **16 evenly spaced positions per prompt, always including
the last**, and the gate's KL term is the mean over those.

Storing only the most probable tokens per position was tried first, because it
would have made every position affordable. It does not work: measured on the
real references, the probability mass outside the top 2048 tokens reaches
**0.35** at the least certain positions. The model is genuinely uncertain early
in a prompt, and a truncated distribution would corrupt KL far past the bound
being tested. Recorded because the idea is attractive enough to be proposed
again.

### The sample size is not yet justified, and that is deliberate

Sixteen is a storage budget, not a measurement. The quantity that would justify
it — how much the sample mean varies around the true mean — is a property of the
*difference* between the reference and the engine, and **the engine does not
produce logits until #12**. It cannot be measured today.

**#12 therefore owes this ADR a validation**: with the engine producing logits,
compare the sampled mean against a mean over a much denser set of positions for
at least one prompt, and either confirm 16 or change it. Until that is done, a
KL figure quoted against this gate should say it is a sample mean over 16
positions per prompt.

### Consequences

- A gate report states both terms separately. The top-1 figure is exact; the KL
  figure carries its sample size.
- `LOGIT_SAMPLES` in the generator is the single place the sample size is set,
  and changing it invalidates every stored reference — the manifest records it
  so a mismatch is visible.
