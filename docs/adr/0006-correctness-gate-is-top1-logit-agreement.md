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

### Validated by #12: 16 is kept, and the sample errs high

#12 had the reference keep logits at every position for two prompts, and
compared the 16-position sample mean of KL against the dense mean with the
engine producing logits:

| prompt | positions | dense mean KL | sampled mean KL | position 0's share of the total | sampled / dense, position 0 excluded |
|---|---:|---:|---:|---:|---:|
| long-03 | 441 | 6.3e-6 | 1.0e-4 | 56% | 1.5 |
| adversarial-00 | 207 | 1.31e-2 | 1.28e-2 | 0.1% | 1.02 |

**The sample errs high, never low.** It always includes position 0, and on an
ordinary prompt the first position carries most of the KL there is. At a
weight of 1/16 rather than 1/441, it makes long-03's sample mean sixteen times
its dense one. Where the engine and the reference really differ, as on
adversarial-00, the difference is spread along the sequence, and the sample
lands within 3% of the dense mean.

So a KL figure against this gate is an **over**-estimate for ordinary prompts:
conservative, and in the direction that fails a good engine rather than passing
a bad one. Sixteen positions are kept, and so is position 0. Removing it would
make the estimate fairer, but it would change every stored reference to buy an
estimate that can only become more lenient. `test_inference.py` asserts both
halves: the sample never understates the dense mean, and without position 0 it
agrees with it to within a factor of two.

### Consequences

- A gate report states both terms separately. The top-1 figure is exact; the KL
  figure carries its sample size.
- `LOGIT_SAMPLES` in the generator is the single place the sample size is set,
  and changing it invalidates every stored reference — the manifest records it
  so a mismatch is visible.

---

## Amendment: a kernel whose output is a sum is measured against its terms where they cancel

The ulp tolerances above divide each error by the reference value, floored at
fp16's smallest normal. That is right for RMSNorm and SwiGLU, whose outputs are
products. It is wrong for a kernel whose output is a *sum* — a projection's dot
product, or RoPE's `x1*cos - x2*sin` — and #7 found out why on its first run.

A sum's rounding error is bounded by the magnitude of its terms, not of its
result. Where the terms cancel, the result is small and its error is not, so the
relative error grows without limit however correct the kernel is. Measured on
the cuBLAS wrapper with fp16-exact inputs:

| | value |
|---|---:|
| Worst error, outputs with \|y\| > 0.05 | 1.00 ulp — the fp16 store alone |
| Worst error, all outputs | 60 ulp, at y = -8.6e-6 |
| Outputs over 4 ulp | 74 of 311,296, all with \|y\| < 1.1e-3 |

RoPE fed arbitrary fp32 showed the same thing from a different source: rounding
x1 and x2 to fp16 on the way in cost 1084 ulp at an output of -9.7e-5 whose terms
summed to 1.64. That error belongs to the caller's downcast, not to any kernel.

**Such a kernel's test therefore floors the denominator per output element**,
using `terms` — the sum of the absolute values of what that element adds. There
are two floors, and each admits exactly one error source (`tests/ulp_gate.py`):

- **fp16-exact input — the kernel's gate.** `sqrt(n) * u32 * terms / MAX_REL`
  for a sum of `n` terms: the magnitude at which fp32 accumulation's
  probabilistic error bound (Higham & Mary, 2019) uses the whole 4-ulp max
  budget, and no more.
- **Arbitrary fp32 input — the Seam B check.** `terms` itself: rounding each
  operand to fp16 moves the result by up to half an fp16 ulp of its terms, and
  no kernel can avoid that.

### Why the first floor is not simply `terms`

A dot product's terms outweigh its result by about `sqrt(n)`, so measuring
against them forgives errors that would matter. Simulating the failure this gate
most needs to catch — partial sums held in fp16, which is what cuBLAS's
reduced-precision split-K reduction does — at the model's widest reduction
(n = 4864):

| floor | max | mean | verdict |
|---|---:|---:|---|
| `terms` | 3.0 ulp | 0.33 ulp | **passes** |
| `sqrt(n) * u32 * terms / MAX_REL` | 370 ulp | 30.5 ulp | fails |

### How much of the output the floor judges

A floor replaces |y| for every output smaller than it, so its size decides how
much of a test is still a relative-error test. The review of #7 measured this
against a first version of the floor that spent one ulp, not four, on
accumulation — and found it judging **51%** of `down_proj`'s outputs at n = 8960
against the floor. That was measuring the floor.

The floor is therefore calibrated against what fp32 accumulation actually does.
On the model's projections the observed error is at most **0.19** of the
probabilistic bound, and the bound is allowed the full max budget:

| projection | n | outputs judged against the floor | worst error |
|---|---:|---:|---:|
| q_proj, 0.5B | 896 | 1.4% | 1.05 ulp |
| down_proj, 0.5B | 4864 | 7.6% | 1.00 ulp |
| down_proj, 1.5B | 8960 | 13.7% | 1.03 ulp |

Even at the bound itself — five times the error observed — an output at the
floor would sit exactly at 4 ulp, not past it.

Both simulations live in the test suite as tests *of the gate* — fp16
accumulation in `test_linear.py`, an fp32 RoPE angle in `test_rope.py` — so a
future loosening of the floor that stops catching either shows up as a failure.

### Attention's floor admits the scores' rounding as well (#8)

Attention's output is a sum of value rows weighted by softmax, and those
weights are not exact. Each score is itself an fp32 dot product over
`head_dim`, and its rounding becomes a relative error in every weight. The
accumulation floor does not see this. With that floor alone the kernel read
5–7 ulp at head_dim 128, on outputs whose weights were right to fp32 precision.
So attention's bound has a second term:

    (sqrt(n) + sqrt(d) * max_j sum_d |q_d k_jd| / sqrt(d)) * u32 * terms

The term is real rather than fitted. A simulation in which *only* the scores
are fp32, with everything else exact, already reads 5.4 ulp without it.

It is also not generous. The first version counted the term twice, on the
argument that each weight is divided by a denominator carrying the same error.
The review of #8 measured that version at about ten times the error the
scores actually cause, so the factor went. At one the term has the same slack
as the accumulation term, about five times:

| case | outputs judged against the floor | worst error |
|---|---:|---:|
| prefill, head_dim 64 | 0.8% | 1.04 ulp |
| prefill, head_dim 128 | 1.4% | 1.03 ulp |
| rising maximum, head_dim 64 / 128 | 0.06% / 0.37% | 1.04 ulp |

The gate still rejects the next optimisation a kernel would reach for, holding
the softmax weights in fp16 for the P·V product, at 83 ulp. That trade may be
worth making later, but it would go through this ADR, not past it.

A score offset of about 1000, which the stability test uses, is a different
matter. An fp32 score that large has an ulp of 6e-5. The floor is honest about
that, so it judges most of that test's outputs against their terms, and the
test is a stability test only. The precision claim at large scores belongs to
the test in which the maximum rises tile by tile.

### RoPE is closer to float64 than HuggingFace is

The RoPE kernel forms its angle in fp64. HuggingFace forms it in fp32, which is
18 ulp out at position 2049 and 60 ulp out at 32767 at the fastest frequency.
The kernel is judged against float64 here, so at long contexts it will
disagree with the golden reference **because the reference is the less
accurate of the two**. The golden prompts are short, so the difference does not
reach the gate today. **#12 should expect it** before treating a late-position
logit mismatch as an engine bug.

### What this does not change

The tolerances stay at 4 ulp max and 1 ulp mean. Product-shaped kernels pass no
floor and are measured exactly as before. The merge gate is untouched.
