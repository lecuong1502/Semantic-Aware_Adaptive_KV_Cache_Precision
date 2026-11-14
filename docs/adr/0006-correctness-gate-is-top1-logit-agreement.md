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

### Correction from #14: the sample errs both ways

The section above says the sample "errs high, never low". #14 showed that is
false. It changed how attention completes each key's bias, reading cos and sin
from a table rounded to fp32 as HuggingFace's are (ADR-0009, note from #14).
That moved adversarial-00's sampled positions closer to the reference more
than it moved the prompt as a whole:

| prompt | dense mean KL | sampled mean KL | sampled / dense |
|---|---:|---:|---:|
| long-03 | 6.2e-6 | 1.0e-4 | 16 |
| adversarial-00 | 1.10e-2 | 5.3e-3 | **0.48** |

Where the divergence concentrates at positions the sample misses, the sample
understates it by half. Neither direction is guaranteed. What the gate needs
is narrower: the sample must not *decide* it. Putting both prompts' dense means
in place of their samples moves the gate's KL from 4.4e-4 to 7.1e-4, still
under 1e-3, so the verdict is unchanged. `test_inference.py` now asserts exactly
that, instead of a one-sided bound.

A KL figure against this gate is therefore a sample mean that may be off by a
factor of two for a prompt like adversarial-00. The gate's margin should be
read with that in mind. At 7.1e-4 on the dense estimate, it is about 30%.

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

---

## Note from #15: chunked prefill agrees within a bound, not to the bit

#15 prefills a long prompt in chunks, so that the workspace is sized to a chunk
rather than to the prompt. Its ticket asked for output *identical* to
single-shot prefill. That is not available, for a reason the engine does not
control:

- **Attention is row-invariant.** A query's output does not depend on which
  other queries share its launch. The last 37 queries alone give the same
  bits as they do inside a launch of 300.
- **cuBLAS is not.** It chooses algorithm, tiling and split-K from the GEMM's
  shape, so one row of a projection can round differently depending on how
  many rows the GEMM has. Measured against a 1,090-row GEMM, the same rows
  computed in GEMMs of 1, 37, 128 and 512 rows were bit-identical for some
  sizes and projections and not for others. This happens between prefill and
  decode in any engine that uses cuBLAS, not only between chunks.

The owner chose to state equivalence as a measured bound rather than to pin
cuBLAS's algorithm. `tools/chunked_agreement.py` measures it and logs it as
`chunked-prefill-agreement` (entry `39d5fd7c`, at 14bc758). On the 23 prompts
other than adversarial-00, at chunks of 1, 37, 128 and 512, over every
position, the logits agree within 0.096 and KL within 7.5e-5. The argmax
changes at two positions at a chunk of 1 and two at 128, all of them near-ties
(single-shot's top two logits within 0.2). `test_chunked_prefill.py` allows
twice the logit difference and twice the KL, and allows an argmax to change
only at such a near-tie.

adversarial-00 is excluded, as ADR-0010 explains, and the same entry records
it apart. It amplifies these differences about a thousand times, to logit
differences of up to 9.1 and nine changed argmaxes at a chunk of 1. With the
default chunk of 512 it runs in one chunk, and is unaffected.

---

## Note from #204: a long launch splits its keys, and is gated as a new kernel

#204 made decode attention use the whole GPU. A launch with too few blocks
to fill the device splits each query's keys into runs, in the manner of
flash-decoding: each block leaves a partial m, l and unnormalised output, and
a second kernel folds them. A decode step's block also serves every query
head of a KV head's group. Splitting changes the order in which the output is
summed, so a split launch's logits differ from an unsplit one's in their last
bits.

**When a launch splits.** Only when its caller passes partials, which the
engine's workspace holds; only when it has fewer blocks than eight a
multiprocessor; and never into runs shorter than 256 keys. On the RTX 4050
that is a decode step from 512 positions, and a prefill chunk of a few tiles
past 512: a step under pressure, or a prompt's last, shorter chunk. A chunk
of 512 positions never splits, so a prompt whose length is a multiple of 512,
as perplexity's windows are, is computed exactly as before. How a launch
splits depends on the launch's shape and the device's multiprocessor count,
never on the layout its keys are read through: through pages a split launch
gives the bits it gives over contiguous keys, and asking for the masses
changes no bit of its output.

**The per-kernel gate is unchanged.** `test_split_attention.py` holds split
launches to float64 within 4 ulp max and 1 ulp mean, with the floor of the
amendment of #8, at decode and at prefill tiles, over sealed pages at every
tier, and with the running maximum in the first run and in the last. The
masses are held within 32 fp32 ulps of the float64 attention matrix's
per-page sums, the tolerance the unsplit kernel is held to.

**The merge gate, between builds.** The golden prompts are too short to
reach a split, so the gate's terms were taken between the build before #204
and the build after it: WikiText-2's test split, 8192 positions prefilled,
then 128 forced one by one as decode steps, at every tier and over a map
cycling the four, on both models (`decode-agreement` entry `45ba5d6f…`, at
ec1e1c7). Top-1 agreement is over the positions named, and KL is the mean
over 32 sampled positions, 8 of them before a forced page seals:

| | Before a forced page seals | Every forced position | Control: one build, chunks of 256 |
|---|---:|---:|---:|
| 0.5B FP16 | 100%, 1.0e-6 | 100%, 2.1e-6 | 100%, 1.6e-6 |
| 0.5B INT8 | 100%, 1.6e-6 | 100%, 5.4e-6 | 100%, 2.2e-5 |
| 0.5B INT4 | 100%, 7.3e-7 | 99.2%, 1.7e-4 | 99.2%, **3.3e-3** |
| 0.5B INT2 | 100%, 1.4e-6 | **98.4%, 4.7e-3** | **88.3%, 9.7e-2** |
| 0.5B mixed | 100%, 9.4e-7 | 100%, 3.4e-4 | **96.9%, 2.0e-2** |
| 1.5B FP16 | 100%, 1.6e-6 | 100%, 1.2e-6 | 100%, 1.1e-6 |
| 1.5B INT8 | 100%, 9.8e-7 | 100%, 3.2e-6 | 100%, 1.3e-5 |
| 1.5B INT4 | 100%, 6.9e-7 | 100%, 1.1e-4 | **98.4%, 1.7e-3** |
| 1.5B INT2 | 100%, 1.3e-6 | **98.4%, 2.8e-3** | **92.2%, 4.5e-2** |
| 1.5B mixed | 100%, 8.6e-7 | 100%, 1.2e-4 | **94.5%, 9.7e-3** |

- **Until a page of forced positions seals, the two builds agree at every
  tier as closely as at FP16,** KL near 1e-6. Those positions read the
  prefill's pages, which both builds compute alike, and their own open page
  at FP16: the difference is the kernel's alone, and it passes the gate
  everywhere.
- **Once a forced page seals at a quantised tier, the difference grows.** A
  key that moved in its last bit can round to another code, and at INT2
  the step between codes is a third of a channel's range. The jump comes at
  the first seal, forced position 32, in every run measured.
- **INT2 then fails the gate between builds, and that measures INT2.** The
  control moves the same logits by cuBLAS's rounding alone, prefilling in
  chunks of 256 rather than 512 (#15's note), and INT2 carries that noise
  twenty times further than #204's: top-1 88% to 92%. INT4 and the mixed map
  fail the control as well and pass between builds. So the gate between
  builds is passed wherever the noise it measures is the kernel's; where a
  tier amplifies any last-bit noise past the gate, a kernel that sums in a
  new order cannot be held to it, and is held instead to the positions
  before a seal.

The gate against HuggingFace, at FP16 over the golden prompts, passes on
both models at ec1e1c7: top-1 99.74% and KL 1.9e-4 on 0.5B (`gate` entry
`605dd563…`), 99.97% and 3.5e-6 on 1.5B (`8342d7f1…`).

**Speed, and why INT4 and INT2 are still slower than FP16.** One launch at
32K positions of Qwen2.5-1.5B's shapes took 62 to 70 ms at every tier before
#204 (`attention-speed` entries `d2aa5d29…` to `3e028bfa…`, at 51a33ba) and
takes 0.57 ms at FP16, 0.68 at INT8, 0.75 at INT4 and 0.73 at INT2 after it
(`1427c496…` to `690f2e84…`, at 45e6d41, the code of 3941be1): about a
hundredth. A prefill tile of 16 queries there went from 62 ms to 6.2 ms.
Dequantisation no longer costs what it did, a block reading each page once
for its group's six heads, but INT4 and INT2 still take 15% to 30% longer a
launch than FP16; measured interleaved, 14% and 10%. The kernel is still
bound by the latency of its loads and by issuing instructions, not by the
device's bandwidth: 32 MB a layer in 0.57 ms is a third of it. A quantised
element costs three loads, its code and its channel's or token's scale and
zero, where an FP16 one costs one, and that the fewer bytes do not repay.
Over a decode step, 28 layers of it are about 5 ms against some 60, the
weights' reading most of the rest.

**In the engine** (`tools/step_cost.py`, Qwen2.5-1.5B, random token ids; a
decode step is one position forced as decode runs it, the median of 16; at
"32K" the conversation is prefilled to 32,352 positions, leaving the window
room for the steps timed):

| | 16K, after | 32K, before | 32K, after |
|---|---:|---:|---:|
| Decode, FP16 | 40.1 tokens/s | 0.58 | 29.5 |
| Decode, INT8 | 37.0 | 0.76 | 26.5 |
| Decode, INT4 | 36.4 | 0.44 | not yet measured |
| Decode, INT2 | 36.9 | 0.30 | not yet measured |
| A prefill step's fixed part, FP16 | 15 ms | about 1.7 s a step of one tile | 12 ms |
| A prefill step's part a position, FP16 | 5.8 ms | | 11.4 ms |

The figures before are `decode-throughput` entries at 32K (1034e2c and
c9cfe2a, through `generate`), and ADR-0016's amendment of #135 for a step of
one tile from 26K positions. After: `decode-step` and `prefill-step-cost`
entries at f7ef85d (16K) and 74b475e (32K). At 16K, INT4 and INT2 decode 9%
and 8% slower than FP16, and INT8 8%: the kernel's gap above, diluted by the
weights.

A step's fixed part, most of a step before, is now a few hundredths of
`pressure_chunk_seconds`. The chunk sizing of ADR-0016's amendment of #135
therefore no longer meets the case where the fixed part alone overruns the
target, and a chunk under pressure holds about as many positions as 0.8 s
of the part per position buys: some 70 at 32K. Prefill's part per position
is now what bounds reaction during prefill, which #213 addresses.

**Correction to #15's note.** That note says attention is row-invariant: a
query's output does not depend on which other queries share its launch. That
now holds only for launches that do not split. How a launch splits depends
on its number of queries, so a query decoded alone at a long context and the
same query inside a chunk of 20 sum in different orders. Chunked prefill's
bounds (`test_chunked_prefill.py`) still hold.

**What can no longer be repeated to the bit.** RQ3's controlled plans repeat
exactly on one commit. Every logged result that decodes, or forces a position
a step, beyond 511 positions, or prefills under pressure in short chunks at
long contexts, was made with the unsplit kernel and will not repeat to the
bit on this one:

- `rq3-needle`, `rq3-anchor`, `rq3-continuation`, `rq3-recovery`,
  `rq3-end-to-end`, `rq3-eviction-summary` and their summaries;
- `survival` and `survival-summary`, and `page-damage`;
- `chunked-prefill-agreement`, at a chunk of 1;
- `generation-sample`, where a prompt and its answer pass 511 positions;
- every timing: `decode-throughput`, `prefill-throughput`,
  `monitor-overhead`, `monitor-scorer-overhead`, `requantisation-latency`.

`perplexity`, `layer-sensitivity` and `kv-quantisation-roundtrip` do not
use a split launch and repeat as they did. The `gate` does where a prompt's
last chunk splits, and was logged again here. #216 re-runs RQ3 and the
survival matrix on this engine.
