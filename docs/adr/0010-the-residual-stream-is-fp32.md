# The residual stream is fp32

Every activation the engine keeps is fp16, except the residual stream, which is
fp32 from the embedding to the final norm. The embedding is widened into it
exactly. RMSNorm reads it directly. The attention and MLP output projections
add into it inside a single GEMM, with an fp32 output and beta = 1, so neither
is rounded to fp16 on the way in. The KV cache, and every other activation,
stay fp16.

## Why

KL against the reference falls on **every one of the 24 prompts**. On the 23
that behave ordinarily, the mean of their KL means falls **22 times**, from
2.2e-4 to 9.9e-6. The cost is one buffer, the residual, at four bytes an
element instead of two. The forward pass is no slower: 0.69 s against 0.71 s
for 1,090 tokens. Benchmark log entries 10 (before, at `4fc55ce`) and 11
(after, at `efe874d`) hold the gate on both sides.

|  | fp16 residual | fp32 residual |
|---|---:|---:|
| gate mean KL, 334 sampled positions | 4.4e-4 | 1.9e-4 |
| gate top-1, 5,761 positions | 99.83% | 99.74% |
| the other 23 prompts: top-1 | 5,551 / 5,554 | 5,550 / 5,554 |
| the other 23 prompts: mean of KL means | 2.2e-4 | 9.9e-6 |
| adversarial-00: top-1 | 200 / 207 | 196 / 207 |
| adversarial-00: dense mean KL | 1.10e-2 | 1.61e-2 |

Top-1 moves by a single position, either way, on three near-ties among the 23.
It falls on adversarial-00, and the next section is about why that prompt
does not count against the decision.

## adversarial-00 is ill-conditioned, and that is its whole story

#36 set out to find why adversarial-00, 207 tokens of repeated "the", stays
furthest from the reference. The answer is that the prompt amplifies any
small error about a thousand times. No single rounding causes the divergence,
so no single fix removes it.

**The evidence, in order.**

1. HuggingFace's hidden states were dumped at every position, not only the 16
   the reference keeps. 82% of this prompt's KL sits at about 10 positions
   the sample misses, and the lowest cosine similarity over all positions is
   0.59, not the 0.994 the sampled positions show.
2. A float64 model, given HuggingFace's exact input at layer 2 or layer 3,
   reproduces HuggingFace's output exactly, with every one of the engine's
   fp16 roundings applied. Given the engine's input, off by 0.12%, it
   reproduces the engine's divergence. The layers are not wrong. They amplify
   what they are given.
3. With the residual in fp16, the error they amplify is made in layer 0, by
   rounding `o_proj`'s output to fp16 before adding it in. This decision
   removes that error, but the divergence does not go. With the residual in
   fp32, no single remaining rounding dominates. Keeping any one of q, v,
   the attention output or the MLP's intermediates exact moves the result by
   about 1e-3, in different directions. Keeping `k_proj` exact makes it
   worse.
4. The conditioning itself, measured by `tools/conditioning.py` (log entries
   12 and 13). The model runs in float64, and the embedding is perturbed by
   random relative noise the size of one ulp:

| prompt | noise | relative error in the embedding | worst relative error over the layers | gain |
|---|---|---:|---:|---:|
| adversarial-00 | fp16 ulp, 2^-11 | 5.4e-4 | **0.56** | **1,041** |
| adversarial-00 | fp32 ulp, 2^-24 | 6.6e-8 | 7.5e-5 | 1,130 |
| medium-01 | fp16 ulp, 2^-11 | 5.3e-4 | 4.8e-4 | 0.91 |
| medium-01 | fp32 ulp, 2^-24 | 6.5e-8 | 5.9e-8 | 0.91 |

An ordinary prompt carries an error forward at the size it was made.
adversarial-00 multiplies it by a thousand. fp32 noise stays at 7.5e-5
afterwards, which is why HuggingFace's fp32 reference agrees with float64 at
every position and state, to a cosine of 1.0000. fp16 noise becomes an error
of order one. An engine that rounds activations to fp16 anywhere cannot follow
the reference on this prompt. The only way would be to keep every activation
near fp32, and that is not what an FP16 engine is.

**What #12 got wrong.** #12 tested an fp32 residual and rejected it, because
it changed nothing on adversarial-00 at the sampled positions. That test kept
the residual itself in fp32 but still rounded each projection to fp16 before
adding it, and it was judged on the one prompt that cannot improve. The other
23 prompts would have shown the gain.

## Considered Options

- **Keep the residual in fp16.** Rejected: it costs the 22-times KL
  improvement on ordinary prompts, and saves one buffer's worth of memory.
- **Keep more activations in fp32, to follow adversarial-00.** Rejected. The
  gain above means every activation would have to be near fp32, which makes
  the engine an fp32 engine. The question the thesis asks is about the
  precision of the KV cache, and an fp32 forward pass would change the
  baseline that question is posed against.
- **Remove adversarial-00 from the gate.** Rejected, as in ADR-0009. It is in
  the set to find exactly this kind of behaviour, and it found it.

## Consequences

- `device.embed_f32`, `rmsnorm_f32` and `linear_accumulate`, over a
  `FloatTensor`, are what the forward pass runs. The fp16 `rmsnorm`, `embed`
  and `add` stay, tested, for everything else that uses them.
- A hidden state captured for the diagnostic is now the fp32 residual itself,
  not an fp16 rounding of it.
- **adversarial-00 is a conditioning test, not an accuracy test, and the
  gate's KL should be read that way.** It carries about 95% of the gate's
  sampled KL. A change that moves only this prompt has moved noise that a
  thousandfold amplifier put there. A change that moves the other 23 has
  moved the engine.
- ADR-0006's correction still holds: the KL sample can misstate this prompt
  by a factor of two either way. The verdict stays the same with its dense
  mean substituted.
- `tools/conditioning.py` measures any prompt's gain. A new prompt with a large
  gain should be expected to diverge, and should not be treated as an engine
  bug before it is measured.
