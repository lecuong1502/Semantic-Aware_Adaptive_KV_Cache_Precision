# Qwen2.5 as the model family: 0.5B for development, 1.5B for experiments

`Qwen2.5-0.5B-Instruct` is the development target and `Qwen2.5-1.5B-Instruct`
is the experimental target. Same family, same plain RoPE (theta 1e6), same GQA
shape (2 KV heads), same 32K native context, Apache-2.0 and ungated — so one
parameterised kernel serves both and only config constants differ. Development
iterates against the 0.5B model, where golden-tensor comparison is fast and
VRAM is never the binding constraint; RQ2 and RQ3 are measured on the 1.5B
model at 32K context.

The driving constraint is arithmetic, not convenience. On this machine (RTX
4050 Laptop, 6141 MiB) the entire KV cache of Qwen2.5-0.5B at 8K context is
96 MiB, and requantising all of it to INT4 reclaims 72 MiB — around 1.2% of the
card, against realistic contention spikes of 200-600 MiB from a browser or a
screen share. At that scale the mechanism cannot avoid a single OOM and RQ2 is
arithmetically dead. Qwen2.5-1.5B at 32K context puts 896 MiB in the cache and
makes 672 MiB reclaimable, which clears a realistic spike. **The mechanism is
only meaningful where reclaimable bytes exceed the size of a real contention
spike, and that is a statement about context length and KV footprint fraction,
not about model quality.**

## Considered Options

- **Qwen2.5-1.5B throughout.** Rejected only on iteration speed: reloading
  2.88 GiB of weights for every kernel test, and debugging a hand-written
  attention kernel with 2.4 GiB of headroom, invites OOM failures that have
  nothing to do with the research.
- **Llama-3.2-1B.** Has the fattest KV cache of the candidates (32 KiB/token,
  8 KV heads) and therefore the most reclaimable memory. Rejected because the
  repository is gated, and because its `llama3` RoPE scaling (factor 32, with
  low/high frequency factors) adds a genuinely error-prone surface to a
  hand-written RoPE kernel for no research benefit.
- **TinyLlama-1.1B.** Rejected outright: 2048 native context cannot support the
  needle-in-a-haystack evaluation in §6.

## Consequences

- Kernels take `head_dim` and `num_kv_heads` as parameters from the start.
  0.5B uses `head_dim=64`, 1.5B uses `head_dim=128`; nothing may be hardcoded.
- Golden tensors are generated per model — two sets, regenerated rarely.
- **A reviewer will observe that aggressive GQA is precisely the industry's
  answer to KV cache size, and that this method matters least on the models
  most likely to be deployed.** The defence is to report KV footprint fraction
  alongside every result and to evaluate on at least two models with different
  GQA ratios, positioning the benefit as scaling with that fraction rather than
  as a universal claim. This obligation is inherited by the evaluation design,
  not optional.
- All model configs above were taken from prior knowledge and must be verified
  against each model's `config.json` at download time before any kernel is
  written against them. **This has now been done — see the amendment below.**

---

## Amendment: constants verified, and two assumptions that were never written down

Every constant above was checked against each model's `config.json` on
2026-09-22 (#5). Both cards are committed under `src/microinfer/model_cards/`,
with the expectation held separately in `models.py` so that the copy cannot
check itself.

**Nothing was wrong.** Layer counts, hidden sizes, head counts, KV head counts,
derived head dimensions, RoPE theta, tied embeddings, 32K context and the
absence of rope scaling all match what was assumed.

Two things were *unstated*, which is a worse failure mode than a wrong constant
because nothing existed to check.

### The checkpoints are bfloat16, not float16

Both models ship `torch_dtype: bfloat16`. Every kernel, ADR-0005 and ADR-0006
assume fp16 throughout, so **every weight load crosses a format boundary**.

The crossing is safe in one direction and lossy in the other, and not in the way
intuition suggests. bf16 is float32 with the low sixteen mantissa bits removed:
same 8-bit exponent, 7 bits of mantissa. fp16 has 5 exponent bits and 10 of
mantissa. So converting bf16 to fp16 *gains* precision and *loses range* —
bf16 reaches past 3e38 where fp16 stops at 65504. A weight beyond that line
becomes an infinity in silence.

The engine therefore range-checks every tensor on load and refuses rather than
converting. For these two checkpoints no tensor comes close, but that is a fact
about these weights, not a guarantee about bf16 checkpoints.

**The consequence reaches past this ticket.** Golden reference tensors (#6) must
be generated in fp32, not in the checkpoint's native bf16. bf16's relative ulp is
2^-8, eight times coarser than fp16's 2^-11, so a bf16 reference would carry
more error than the implementation being measured against it — and would exceed
ADR-0006's max bound of 4 fp16 ulps on its own. A reference must be more
accurate than the thing it judges.

### The card has 6141 MiB; CUDA offers 5762 MiB

`cudaMemGetInfo` on this machine reports 5762 MiB total and 5392 MiB free with
nothing loaded. The 6141 MiB figure from `nvidia-smi`, which the arithmetic
above was computed against, is the physical size; roughly 380 MiB is already
held by the driver and the desktop before the engine starts.

This does not change any conclusion — the percentages shift slightly in the
project's favour, since the same reclaimable bytes are a larger share of a
smaller budget — but the **absolute** headroom is smaller than stated. Reported
figures should say which denominator they use.
