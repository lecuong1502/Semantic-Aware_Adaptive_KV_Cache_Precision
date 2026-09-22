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
  written against them.
