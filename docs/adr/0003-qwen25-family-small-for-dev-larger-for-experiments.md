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

The table at the top of this ADR is computed against 6141 MiB. Re-run against
the memory CUDA actually offers, every figure moves in the project's favour,
because the same reclaimable bytes are a larger share of a smaller budget:

| model | context | KV at FP16 | reclaimed by INT4 | of 6141 MiB | of 5762 MiB |
|---|---:|---:|---:|---:|---:|
| Qwen2.5-0.5B | 8K | 96 MiB | 72 MiB | 1.2% | 1.2% |
| Qwen2.5-0.5B | 32K | 384 MiB | 288 MiB | 4.7% | 5.0% |
| Qwen2.5-1.5B | 8K | 224 MiB | 168 MiB | 2.7% | 2.9% |
| Qwen2.5-1.5B | 32K | 896 MiB | 672 MiB | 10.9% | **11.7%** |

No conclusion changes, and the argument for the 1.5B at 32K gets slightly
stronger. What does change is the **absolute** headroom, which is 379 MiB
smaller than this ADR assumed. Every reported figure should say which
denominator it uses; the engine's own footprint report uses CUDA's.

---

## Amendment (#209): Llama-3.2-1B-Instruct as a second family

The Consequences above oblige the evaluation to a second model with a
different GQA ratio. The owner chose Llama-3.2-1B-Instruct (2026-11-14):
Llama-3.2-3B does not fit 6 GiB at FP16 weights, Gemma-2-2B's sliding
window and soft-capping are more engine work, and Phi-3.5-mini is larger
and slower here. The objection recorded above, llama3 RoPE scaling as an
error-prone surface, is met by testing it as the plain RoPE was: against a
float64 port of HuggingFace's `_compute_llama3_parameters`.

**What differs from Qwen2.5, and where the engine meets it:**

| | Qwen2.5-1.5B | Llama-3.2-1B | where |
|---|---|---|---|
| layers, hidden, heads / KV heads, head_dim | 28, 1536, 12 / 2, 128 | 16, 2048, 32 / 8, 64 | parameters already |
| KV per token, at 32K | 28 KiB, 896 MiB | 32 KiB, 1 GiB | `kv_bytes_per_token` |
| RoPE | theta 1e6, plain | theta 5e5, llama3 (factor 32, low 1, high 4, original 8192) | `rope_scaling.h` |
| q, k, v bias | yes (ADR-0009) | none (`attention_bias` false) | `ModelConfig.attention_bias` |
| vocabulary, tied | 151,936, tied | 128,256, tied | parameters already |
| context | 32,768 | 131,072 | parameters already |
| tokenizer | adds nothing | adds `<|begin_of_text|>` | `chat.encode` |
| chat template | ChatML, `<|im_end|>` | Llama 3 headers, `<|eot_id|>` | `chat.LLAMA3` |

- **RoPE.** `RopeScaling` is a plain struct passed by value into the kernels
  that form an angle, the rope kernel and the table attention reads. The
  frequency is scaled in fp64, before the angle, which stays fp64.
- **No bias.** A layer's biases are None; attention already takes a key
  with no bias as whole, so ADR-0009's path is simply not entered.
- **Tokenisation.** RQ3's turns are built from pieces tokenized apart, and
  Llama's tokenizer would put `<|begin_of_text|>` before each. Pieces are
  encoded with no special token added; the template's opening writes it
  once. Qwen2.5's tokenizer adds none, so its needle set and RULER sets are
  unchanged, byte for byte (checked by their sha256 before and after).
- **The chat template** is kept as pieces in `microinfer.chat`, since RQ3
  assembles turns from them. Each family's is checked against HuggingFace's
  own rendering of a three-message conversation, which `tools/gen_golden.py`
  records in the reference's manifest. Qwen2.5's matched when it was first
  checked. Llama's system message is given the template's fallback date,
  26 Jul 2024, so a sample does not change with the day it is built.
- **The gate** is ADR-0006's, held as Qwen2.5's is: FP16 passes it over the
  whole prompt set, and each narrower tier costs more than the one above
  (`tests/test_llama.py`).

- **Trimming.** Llama-3.2's template renders a message's content trimmed
  (`| trim`); Qwen2.5's renders it as given. A needle set's document opens
  with a space, a haystack line's, so a Llama turn's first piece is encoded
  again without it (`chat.content_opening`). The probe's first message
  opens with a space and ends with a newline, so the check above covers it.
- **Continuation passages.** They are cut from the split encoded with no
  special token added. For a family with a begin-of-text, each passage
  opens with it and takes one fewer of the split's tokens, so every
  passage, not only the first, opens as its family's contexts do.
- **The window.** Llama-3.2's own is 131,072 positions. Every experiment
  runs within 32K, and a cache is sized to the engine's window, at most
  32,768 (`ModelConfig.context_window`): at 131,072 a contiguous cache alone
  would be 4 GiB, which 6 GiB cannot hold beside the weights.

### As measured (#209)

The constants in the table were from prior knowledge, as this ADR's first
ones were. Each matched the checkpoint's own `config.json`, downloaded from
`meta-llama/Llama-3.2-1B-Instruct` on 2026-11-23; the card is committed
beside Qwen2.5's.

On ADR-0006's gate, against a float32 HuggingFace reference over the 24
prompts (5,689 positions):

| | top-1 agreement | mean KL(HF ‖ ours) |
|---|---:|---:|
| FP16, every prompt | 99.98% | 9.8e-7 |
| FP16, the 8-prompt tier subset | 100% | 7.9e-7 |
| INT8 | 100% | 5.3e-6 |
| INT4 | 99.72% | 1.2e-3 |
| INT2 | 98.28% | 7.4e-2 |

FP16 passes the gate. Each narrower tier costs more than the one above,
KL rising strictly and top-1 never rising, as Qwen2.5's tiers do.
