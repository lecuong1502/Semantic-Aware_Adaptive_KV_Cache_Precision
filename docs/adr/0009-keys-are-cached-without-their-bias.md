# Keys are cached without their projection's bias

The KV cache stores each key as `RoPE(x · W_k)`, **without** the key
projection's bias. Attention completes the key as it loads it, adding
`RoPE(b_k, j)` for the key at row `j` in fp32. The stored part is what the
cache's precision tier rounds. The bias is a per-layer constant held once,
never cached and never quantised.

## Why

Qwen2 puts a bias on the key projection, and in Qwen2.5 it is large. Measured
on the 0.5B model's first layers:

| layer | largest \|bias\| | largest \|x · W_k\| | fp16 ulp at the largest \|k\| | spread of a channel across positions (median std) |
|---:|---:|---:|---:|---:|
| 0 | 130.0 | 4.9 | 0.125 | 0.084 |
| 1 | 147.0 | 5.8 | 0.125 | 0.269 |
| 2 | 58.8 | 11.5 | 0.031 | 0.250 |

A key that carries its bias is rounded at the bias's magnitude. At 130–147,
fp16's ulp is 0.125, which is as large as what distinguishes one position's key
from another's. The scores those keys feed reach 1,396 in layer 0, so the
softmax magnifies what is lost.

#12 found this through the gate. With whole keys in fp16, one prompt of the
24, `adversarial-00` (207 tokens of repeated "the"), agreed with the reference
on only 73% of positions. That single prompt put the set at 98.98% top-1
agreement against the 99% bar, and at a mean KL of 9.1e-3 against 1e-3. The
per-layer diagnostic named layer 2 as the first to diverge. A float64 NumPy
model of the first four layers then isolated the cause by rounding one
quantity at a time to fp16:

| quantity rounded to fp16 | lowest cosine similarity at hidden state 4 |
|---|---:|
| nothing (float64) | 1.00000 |
| all weights | 1.00000 |
| the residual stream | 0.99925 |
| the attention output projection | 0.99793 |
| **the cached key, whole** | **0.94000** |
| the key projection's output, before RoPE | 0.92845 |
| the cached key at **bf16**, the checkpoint's own dtype | 0.11115 |
| the cached key **without its bias**, *with every other activation also in fp16* | **0.99912** |

The residual stream was the first suspect, because the same prompt drives it to
about 1,700. Keeping it in fp32 changed nothing that matters: 0.846 against
0.853 at hidden state 8, so it is not the cause.

With keys cached without their bias, the engine passes the gate: **99.81%
top-1 agreement over 5,761 positions, and a mean KL of 8.0e-4 over 334 sampled
positions.** `adversarial-00` rises from 73% to 96% agreement.

**This does not close the case of `adversarial-00`.** It is still the prompt
furthest from the reference. Its own mean KL is 1.28e-2, about 6.1e-4 of the
set's 8.0e-4, which is why the margin under 1e-3 is only about 20%. Its lowest
cosine similarity is 0.981, and the diagnostic still names layer 3's output as
the first below 0.999. The float64 model puts the bias-free engine at 0.9991
there, so something else is lost as well, and it has not been isolated. The
bisection above points at the next candidates: the attention output
projection (0.99793) and the query, which carries a bias of its own. A change
that moves this prompt should be expected to move the gate.

## Considered Options

- **Cache whole keys in fp16, as before.** Rejected: it fails the gate, for
  the reason above.
- **Cache keys in fp32.** It would pass, but it doubles the key half of the
  cache. The FP16 tier would then not be FP16, and every byte figure that
  ADR-0003 and ADR-0008 argue from would be wrong for the top tier.
- **Change the gate, for example by setting adversarial prompts aside.**
  Rejected. The adversarial prompts are in the set to find exactly this kind of
  failure, and excusing them once the result is known would be moving the bar
  after the shot.
- **Keep the residual stream in fp32.** Measured and ineffective (above).

## Consequences

- The attention kernel takes the key bias and theta. It forms each key's
  rotated bias with an fp64 angle, as the RoPE kernel does, and keeps a tile's
  keys in fp32 in shared memory. Without a bias the kernel's arithmetic is
  unchanged: widening fp16 to fp32 is exact.
- **A key's position is its row in the cache.** The contiguous cache satisfies
  this. The paged cache (#14) must either keep it true or pass positions
  explicitly. A page moved by a tail swap (ADR-0007) keeps its tokens'
  positions, which are not its slot.
- Completing a key costs one fp64 `sincos` per element per tile load, repeated
  by every query head that reads the tile. #14 measured it on the 0.5B model
  at 1,090 keys: attention takes **3.3x as long in prefill** (60.5 ms a layer
  against 18.2 ms) and **4x in decode** (2.93 ms against 0.73 ms). Across 24
  layers, that is 70 ms per decoded token spent in attention alone, where 17 ms
  would do. The cost is the same with or without paging, so it belongs to this
  decision, and it does not change a single bit of output. The remedy is
  still open. The angle depends only on position and frequency, not on layer
  or head, so its cos and sin can be tabulated once per position in fp32 and
  shared by every layer. For the 0.5B model that is 256 bytes a position,
  against 12 KiB of cache. A per-layer table of rotated biases would also
  remove the arithmetic, but it would be as large as the cache it completes.
- **The quantised tiers inherit this.** INT8, INT4 and INT2 (#16, #17) quantise
  the bias-free key. ADR-0005's per-channel key quantisation keeps a zero point
  per channel, and the bias would have been most of that zero point. Removing
  it leaves the quantiser the part of the key that varies. This is expected to
  help the lower tiers, but it is unmeasured, and #16 should report it rather
  than assume it.
- The bf16 row above belongs in the paper. At the checkpoint's own precision
  the cached keys cannot represent this prompt at all. So "FP16" is not a
  lossless baseline for Qwen2.5's cache unless the bias is kept out of it. That
  is a statement about the top of the tier ladder that the thesis rests on.
