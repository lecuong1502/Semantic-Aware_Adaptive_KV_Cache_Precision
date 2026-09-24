# Asymmetric quantisation, keys per-channel and values per-token

Keys are quantised with a scale and zero-point per `(head, channel)`, computed
across the 32 token positions of a page. Values are quantised per
`(head, token)`. Both are asymmetric. This follows KIVI's finding that key
outliers concentrate along channels while value outliers do not, so a single
granularity cannot serve both.

The cost is a page whose two halves have different layouts: two quantise
kernels, two dequantise paths, and 1280 bytes of scale metadata per page
(28.9% overhead at INT4, giving roughly 4.6 effective bits).

The reason to pay it is comparability rather than raw quality. The central
experiment holds average compression equal and asks whether *allocating*
precision by importance beats allocating it uniformly. Both arms use this same
quantiser, so a weak quantiser would depress both arms and could mask the
effect being measured. The quantiser must be good enough that what remains is
the allocation policy.

## Consequences

- **The recency floor is load-bearing twice.** A per-channel key scale is
  computed over all 32 tokens of a page, so a page still being filled cannot be
  quantised per-channel at all. The open page must stay at FP16 as a mechanical
  requirement, independently of the policy argument in §3.3 that recent context
  is usually relevant. The paper should state both reasons; they are
  independent and they agree.
- **Verified against the primary source**: Liu, Yuan, Jin, Zhong, Xu,
  Braverman, Chen, Hu, *KIVI: A Tuning-Free Asymmetric 2bit Quantization for
  KV Cache*, arXiv:2402.02750. Keys per-channel, values per-token, asymmetric,
  at 2 bits; 2.6x lower peak memory including weights, 2.35-3.47x throughput.
- Two details remain unverified because they are in the body, not the abstract,
  and both bear on decisions already taken:
  **(a)** how many recent tokens KIVI holds at full precision, which is the
  same mechanism as this project's recency floor; and
  **(b)** the group size KIVI uses along the token axis for per-channel key
  quantisation — **that group is this project's page**. If KIVI found 32
  optimal, it is independent support for ADR-0004; if it found 64 or 128, that
  is evidence against it. Filed as a research issue.
- INT8 gives roughly 8.6 effective bits and INT4 roughly 4.6, once metadata is
  counted. Compression ratios reported in the paper must count metadata;
  quoting "4x" for INT4 would be false.

---

## Note from #16: the quantised page, and what INT8 loses

**The layout** is written down once, in `src/core_kernels/include/microinfer/quant.h`,
and every quantised tier uses it; INT4 and INT2 change only the width of a
code. From byte 0, with `W = kv_heads * head_dim`:

| region | size | indexed by |
|---|---|---|
| key codes | `P * W` codes | row `t` is token `t`, element `h*D + d` |
| value codes | `P * W` codes | the same |
| key scales | `W` fp16 | `h*D + d` |
| key zero-points | `W` fp16 | `h*D + d` |
| value scales | `P * kv_heads` fp16 | `t*kv_heads + h` |
| value zero-points | `P * kv_heads` fp16 | `t*kv_heads + h` |

The code rows are the FP16 page's rows, codes in place of halves. Below 8
bits, `8/bits` consecutive codes share a byte, the first in the lowest bits.

**The quantiser.** A group's zero-point is its minimum, which an fp16 input
holds exactly. Its scale is its range over `2^bits - 1` steps, rounded *up* to
fp16. Rounding to nearest was tried first, and the round trip on real keys
found its flaw: a channel of the 1.5B model whose range was 1.26e-4 had a
scale below fp16's smallest normal, rounded an eighth short, and lost 8 steps
at its maximum to the clamp. Rounded up, the last level always reaches the
maximum and nothing is clamped.

**The metadata is 1280 bytes on Qwen2.5-1.5B**, as this ADR assumed, and 768
on 0.5B. The effective bits follow, and the layout reports them:

| model | INT8 | INT4 | INT2 |
|---|---:|---:|---:|
| Qwen2.5-1.5B | 8.625 | 4.625 | 2.625 |
| Qwen2.5-0.5B | 8.750 | 4.750 | 2.750 |

**A correction.** The overhead figure above, "28.9% at INT4", does not
follow from 1280 bytes. INT4 codes on the 1.5B model take 8192 bytes, so the
metadata adds 15.6% to them and is 13.5% of the page. The effective bits
stated beside it, about 4.6, were right, and they are the figure to quote.

**The open page is refused, not assumed.** `quantise_page` raises `OpenPage`
for a page with fewer than `P` positions.

**What INT8 loses**, measured by `tools/quant_roundtrip.py` on every full
page of every layer of the golden prompts, against the fp16 values the cache
held (entries `53a15bcf` and `0ff5e24f`, at b62e920):

| model | pages | keys: relative RMS, SNR | values: relative RMS, SNR |
|---|---:|---:|---:|
| Qwen2.5-0.5B | 4,056 | 0.24%, 52.4 dB | 0.62%, 44.1 dB |
| Qwen2.5-1.5B | 4,732 | 0.25%, 52.1 dB | 0.70%, 43.1 dB |

The mean error is 0.23 to 0.25 of a step, what rounding to the nearest
level costs a uniformly spread value. No element is more than one step out,
and past half a step only where the fp16 output cannot resolve one.
Keys and values are about as many steps out on average, so values' larger
relative error means their steps are larger relative to their size: a value
group's range is wider, for the size of what is in it, than a key channel's.
Why has not been measured.
