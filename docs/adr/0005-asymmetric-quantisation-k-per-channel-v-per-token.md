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
