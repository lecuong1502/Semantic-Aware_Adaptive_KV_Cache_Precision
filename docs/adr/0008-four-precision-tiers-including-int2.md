# Four precision tiers: FP16, INT8, INT4, INT2

The tier ladder includes INT2, matching the bit-width KIVI (arXiv:2402.02750)
showed to be viable with this quantisation scheme.

The reason is the shape of RQ3, not the byte count. RQ3 asks whether allocating
precision by importance beats allocating it uniformly *at equal average
compression*. A four-tier allocator has strictly more freedom than a three-tier
one at any given average, so the gap between a semantic-aware allocation and a
uniform one has more room to appear. Adding a tier widens the very margin the
thesis is measuring. A secondary benefit is that under RED pressure, a page
already at INT4 can drop once more instead of forcing a PCIe offload.

## The argument against, recorded deliberately

On the headline configuration (Qwen2.5-1.5B at 32K) INT2 reclaims only 112 MiB
beyond INT4 — 1.8% of a 6141 MiB card. Weights are a fixed 2.88 GiB and an
all-INT4 cache is 259 MiB, so roughly 2.5 GiB stays free; for that to be
exhausted, external contention must take 2.5 GiB, at which point 112 MiB more
does not save the session either. **The scenario where INT2 helps is close to
the scenario where it is insufficient.** Scale metadata is a flat 1280 bytes
per page at every tier, so at INT2 it is 23.8% of the page and the tier
delivers 2.63 effective bits rather than 2.

This objection was raised, understood, and overruled. It is recorded because a
reviewer will construct it independently, and the answer above — that the tier
earns its place through policy freedom, not through reclaimed bytes — is the
one to give.

## Consequences

- Every results table carries a fourth tier; every ablation grid multiplies.
- The INT2 quantise and dequantise kernels are Milestone 0 work, alongside
  INT8 and INT4.
- Compression ratios must be reported with metadata counted. INT2 is 2.63
  effective bits here, and claiming "8x" would be false.
