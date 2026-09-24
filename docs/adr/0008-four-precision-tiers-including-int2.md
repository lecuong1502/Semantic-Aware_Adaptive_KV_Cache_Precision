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

---

## Note from #17: what INT2 costs, measured

INT2 is built. Its effective bits are 2.625 on Qwen2.5-1.5B, as stated above,
with the metadata 23.8% of a 5376-byte page. What it loses of the cache is
now measured too (ADR-0005, note from #17). Over every full page of the
golden prompts on the 1.5B model, keys come back with 20.9% relative RMS
error and values with 59.5%: a value's signal-to-noise ratio is 4.5 dB.

That is a second cost to set beside the 112 MiB above, and a larger one. The
answer recorded here, that the tier earns its place through the allocator's
freedom and not through bytes, assumed a page at INT2 still carries
something. Whether it does, for the pages an importance score would send
there, is a question about the model's output and not about the cache. It is
answered by #18's perplexity per tier, and by the allocation experiments
after it, not by this measurement.
