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

## Note from #17: what INT2 costs in quality, measured

INT2 is built. Its effective bits are 2.625 on Qwen2.5-1.5B, as stated above,
with the metadata 23.8% of a 5376-byte page. What it loses of the cache is
now measured (ADR-0005, note from #17). Over every full page of the golden
prompts on the 1.5B model, keys come back with 20.9% relative RMS error and
values with 59.5%: a value's signal-to-noise ratio is 4.5 dB.

That is INT2's quality cost, the other side of its 112 MiB. It is no
surprise: the same note shows each tier's error following its step exactly,
and INT2's step is 5 times INT4's and 85 times INT8's. It was implied when
this decision was taken; it is now a number.

It does not answer the objection above, nor change the answer to it. The
tier earns its place, if it does, through the allocator's freedom to send the
least important pages there. Whether a page at INT2 still carries what the
model needs from it is a question about the model's output, not about the
cache, and #18's perplexity per tier is where it is first measured.

---

## Note from #18: INT2 as a whole-cache tier fails, as the objection predicted

#18 measured INT2 end to end (ADR-0011, measurements). As the tier of a whole
cache it costs +19.7% perplexity on Qwen2.5-1.5B and +26.8% on 0.5B, which is
in line with KVQuant's 2-bit and no sign of a fault. Its generation, read,
is not coherent. The sentences stay grammatical on 1.5B, but it invents
facts and loops; on 0.5B it degenerates.

This does not settle the objection, and it was never going to. The tier was
kept for the allocator's freedom to send the *least important* pages to it.
A static cache sends every page there, including the ones attention needs
most. Whether a page an importance score picks for INT2 still carries what
the model needs from it is Milestone 2's question, and the allocation
experiments' to answer. What #18 adds is a floor: INT2 everywhere is not a
usable configuration, so any result that uses the tier must show it was
used selectively.
