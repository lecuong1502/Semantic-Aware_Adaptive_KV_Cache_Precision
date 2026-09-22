# KIVI: full-precision residual and per-channel group size

Answers to the two questions raised in issue #2. Both bear on decisions already
recorded in `docs/adr/`.

**Primary source.** Zirui Liu, Jiayi Yuan, Hongye Jin, Shaochen Zhong, Zhaozhuo
Xu, Vladimir Braverman, Beidi Chen, Xia Hu. *KIVI: A Tuning-Free Asymmetric 2bit
Quantization for KV Cache.* ICML 2024. arXiv:2402.02750.

**Provenance, stated plainly.** The paper body was read through the ar5iv HTML
rendering at `https://ar5iv.labs.arxiv.org/html/2402.02750`, not from the PDF.
Section, table and algorithm numbers below are as they appear in that rendering.
Two separate passes over that source agreed on every number, but both drew on the
same rendering, so this is consistency of extraction rather than independent
confirmation. **Before any number here is quoted in the thesis, check it against
the PDF.** `https://arxiv.org/html/2402.02750v3` returns 404; the ar5iv mirror was
the only HTML route available.

---

## (b) Group size — G = 32

> "the group size G in Algorithm 1 for quantization is set as 32 across all
> experiments" — Section 4.1 (Settings)

For the key cache, a group spans **32 consecutive token positions**, and within
that group each channel carries its own scale — the algorithm applies
`GroupQuant(X_Kg, dim=channel, numGroup=l//G)`, dividing the token axis into
`l/G` groups and quantising per channel inside each. The value cache is quantised
per token.

The paper ablates the group size (Section 4.2.3, Table 5), holding residual
length at 128, measuring exact match on GSM8K with Llama2-13B:

| Group size | GSM8K |
|---:|---:|
| **32** | **20.77** |
| 64 | 21.00 |
| 128 | 17.29 |

> "group sizes 32 and 64 yield similar results, whereas the performance
> significantly decreases when the group size reaches 128" — Section 4.2.3

### What this means for ADR-0004

**No conflict. ADR-0004 is supported.**

ADR-0004 sets the page at 32 tokens, and recorded that it was chosen on internal
reasoning — needle isolation and importance-score stability — with no external
evidence. KIVI independently arrived at 32 for the same structural quantity: the
group over which a per-channel key scale is computed *is* this project's page.

Two qualifications worth carrying into the thesis rather than celebrating:

- **The evidence supports 32 *or* 64, not 32 over 64.** KIVI's own reading is
  that the two are equivalent (20.77 versus 21.00), and 64 in fact scored
  marginally higher. What the table rules out is 128. So this is evidence that
  the page is not too coarse, not evidence that 32 is optimal.
- **The ablation over `P` promised by ADR-0004 is still owed.** KIVI measured
  GSM8K under static quantisation; this project measures allocation quality under
  runtime pressure, where page size also controls targeting granularity — a
  dimension KIVI's experiment does not exercise at all.

---

## (a) Full-precision residual — R = 128, fixed

> "the residual length R for key and value cache is set to 128" — Section 4.1

It is a **fixed token count, not a proportion**. Both caches keep a residual, but
they fill differently (Section 3.3, Algorithm 1):

- **Keys** accumulate: once the key residual reaches `R` tokens, it is quantised
  as a whole and concatenated to the already-quantised cache.
- **Values** use a queue: each new value is pushed, and once the queue reaches
  `R`, the oldest entry is popped, quantised per token, and concatenated.

That asymmetry follows from the granularities. A per-channel key scale cannot be
computed until a full group of tokens exists, so keys must be held back in
group-sized batches. A per-token value scale needs only one token, so values can
be quantised one at a time as they age out.

**A constraint the paper states explicitly:**

> "We note that R should be divisible by G."

R = 128 and G = 32 means the residual is exactly **four groups**.

The residual length ablation (Table 5, group size fixed at 32) shows it matters
far less than group size:

| Residual length | GSM8K |
|---:|---:|
| 32 | 20.62 |
| 64 | 19.86 |
| 96 | 20.55 |
| **128** | **20.77** |

The spread is 0.91 points and is not monotonic — 64 scores below 32. On this
evidence the residual length is close to a free parameter over this range, and
the default of 128 is not strongly justified by the ablation.

### What this raises for ADR-0005 — a decision, not a conflict

ADR-0005 establishes that a page still being filled **must** stay at FP16,
because a per-channel key scale cannot be computed over an incomplete page. That
is confirmed by KIVI's key-residual mechanism, which exists for exactly this
reason.

But it leaves a quantity open. The mechanical constraint requires only **one**
page — the page currently being written. KIVI holds **four** (R=128, G=32). The
difference is not mechanical; it is the policy-level recency floor, and KIVI's
choice is a data point for setting it.

Two things follow, and neither is decided here:

1. **The recency floor should be a whole number of pages.** KIVI's "R divisible
   by G" is not a stylistic preference — a partial group cannot be quantised
   per-channel at all. The same applies here by the same argument.
2. **How many pages is open.** KIVI's default of 4 is weakly supported by its
   own ablation, which shows a 0.91-point non-monotonic spread across 1 to 4
   groups. And KIVI's setting is static: it never restores precision. This
   project downgrades under pressure and upgrades when pressure passes, so the
   cost of too small a floor is recoverable here in a way it is not for KIVI.
   That argues the floor could be smaller, but it is an argument, not a
   measurement.

**Raised for decision:** does the recency floor become a build-time constant in
whole pages, and is its default 1 page (the mechanical minimum) or 4 (KIVI's
choice)? Per the ticket, no ADR is amended here.

---

## Not answered by this reading

- Whether KIVI's group size interacts with model architecture — all reported
  ablations use Llama2-13B on GSM8K, and neither the group size nor the residual
  length is varied across models or across `num_kv_heads`. This project runs
  Qwen2.5 with 2 KV heads; KIVI's Llama2-13B has 40. Whether G=32 transfers is
  untested by this evidence.
- **The paper has no limitations section.** Its conclusion notes only future work:
  "we will further optimize the implementation to reduce the overhead of
  quantization process during the prefill and decoding phase."
