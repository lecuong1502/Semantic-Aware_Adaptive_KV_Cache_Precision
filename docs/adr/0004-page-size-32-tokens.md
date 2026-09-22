# Page size is 32 tokens, not vLLM's 16

A page holds 32 token positions of one layer's keys and values. vLLM's default
is 16, and this project compares directly against PagedAttention, so the
divergence needs a reason on record.

Two reasons. First, the needle-in-a-haystack evaluation (§5.4) works by
controlling which page the planted fact lands in and then observing what
happens when that page's precision tier is downgraded; a needle sentence of
roughly 20-30 tokens sits inside a single 32-token page but straddles two
16-token pages, and straddling muddies the cleanest experiment supporting RQ3.
Second, an importance score averaged over 32 tokens is less noisy than one
averaged over 16, which reduces how much the result depends on tuning the EWMA
`alpha` in §3.2.

The cost is accepted knowingly: coarser pages narrow the margin by which
semantic-aware allocation can beat uniform allocation, and that margin is the
project's central claim.

## Consequences

- `P` is a build-time parameter. Nothing may assume 32.
- Milestone 3 owes an ablation over `P ∈ {16, 32, 64}`. This is not optional —
  it is the evidence that 32 was chosen for the reasons above rather than
  chosen because it flattered the results.
