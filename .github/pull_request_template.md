## What this changes

<!-- The behaviour this makes work, from a user's perspective. Not a file-by-file list. -->

Closes #

## Correctness gate

<!-- Paste the numbers. "Tests pass" is not a result. Delete rows that do not apply. -->

| Check | Result |
|---|---|
| Top-1 logit agreement (≥ 99%) | |
| Mean KL(HF \|\| ours) (< 1e-3) | |
| Per-kernel max relative error (< 2e-3) | |
| Per-kernel mean relative error (< 2e-4) | |
| Per-layer cosine similarity — first layer below 0.999 | |
| Greedy 64-token match (smoke, non-blocking) | |

If the greedy smoke test is red, say what you found. It is not a blocker — the
accumulation path differs from HuggingFace's and an argmax flip at a near-tie
diverges everything after it — but an unexplained red is worth a sentence.

## If this touched the allocator

<!-- Delete this section if it did not. -->

- [ ] NVML-reported free memory rises by at least the granule size when a granule empties
- [ ] Granule size was queried, not assumed
- [ ] Page contents survive a tail swap
- [ ] No device address is cached across an allocator operation

The contract is not "a page was freed". It is "NVML-reported free memory went
up". Paste the measured delta:

## Checklist

- [ ] Vocabulary matches `CONTEXT.md` — `page` and `page table`, never `block`
- [ ] No `import torch` reachable from the engine
- [ ] No hardcoded `head_dim`, `num_kv_heads`, `P`, RoPE theta or granule size
- [ ] Any compression ratio quoted counts scale metadata
- [ ] New measurements are recorded in the benchmark log with commit and hardware
- [ ] Tests assert on behaviour, not on tile sizes, launch configuration or occupancy

## ADR impact

- [ ] This change is consistent with every ADR in the area
- [ ] **Or:** it contradicts one — named here, with the reason it is worth reopening

<!-- Contradicting an ADR is fine. Doing it silently is not. -->

## Notes for the reviewer

<!-- Anything you are unsure about, or deliberately deferred. -->
