# Contributing

This is a research project, and most of the rules below exist because breaking
them would quietly invalidate a measurement rather than break a build. Where a
rule has a reason, the reason is given — follow the reasoning, not the letter.

## Language

**Everything in the repository is written in English** — code, comments, commit
messages, issue titles and bodies, `CONTEXT.md`, ADRs, test names, benchmark
logs and the paper.

Conversation with the repository owner happens in Vietnamese. That is a separate
concern and never means a Vietnamese identifier, issue or document.

## Vocabulary

Read [`CONTEXT.md`](CONTEXT.md) before writing anything. Several words in this
domain mean more than one thing in the surrounding literature, and the glossary
picks one meaning for each.

The one that bites most often: the unit of KV cache memory is a **page**, and the
structure mapping it is the **page table**. vLLM calls these a *block* and a
*block table*. The paper notes the correspondence once; the codebase does not use
those names. `block_id` in a diff is a review comment.

If you need a concept the glossary does not have, that is a signal. Either you
are inventing language the project does not use — reconsider — or there is a real
gap, and the glossary should gain an entry in the same change.

## Architectural decisions

`docs/adr/` records decisions that are **hard to reverse**, **surprising without
context**, and **the result of a real trade-off**. All three must hold; otherwise
skip the ADR.

Each ADR records the rejected options too. That is most of their value — without
it, someone re-proposes the rejected option in six months, and nobody remembers
why it was rejected.

**If your change contradicts an ADR, say so explicitly in the pull request.** Do
not silently override it. An ADR being wrong is a normal outcome; an ADR being
quietly ignored is how a project loses track of its own reasoning.

Several ADRs are load-bearing for the thesis rather than for the code, and are
worth knowing before touching the relevant area:

- **[ADR-0002](docs/adr/0002-engine-owns-vram-torch-is-offline-only.md)** — never
  `import torch` in a process that runs the engine.
- **[ADR-0006](docs/adr/0006-correctness-gate-is-top1-logit-agreement.md)** — the
  correctness gate is logit agreement, not generated text.
- **[ADR-0007](docs/adr/0007-vmm-allocator-with-per-tier-address-ranges.md)** —
  reclaimed memory must reach the driver, not a private pool.

## Where tests go

Two seams. Both are architectural boundaries that exist anyway; neither was
invented for testing.

**Seam A — `microinfer.Engine`.** `forward(token_ids) -> logits`,
`generate(...)`, and `capture_hidden_states=True`. Model-level correctness is
tested here.

**Seam B — the `_microinfer` extension boundary.** Individual kernels, and the
`PagedKVCache` object. Kernel numerics and allocator behaviour are tested here.
The binding layer accepts and returns NumPy arrays at its test-facing surface,
handling host/device transfer internally, so a kernel test needs no manual buffer
management.

A good test asserts on what the code produces, not how. Kernel tests assert
numerical agreement with an independent reference; they do not assert tile sizes,
launch configurations or occupancy. Allocator tests assert on NVML-reported free
memory; they do not assert which granule was unmapped.

## The correctness gate

Three layers, with distinct jobs:

| Layer | Where | Tolerance | Blocks a merge? |
|---|---|---|---|
| Per-kernel vs float64 NumPy | Seam B | max rel err < 2e-3, mean < 2e-4 | **Yes** |
| Per-layer vs golden tensors | Seam A | cosine similarity > 0.999 | No — diagnostic |
| **Logits vs golden** | Seam A | **top-1 ≥ 99%, mean KL < 1e-3** | **Yes — this is the gate** |
| Greedy text vs HuggingFace | Seam A | 64 tokens, ≥ 8 of 10 prompts | No — smoke test |

**A failing greedy text comparison is not a blocker.** The engine calls cuBLAS
along a different accumulation path than HuggingFace, so bit-reproducibility is
not available. A 1e-4 logit difference at a near-tie flips an argmax and every
token after it diverges — the suite then goes red across the board while pointing
at nothing. Investigate it; do not gate on it.

The per-layer layer exists to answer *where*. When the gate goes red, run it
first; it names the first diverging layer.

## Things that are always wrong

- **Importing `torch` in the engine.** Its caching allocator retains freed device
  memory, so NVML would report that reservation as used memory, indistinguishable
  from the external contention this project exists to measure. A test asserts
  `torch` is absent from `sys.modules`.
- **Hardcoding a dimension.** `head_dim`, `num_kv_heads`, layer count, RoPE theta
  and page size `P` are parameters. The same kernels serve Qwen2.5-0.5B
  (`head_dim=64`) and Qwen2.5-1.5B (`head_dim=128`).
- **Hardcoding the VMM granule size.** Query it with
  `cuMemGetAllocationGranularity`. It is typically 2 MiB and the engine must not
  be silently wrong where it is not.
- **Caching a page's device address across an allocator operation.** Freeing a
  page swaps the tail page into the vacated slot, so addresses move. Resolve
  through the page table on every launch.
- **Quoting a compression ratio without metadata.** Scale metadata is a flat 1280
  bytes per page at every quantised tier. INT4 is 4.63 effective bits, not 4.
  Claiming "4x" is false.
- **Reporting a benchmark without its context.** Measurements come from a laptop
  whose memory and clocks depend on what else is running. Every entry records
  configuration, hardware, driver version and git commit.

## Working a ticket

Issues carry native GitHub blocking links. Any issue with no open blocker can be
started; the frontier is visible in the issue list.

**One ticket per context window.** Start a fresh session for each. Tickets are
written to be self-contained precisely so this works — a session carrying three
tickets' worth of history reasons worse than one starting clean.

Triage labels: `needs-triage`, `needs-info`, `ready-for-agent`, `ready-for-human`,
`wontfix`. A ticket produced by planning is already agent-ready and does not go
through triage; triage is for issues that arrive raw.

`ready-for-human` means it. Issue #4 corrects the author's own research notes,
and an agent should not silently rewrite someone's research record.

## Commits and pull requests

Branch from `main`; do not commit to it directly. Keep commits small enough that
a reviewer can follow one idea at a time.

The pull request template asks for gate results, not for a claim that tests pass.
Paste the numbers.
