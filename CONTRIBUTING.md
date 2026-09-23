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
| Per-kernel vs float64 NumPy | Seam B | max < 4 ulp, mean < 1 ulp of the output dtype | **Yes** |
| Per-layer vs golden tensors | Seam A | cosine similarity > 0.999 | No — diagnostic |
| **Logits vs golden** | Seam A | **top-1 ≥ 99%, mean KL < 1e-3** | **Yes — this is the gate** |
| Greedy text vs HuggingFace | Seam A | 64 tokens, ≥ 8 of 10 prompts | No — smoke test |

**A failing greedy text comparison is not a blocker.** The engine calls cuBLAS
along a different accumulation path than HuggingFace, so bit-reproducibility is
not available. A 1e-4 logit difference at a near-tie flips an argmax and every
token after it diverges — the suite then goes red across the board while pointing
at nothing. Investigate it; do not gate on it.

**A kernel whose output is a sum is measured against its terms where they
cancel.** Projections, RoPE and attention add terms that can cancel to near
zero, where a relative error has no bound however correct the kernel is. Their
tests floor the denominator per output (ADR-0006, amendment on cancellation).
A floor comes from `tests/ulp_gate.py`: `accumulation_floor` for a plain sum,
or `floor_from_bound` with a bound that names each error source it admits, as
attention's does for its scores. A new bound is calibrated against measured
error before it is used, and the ADR records the share of outputs it judges.
On ordinary inputs that share has been under 14%. A test in which a floor
judges most outputs, as attention's stability test does, is not a precision
test, and its docstring says so.

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
- **Quoting a per-kernel tolerance as an absolute number.** ADR-0006 states them
  in ulps of the kernel's *output* format, because a bound near a format's noise
  floor measures the format rather than the implementation. For fp16 that is
  max < 1.953e-3 and mean < 4.883e-4; derive it, do not hardcode it.
- **Reporting a benchmark without its context.** Measurements come from a laptop
  whose memory and clocks depend on what else is running. Every entry records
  configuration, hardware, driver version and git commit.

## Golden reference tensors

The engine is judged against a reference generated by `tools/gen_golden.py` —
**the only file in this repository that may import torch** (ADR-0002). It runs
in its own environment, which must never sit beside the engine:

```bash
python3 -m venv .venv-golden
.venv-golden/bin/pip install torch==2.14.0 --index-url https://download.pytorch.org/whl/cpu
.venv-golden/bin/pip install transformers==5.17.0 safetensors numpy
.venv-golden/bin/python tools/gen_golden.py --model qwen2.5-0.5b-instruct
```

**The CPU-only wheel is the point, not a convenience.** A torch that can reach
the GPU can take video memory, and VRAM taken by the reference generator is
indistinguishable at the NVML reading from the external contention RQ2 exists to
measure. `torch.cuda.is_available()` must be `False` in that environment.

Three properties of the output are load-bearing:

- **It is generated in fp32, never in the checkpoint's bfloat16.** bf16's
  relative ulp is `2^-8`, eight times coarser than the fp16 the engine stores. A
  bf16 reference would carry more error than the implementation it judges and
  would exceed ADR-0006's own 4-ulp bound before the engine rounded once.
- **Its KL term is a sample.** Top-1 agreement is exact — an argmax is stored at
  every position. Mean KL needs whole distributions, which are 608 KiB each, so
  full logits are kept at 16 positions per prompt and the gate's KL is the mean
  over those. ADR-0006's second amendment records that the sample size is a
  storage budget rather than a measurement, and makes validating it part of #12.
- **It is not committed.** 424 MiB across both models, and git keeps every
  regeneration forever. `GoldenSet` refuses with the command that would make
  one, and tests skip rather than fail. The manifest hashes the `config.json`
  it was generated from, so a stale reference is caught rather than mistaken for
  a kernel bug.

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

## Formatting and the commit hook

Install once, after cloning:

```bash
pip install -e ".[dev]"
pre-commit install
```

`.pre-commit-config.yaml` then runs on every `git commit`: whitespace hygiene,
`clang-format` over C++ and CUDA, and two project-specific guards described
below. To format by hand without committing:

```bash
clang-format -i $(git ls-files '*.cu' '*.cpp' '*.h')
```

**Tests are deliberately not in the hook.** They need the CUDA extension built
and a visible GPU, so a hook that ran them would block a commit made from a
machine without one — and a commit hook is only worth having if it is cheap
enough that nobody is tempted to skip it. Run `pytest` before pushing.

### The two guards

`tools/check_engine_invariants.py` enforces the two rules above that a reader
would otherwise have to remember:

- **ADR-0002** — `torch` and `transformers` may not be imported under `src/` or
  `tests/`, and may not be pinned in `requirements.txt`. They belong to the
  offline golden tensor generator under `tools/`, which runs in its own
  environment.
- **CONTEXT.md** — `block table`, `block_id` and their variants are rejected
  under `src/`, `tests/` and `tools/`. CUDA's own `blockIdx`, `blockDim` and
  thread-block vocabulary is a different word for a different thing and is not
  matched.

When the forbidden text is the subject rather than the sin — quoting vLLM's
terminology, say — end the line with `invariant-ok`.

`docs_research/` is excluded from the whitespace hooks. It is the author's
research record and a hook does not get to rewrite it.

The style was chosen to agree with what the editor already in use produces —
Allman braces, two-space indent, indented namespaces, pointers bound right —
so that saving a file does not churn the diff. It differs in one respect: the
editor enforces no column limit, so indenting a block silently pushed comments
past 80 columns. The config rewraps them.

Python has no formatter configured. Add one when it starts to matter.

## Commits and pull requests

Branch from `main`; do not commit to it directly. Keep commits small enough that
a reviewer can follow one idea at a time.

The pull request template asks for gate results, not for a claim that tests pass.
Paste the numbers.
